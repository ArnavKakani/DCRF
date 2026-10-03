"""Stage 2 — detection + recovery ablation (with diagnostic trajectories).

For each target problem we run a full detection pass and record, for EVERY problem
(triggered or not), a trajectory row: max_z, whether it triggered, the trigger
token, and whether the un-intervened baseline answer matched the GSM8K gold. Those
rows are what the labeling stage needs to separate formatting collapse from genuine
reasoning error and to compute AUC / false-trigger rate.

For triggered problems we additionally generate the four recovery paths from the
token state at the trigger and write the legacy-schema ablation record (now also
carrying gold, baseline_correct, and trigger_token).

The random control skips detection and triggers at a uniform random token.

Two outputs:
    trajectories.jsonl   one row per problem (diagnostic; drives the labeling stage)
    ablation.jsonl       one row per triggered problem (the four recovery paths)
"""

import json

import numpy as np

import runtime
from drift_metrics import METRICS


def _baseline_text(model, tokens, prompt):
    prompt_len = runtime._to_input_tokens(model, prompt).shape[1]
    return model.tokenizer.decode(tokens[0, prompt_len:])


def run_ablation(model, metric, layer, z, cfg, seed, n_samples, out_path,
                 traj_path=None, spike_path=None, ctx=None, dataset="gsm8k"):
    """Run the ablation for one seed.

    Appends recovery records to out_path and (if traj_path) diagnostic rows to
    traj_path. If spike_path is given, saves up to FIGURE_TRACE_LIMIT per-token
    Z-traces of triggered cases for the Fig-1 spike overlay. Returns trigger stats.
    """
    items = runtime.load_target_items(cfg, n_samples, dataset=dataset, model=model)
    rng = np.random.default_rng(seed)

    triggered = 0
    saved_traces = 0
    fout = open(out_path, "a")
    ftraj = open(traj_path, "a") if traj_path else None
    fspike = open(spike_path, "a") if spike_path else None
    try:
        for qi, item in enumerate(items):
            prompt, question, gold = item["prompt"], item["question"], item["gold"]
            try:
                if metric.is_control:
                    tokens = runtime.sample_trajectory(model, prompt, seed, cfg)
                    prompt_len = runtime._to_input_tokens(model, prompt).shape[1]
                    cut = runtime.random_trigger_index(prompt_len, tokens.shape[1], cfg, rng)
                    if cut is None:
                        continue
                    tokens_at_trigger = tokens[:, :cut]
                    spike_idx = cut - prompt_len
                    trig_z, max_z = None, None
                    trig_tok = model.tokenizer.decode(tokens[0, cut:cut + 1])
                    did_trigger = True
                    final_tokens = tokens
                    brk = None
                    det = None
                else:
                    want_trace = fspike is not None and saved_traces < cfg.FIGURE_TRACE_LIMIT
                    det = runtime.detect(model, prompt, metric, layer, z, seed, cfg,
                                         full_trajectory=True, collect_trace=want_trace)
                    did_trigger = det["triggered"]
                    spike_idx = det["spike_idx"]
                    trig_z, max_z = det["trigger_z"], det["max_z"]
                    trig_tok = det["trigger_token"]
                    tokens_at_trigger = det["tokens_at_trigger"]
                    final_tokens = det["tokens"]
                    brk = det["break_idx"]

                if (fspike is not None and det is not None and did_trigger
                        and saved_traces < cfg.FIGURE_TRACE_LIMIT and det["z_trace"]):
                    fspike.write(json.dumps({
                        "question": question, "spike_idx": spike_idx,
                        "break_idx": det["break_idx"], "z_threshold": z,
                        "z_trace": det["z_trace"],
                    }) + "\n")
                    fspike.flush()
                    saved_traces += 1

                baseline_text = _baseline_text(model, final_tokens, prompt)
                baseline_answer = runtime.extract_final_number(baseline_text)
                baseline_correct = runtime.answers_match(baseline_answer, gold)

                if ftraj:
                    ftraj.write(json.dumps({
                        "uid": qi,
                        "metric": metric.name, "layer": layer, "z": z, "seed": seed,
                        "question": question, "gold": gold,
                        "max_z": max_z, "triggered": bool(did_trigger),
                        "spike_idx": spike_idx, "trigger_token": trig_tok,
                        "baseline_answer": baseline_answer,
                        "baseline_correct": bool(baseline_correct),
                        # store the sampled text so the primary correctness label
                        # can be audited post hoc (it could not be for the June runs)
                        "baseline_text": baseline_text,
                    }) + "\n")
                    ftraj.flush()

                if not did_trigger:
                    continue

                paths = runtime.recovery_paths(model, tokens_at_trigger, cfg)
                fout.write(json.dumps({
                    "uid": qi,
                    "metric": metric.name, "layer": layer, "z": z, "seed": seed,
                    "question": question, "gold": gold,
                    "spike_idx": spike_idx, "break_idx": brk,
                    "trigger_z": trig_z, "trigger_token": trig_tok,
                    "baseline_answer": baseline_answer, "baseline_correct": bool(baseline_correct),
                    **paths,
                }) + "\n")
                fout.flush()
                triggered += 1
            except Exception as e:
                if ctx:
                    ctx.log(f"  [warn] problem {qi} failed: {e}")
    finally:
        fout.close()
        if ftraj:
            ftraj.close()
        if fspike:
            fspike.close()

    return {"n_samples": n_samples, "triggered": triggered,
            "trigger_rate": triggered / max(n_samples, 1)}


def run(model, metric_name, ctx, cfg, layer, z):
    metric = METRICS[metric_name]
    out_path = ctx.artifact("ablation.jsonl")
    traj_path = ctx.artifact("trajectories.jsonl")
    spike_path = ctx.artifact("spike_traces.jsonl")
    open(out_path, "w").close()
    open(traj_path, "w").close()
    open(spike_path, "w").close()
    ctx.log(f"ablation: metric={metric_name} layer={layer} z={z} "
            f"n={cfg.ABLATION_SAMPLES} seed={cfg.BASE_SEED}")
    stats = run_ablation(model, metric, layer, z, cfg, cfg.BASE_SEED,
                         cfg.ABLATION_SAMPLES, out_path, traj_path=traj_path,
                         spike_path=spike_path, ctx=ctx, dataset=ctx.dataset)
    ctx.log(f"ablation: {stats['triggered']}/{stats['n_samples']} triggered "
            f"(rate={stats['trigger_rate']:.2f})")
    ctx.finish_stage("stage2_ablation", artifacts=[out_path, traj_path, spike_path], **stats)
    return out_path
