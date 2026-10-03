"""Residual-stream drift monitor evaluation pipeline — one command, every (model x metric), the entire history.

Loads ONE model at a time, then runs the full ground-up chain for each requested
metric version:

    stage0 layer-select -> stage1 threshold -> stage2 ablation -> stage_labeling
        -> stage3 robustness -> stage4 judge -> stage5 aggregate

and repeats the whole thing for each model in --model (the second-model
generalization study). Every (model, metric) run writes a self-contained directory
runs/<timestamp>-<model>-<metric>/ with a manifest.json (git SHA + full config
snapshot + per-stage status/timings/artifacts), all intermediate CSV/JSONL, plots,
and a pipeline.log. The cross-everything head-to-head lands in runs/master_results.csv.

Usage (run from the repository root, on a GPU machine with the packages in requirements.txt):

    python auto_pipeline/run_all.py                       # all models x all metrics
    python auto_pipeline/run_all.py --metric both         # l2 + sinkhorn
    python auto_pipeline/run_all.py --metric l2 --model Qwen/Qwen2.5-7B-Instruct
    python auto_pipeline/run_all.py --stages stage0_layer_select,stage1_threshold_sweep
    python auto_pipeline/run_all.py --metric l2 --resume runs/20260621-...-l2
    python auto_pipeline/run_all.py --profile large       # N>=1000 for significance
    python auto_pipeline/run_all.py --profile smoke --metric l2   # wiring check
    python auto_pipeline/run_all.py --gpus 0,1            # parallel across 2 GPUs
    python auto_pipeline/run_all.py --gpus 2 --profile large  # all 2 GPUs, heavy run

Profiles (set sample sizes; see apply_profile):
    smoke   tiny, for a wiring check (minutes)
    default the config.py values (N=100 ablation)
    large   N>=1000 ablation for statistical significance (heavy — needs real compute)

Parallel mode (--gpus): distributes (model x dataset x metric) work units across
GPUs round-robin. Each GPU runs its own subprocess with CUDA_VISIBLE_DEVICES set.
Results land in the shared runs/ directory just like sequential mode.

Nothing executes on import; only under __main__.
"""

import argparse
import multiprocessing as mp
import os
import sys

# allow `python auto_pipeline/run_all.py` from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import runtime
import stage0_layer_select
import stage1_threshold_sweep
import stage2_ablation
import stage_labeling
import stage_threshold_downstream
import stage3_robustness
import stage4_judge
import stage_results
import stage5_aggregate
from drift_metrics import METRICS

ALL_STAGES = [
    "stage0_layer_select",
    "stage1_threshold_sweep",
    "stage2_ablation",
    "stage_labeling",
    "stage_threshold_downstream",
    "stage3_robustness",
    "stage4_judge",
    "stage_results",
    "stage5_aggregate",
]


def resolve_metrics(arg):
    if arg in (None, "", "all"):
        return list(config.METRICS)
    if arg == "both":
        return ["l2", "sinkhorn"]
    out = []
    for m in arg.split(","):
        m = m.strip()
        if m not in METRICS:
            raise SystemExit(f"unknown metric '{m}'. choices: {list(METRICS)}")
        out.append(m)
    return out


def resolve_models(arg):
    if arg in (None, "", "all"):
        return list(config.MODELS)
    return [m.strip() for m in arg.split(",") if m.strip()]


def resolve_datasets(arg):
    if arg in (None, "", "all"):
        return list(config.TARGET_DATASETS)
    return [d.strip() for d in arg.split(",") if d.strip()]


def apply_profile(cfg, profile):
    if profile == "smoke":
        cfg.LAYER_BAND = [22, 24]
        cfg.LAYER_SELECT_SAMPLES = 2
        cfg.LAYER_SELECT_CONTROL_SAMPLES = 2
        cfg.Z_GRID = [10.0, 12.5]
        cfg.ABLATION_SAMPLES = 3
        cfg.ROBUSTNESS_SEEDS = [42, 43]
        cfg.ROBUSTNESS_SAMPLES = 2
        cfg.MAX_NEW_TOKENS_DETECTION = 60
        cfg.MAX_NEW_TOKENS_RECOVERY = 40
    elif profile == "large":
        # Statistical-significance run . HEAVY: detection is
        # O(n^2) per token and the model runs every token, so this is the part
        # that genuinely needs rented compute. Scale these to taste / budget.
        cfg.LAYER_SELECT_SAMPLES = 100
        cfg.LAYER_SELECT_CONTROL_SAMPLES = 100
        cfg.ABLATION_SAMPLES = 1000
        cfg.ROBUSTNESS_SAMPLES = 200
    # "default" leaves config.py values untouched


def run_metric(model, model_id, dataset, metric_name, cfg, stages, resume_dir):
    metric = METRICS[metric_name]
    ctx = runtime.RunContext(metric_name, cfg, run_dir=resume_dir, model_id=model_id,
                             dataset=dataset)
    ctx.log(f"### model={model_id} dataset={dataset} metric={metric_name} "
            f"control={metric.is_control} run_dir={ctx.dir}")

    def want(name):
        return name in stages and not ctx.stage_done(name)

    layer = ctx.manifest.get("selected_layer")
    z = ctx.manifest.get("selected_z")
    traces_path = ctx.artifact("layer_traces.csv")
    ablation_path = ctx.artifact("ablation.jsonl")

    if metric.is_control:
        layer = layer or (config.LAYER_BAND[len(config.LAYER_BAND) // 2])
        z = z if z is not None else config.LAYER_SELECT_REF_Z
        ctx.set_selection(layer=layer, z=z)
        ctx.log(f"control metric: skipping layer/threshold selection "
                f"(nominal layer={layer}; triggers at random position)")
    else:
        if want("stage0_layer_select"):
            ctx.start_stage("stage0_layer_select")
            layer, traces_path = stage0_layer_select.run(model, metric, ctx, cfg)
        else:
            layer = ctx.manifest.get("selected_layer", layer)
        if want("stage1_threshold_sweep"):
            if not os.path.exists(traces_path):
                raise SystemExit(
                    f"stage1 needs {traces_path} (run stage0 first; it was skipped/absent)")
            ctx.start_stage("stage1_threshold_sweep")
            z = stage1_threshold_sweep.run(metric, ctx, cfg, layer, traces_path)
        else:
            z = ctx.manifest.get("selected_z", z)

    # downstream stages can't run without a chosen layer/threshold
    _need_sel = ("stage2_ablation", "stage_threshold_downstream", "stage3_robustness")
    if any(want(s) for s in _need_sel) and not metric.is_control:
        if layer is None or z is None:
            raise SystemExit(
                f"metric={metric_name}: layer/z not selected — run stage0+stage1 "
                f"before stage2/3 (got layer={layer}, z={z})")

    if want("stage2_ablation"):
        ctx.start_stage("stage2_ablation")
        ablation_path = stage2_ablation.run(model, metric_name, ctx, cfg, layer, z)

    if want("stage_labeling"):
        ctx.start_stage("stage_labeling")
        stage_labeling.run(ctx, cfg)

    if want("stage_threshold_downstream"):
        ctx.start_stage("stage_threshold_downstream")
        stage_threshold_downstream.run(model, metric_name, ctx, cfg, layer)

    if want("stage3_robustness"):
        ctx.start_stage("stage3_robustness")
        stage3_robustness.run(model, metric_name, ctx, cfg, layer, z)

    if want("stage4_judge"):
        ctx.start_stage("stage4_judge")
        stage4_judge.run(model, ctx, cfg, ablation_path)

    if want("stage_results"):
        ctx.start_stage("stage_results")
        stage_results.run(ctx, cfg)

    if want("stage5_aggregate"):
        ctx.start_stage("stage5_aggregate")
        stage5_aggregate.run(ctx, cfg)

    ctx.log(f"### model={model_id} dataset={dataset} metric={metric_name} DONE -> {ctx.dir}")
    return ctx.dir


# ---------------------------------------------------------------------------
# Parallel GPU worker
# ---------------------------------------------------------------------------
def _worker(gpu_id, profile, work_units, stages_str, resume_dir):
    """Run a subset of (model, dataset, metric) work units on one GPU.

    Called in a spawn subprocess. All module-level names (config, runtime,
    apply_profile, run_metric) are available via globals() because the
    subprocess re-imports this file as __mp_main__.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        apply_profile(config, profile)
        _stages = [s.strip() for s in stages_str.split(",") if s.strip()]

        done = []
        for model_id, dataset, metric_name in work_units:
            model = runtime.load_model(config, model_id)
            run_dir = run_metric(model, model_id, dataset, metric_name, config, _stages,
                                 resume_dir)
            done.append(run_dir)
            if len(work_units) > 1:
                runtime._MODEL = None
                runtime._MODEL_ID = None

        for d in done:
            print(f"[GPU {gpu_id}]  {d}")
        return done
    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id}] CRASHED: {e}")
        traceback.print_exc()
        raise


def _run_parallel(gpu_ids, models, datasets, metrics, stages, args):
    """Distribute (model x dataset x metric) across GPUs round-robin."""
    work_units = [(m, d, me)
                  for m in models
                  for d in datasets
                  for me in metrics]
    if not work_units:
        print("No work units to distribute.")
        return

    gpu_work = {g: [] for g in gpu_ids}
    for i, wu in enumerate(work_units):
        gpu_work[gpu_ids[i % len(gpu_ids)]].append(wu)

    for gid, units in gpu_work.items():
        short = ", ".join(m.split("/")[-1] for m, _, _ in units)
        print(f"GPU {gid}: {len(units)} work unit(s)  [{short}]")

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, units in gpu_work.items():
        p = ctx.Process(target=_worker, args=(
            gpu_id, args.profile, units, args.stages, args.resume,
        ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    print("\n=== all GPU workers finished ===")


def _parse_gpus(raw):
    """Parse --gpus value into a list of GPU indices."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1 and parts[0].isdigit():
        n = int(parts[0])
        return list(range(n))
    if all(p.isdigit() for p in parts):
        return [int(p) for p in parts]
    raise SystemExit(f"invalid --gpus '{raw}'. use e.g. '0,1' or '2' (count).")


def main():
    ap = argparse.ArgumentParser(description="Residual-stream drift monitor evaluation pipeline")
    ap.add_argument("--model", default="all",
                    help="model id(s), comma-separated, or 'all' (config.MODELS)")
    ap.add_argument("--dataset", default="all",
                    help="target dataset(s), comma-separated, or 'all' (config.TARGET_DATASETS)")
    ap.add_argument("--metric", default="all",
                    help="metric(s): all | both | l2,sinkhorn,cosine,residual_norm,random")
    ap.add_argument("--stages", default=",".join(ALL_STAGES),
                    help="comma-separated stage names (default: all)")
    ap.add_argument("--profile", default="default", choices=["smoke", "default", "large"])
    ap.add_argument("--resume", default=None,
                    help="existing run dir to resume (single model + single metric only)")
    ap.add_argument("--gpus", default=None,
                    help="GPU indices for parallel, e.g. '0,1' or count '2'")
    args = ap.parse_args()

    cfg = config
    apply_profile(cfg, args.profile)

    models = resolve_models(args.model)
    datasets = resolve_datasets(args.dataset)
    metrics = resolve_metrics(args.metric)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ALL_STAGES:
            raise SystemExit(f"unknown stage '{s}'. choices: {ALL_STAGES}")
    for d in datasets:
        if d not in runtime.DATASET_ADAPTERS:
            raise SystemExit(f"unknown dataset '{d}'. choices: {list(runtime.DATASET_ADAPTERS)}")

    if args.resume and (len(models) != 1 or len(datasets) != 1 or len(metrics) != 1):
        raise SystemExit("--resume requires exactly one --model, --dataset and --metric")
    if args.gpus and args.resume:
        raise SystemExit("--resume and --gpus are mutually exclusive")

    os.makedirs(cfg.RUNS_DIR, exist_ok=True)

    if args.gpus:
        gpu_ids = _parse_gpus(args.gpus)
        _run_parallel(gpu_ids, models, datasets, metrics, stages, args)
    else:
        done = []
        for model_id in models:
            model = runtime.load_model(cfg, model_id)
            for dataset in datasets:
                for metric_name in metrics:
                    run_dir = run_metric(model, model_id, dataset, metric_name, cfg, stages,
                                         args.resume if args.resume else None)
                    done.append(run_dir)

        print("\n=== pipeline complete ===")
        for d in done:
            print(f"  {d}")
        print(f"results table:  {os.path.join(cfg.RUNS_DIR, 'results_table.csv')}")
        print(f"metric rollup:  {os.path.join(cfg.RUNS_DIR, 'master_results.csv')}")


if __name__ == "__main__":
    main()
