"""Labeling stage — separate formatting collapse from genuine reasoning error.

Separates the two causes of a trigger: the detector
can fire when the model breaks its chat template, not only when it reasons
wrong. Here we post-process Stage 2's trajectories (no new model runs) to answer,
with numbers:

  * Does a drift spike PREDICT a wrong final answer?            -> AUC(max_z vs wrong)
  * Of the triggers, how many are formatting vs genuine errors? -> formatting split
  * How often does it false-alarm on already-correct problems?  -> false-trigger rate
  * On the GENUINE-error subset, does recovery flip wrong->right?-> per-path flip rate

"Genuine reasoning error" = the un-intervened baseline answer is wrong AND the
trigger is not a chat-template / role / end-of-text token. Everything is reported
on that subset, so the headline number is no longer inflated by formatting glitches.

Inputs (from the run dir): trajectories.jsonl, ablation.jsonl.
Output: labeling.csv (per-path recovery on the genuine subset) + a summary block
in the manifest.
"""

import csv
import json

import runtime
from drift_metrics import METRICS
from failure_typology import FAILURE_TYPES, classify_failure

PATHS = ["path_a_baseline", "path_b_full_cure", "path_c_prompt_only", "path_d_rewind_only"]


def _load_jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out


def _auc(scores, labels):
    """AUC of `scores` predicting label==1, via the Mann-Whitney U statistic."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _excerpt(text, n=600):
    text = (text or "").strip().replace("\n\n", "\n")
    return text if len(text) <= n else text[:n] + " …[truncated]"


def _write_examples(ctx, genuine, k=8):
    """Write up to k before/after cases: baseline failure vs recovery (Path B)."""
    # surface the ones Path B fixed first
    def fixed(r):
        return runtime.answers_match(runtime.extract_final_number(r.get("path_b_full_cure", "")),
                                     r.get("gold"))
    ordered = sorted(genuine, key=lambda r: not fixed(r))[:k]
    lines = [f"# Before/after examples — {ctx.metric} (genuine reasoning errors)\n",
             "Baseline (Path A) is the un-intervened continuation that got the answer "
             "wrong; Path B is rewind+reprompt from the latent spike.\n"]
    for i, r in enumerate(ordered, 1):
        lines += [
            f"\n## Example {i}  (spike@{r.get('spike_idx')}, "
            f"text-break@{r.get('break_idx', '—')}, gold={r.get('gold')}, "
            f"{'RECOVERED' if fixed(r) else 'not recovered'})",
            f"\n**Question:** {r.get('question', '')}\n",
            f"\n**Baseline (failed):**\n\n> {_excerpt(r.get('path_a_baseline'))}\n",
            f"\n**Path B:**\n\n> {_excerpt(r.get('path_b_full_cure'))}\n",
        ]
    path = ctx.artifact("examples.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run(ctx, cfg):
    traj = _load_jsonl(ctx.artifact("trajectories.jsonl"))
    abl = _load_jsonl(ctx.artifact("ablation.jsonl"))
    is_control = METRICS[ctx.metric].is_control

    # --- whole-population diagnostics (from trajectories) ---
    n_problems = len(traj)
    n_correct = sum(1 for r in traj if r.get("baseline_correct"))
    n_wrong = n_problems - n_correct
    triggered_rows = [r for r in traj if r.get("triggered")]
    n_triggered = len(triggered_rows)

    # AUC only where we have a real drift score (skip the random control)
    scored = [r for r in traj if r.get("max_z") is not None]
    auc = _auc([r["max_z"] for r in scored],
               [0 if r.get("baseline_correct") else 1 for r in scored])

    # false-trigger rate: fires on an already-correct problem
    false_triggers = sum(1 for r in triggered_rows if r.get("baseline_correct"))
    false_trigger_rate = (false_triggers / n_correct) if n_correct else None
    # trigger precision: of triggers, fraction on wrong-answer problems
    trigger_precision = ((n_triggered - false_triggers) / n_triggered) if n_triggered else None

    # --- formatting vs genuine split (from ablation records w/ trigger_token) ---
    n_formatting = 0
    genuine = []  # records that are genuine reasoning errors
    for r in abl:
        fmt = runtime.is_formatting_trigger(r.get("trigger_token"), "", cfg)
        if fmt:
            n_formatting += 1
        elif not r.get("baseline_correct"):
            genuine.append(r)
    n_genuine = len(genuine)
    genuine_precision = (n_genuine / len(abl)) if abl else None

    # --- per-path recovery on the GENUINE-error subset (wrong->right flips) ---
    path_rows = []
    for p in PATHS:
        flips = 0
        for r in genuine:
            pred = runtime.extract_final_number(r.get(p, ""))
            if runtime.answers_match(pred, r.get("gold")):
                flips += 1
        rate = (flips / n_genuine) if n_genuine else 0.0
        path_rows.append({"path": p, "genuine_n": n_genuine, "recovered": flips, "flip_rate": rate})

    csv_path = ctx.artifact("labeling.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "genuine_n", "recovered", "flip_rate"])
        w.writeheader()
        w.writerows(path_rows)

    # Concrete before/after examples (question text, spike
    # location, visible failure point, recovered output). Prefer cases Path B fixed.
    examples_path = _write_examples(ctx, genuine)

    # 5-way failure typology over the triggered baseline FAILURES :
    # arithmetic/logical, semantic drift, format/template collapse, premature
    # termination, other. This is what separates "we catch reasoning errors" from
    # "we catch the model forgetting its chat template".
    failures = [r for r in abl if not r.get("baseline_correct")]
    type_counts = {t: 0 for t in FAILURE_TYPES}
    for r in failures:
        type_counts[classify_failure(r, cfg)] += 1
    ftypes_path = ctx.artifact("failure_types.csv")
    with open(ftypes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["failure_type", "count", "fraction"])
        nf = max(len(failures), 1)
        for t in FAILURE_TYPES:
            w.writerow([t, type_counts[t], type_counts[t] / nf])
    ctx.log(f"labeling: failure types (n={len(failures)}): " +
            ", ".join(f"{t}={type_counts[t]}" for t in FAILURE_TYPES))

    summary = {
        "n_problems": n_problems,
        "n_baseline_correct": n_correct,
        "n_baseline_wrong": n_wrong,
        "n_triggered": n_triggered,
        "auc_maxz_vs_wrong": auc,
        "false_trigger_rate_on_correct": false_trigger_rate,
        "trigger_precision": trigger_precision,
        "n_formatting_triggers": n_formatting,
        "n_genuine_error_triggers": n_genuine,
        "genuine_error_fraction": genuine_precision,
        "is_control": bool(is_control),
        "per_path_genuine_flip": path_rows,
        "failure_types": type_counts,
        "n_failures_typed": len(failures),
    }

    ctx.log(f"labeling: AUC(max_z->wrong)={auc} | genuine triggers={n_genuine}/"
            f"{len(abl)} (formatting={n_formatting}) | "
            f"false-trigger-on-correct={false_trigger_rate}")
    for r in path_rows:
        ctx.log(f"labeling: {r['path']} genuine wrong->right "
                f"{r['recovered']}/{r['genuine_n']} ({r['flip_rate']:.2f})")
    ctx.finish_stage("stage_labeling", artifacts=[csv_path, examples_path, ftypes_path],
                     **summary)
    return summary
