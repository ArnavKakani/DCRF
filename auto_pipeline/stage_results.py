"""Results-table stage — the cross-method results table.

For each method, over the WHOLE dataset, report: dataset, model, N, trigger rate,
baseline accuracy, accuracy-after-intervention, recovery on baseline failures, and
false-trigger rate on correct baselines.

Methods emitted per run:
  * non-control metric run -> baseline_cot, prompt_only (path C), sdr_<metric> (path B; the method key keeps this name in the CSVs)
  * random control run     -> random_trigger (path B at a random token)

So a sinkhorn run + a random run together give the four methods
(baseline / prompt-only / full / random) on that (model, dataset). Rows are appended
to runs/results_table.csv — the cross (model x dataset x method) table.
"""

import csv
import json
import os

import analysis
from drift_metrics import METRICS

FIELDS = ["model", "dataset", "metric", "method", "n", "trigger_rate",
          "baseline_accuracy", "final_accuracy", "recovery_on_failures",
          "false_trigger_on_correct"]


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


def run(ctx, cfg):
    m = ctx.manifest
    model, dataset, metric = m.get("model"), ctx.dataset, ctx.metric
    traj = _load_jsonl(ctx.artifact("trajectories.jsonl"))
    abl_idx = analysis.index_by_uid(_load_jsonl(ctx.artifact("ablation.jsonl")))

    rows = []

    def add(method, summ):
        rows.append({"model": model, "dataset": dataset, "metric": metric,
                     "method": method, **summ})

    if METRICS[metric].is_control:
        add("random_trigger", analysis.method_summary(traj, abl_idx, "path_b_full_cure"))
    else:
        add("baseline_cot", analysis.baseline_summary(traj))
        add("prompt_only", analysis.method_summary(traj, abl_idx, "path_c_prompt_only"))
        add(f"sdr_{metric}", analysis.method_summary(traj, abl_idx, "path_b_full_cure"))

    local = ctx.artifact("results_methods.csv")
    with open(local, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    rollup = os.path.join(cfg.RUNS_DIR, "results_table.csv")
    new = not os.path.exists(rollup)
    with open(rollup, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_dir"] + FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"run_dir": os.path.basename(ctx.dir), **r})

    for r in rows:
        ctx.log(f"results[{r['method']}]: base_acc={r['baseline_accuracy']} "
                f"final_acc={r['final_accuracy']} recov={r['recovery_on_failures']} "
                f"trig={r['trigger_rate']:.2f}")
    ctx.finish_stage("stage_results", artifacts=[local, rollup], rows=rows)
    return local
