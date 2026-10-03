"""Threshold sweep WITH downstream metrics.

The plain Stage 1 sweep reports trigger/false-trigger from the standardized scores.
At each Z in {2,4,6,8,10,12.5,15}, actually run
detection+recovery and report trigger rate, recovery rate, false-trigger rate, and
FINAL ACCURACY — so the choice of Z=12.5 is justified by its downstream effect, not
just its trigger geometry.

This re-runs the ablation once per Z (on THRESHOLD_DOWNSTREAM_SAMPLES problems), so
it is the expensive stage; keep the sample modest. Control metrics have no Z and are
skipped.

Output: threshold_downstream.csv  (z, trigger_rate, recovery_rate, false_trigger_rate, final_accuracy)
"""

import csv
import json
import os

import analysis
import stage2_ablation
from drift_metrics import METRICS


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


def run(model, metric_name, ctx, cfg, layer):
    metric = METRICS[metric_name]
    if metric.is_control:
        ctx.log("threshold-downstream: skipped (control metric has no Z)")
        ctx.finish_stage("stage_threshold_downstream", status="ok", skipped=True)
        return None

    sub = ctx.artifact("threshold_downstream")
    os.makedirs(sub, exist_ok=True)
    rows = []
    n = cfg.THRESHOLD_DOWNSTREAM_SAMPLES
    for z in cfg.Z_GRID:
        abl_path = os.path.join(sub, f"ablation_z{z}.jsonl")
        traj_path = os.path.join(sub, f"trajectories_z{z}.jsonl")
        open(abl_path, "w").close()
        open(traj_path, "w").close()
        ctx.log(f"threshold-downstream: Z={z} (n={n})")
        stage2_ablation.run_ablation(model, metric, layer, z, cfg, cfg.BASE_SEED, n,
                                     abl_path, traj_path=traj_path, ctx=ctx,
                                     dataset=ctx.dataset)
        traj = _load_jsonl(traj_path)
        abl_idx = analysis.index_by_uid(_load_jsonl(abl_path))
        s = analysis.method_summary(traj, abl_idx, "path_b_full_cure")
        rows.append({
            "z": z,
            "trigger_rate": s["trigger_rate"],
            "recovery_rate": s["recovery_on_failures"],
            "false_trigger_rate": s["false_trigger_on_correct"],
            "final_accuracy": s["final_accuracy"],
        })

    out_path = ctx.artifact("threshold_downstream.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["z", "trigger_rate", "recovery_rate",
                                          "false_trigger_rate", "final_accuracy"])
        w.writeheader()
        w.writerows(rows)

    best = max((r for r in rows if r["final_accuracy"] is not None),
               key=lambda r: r["final_accuracy"], default=None)
    if best:
        ctx.log(f"threshold-downstream: best final accuracy at Z={best['z']} "
                f"({best['final_accuracy']:.2f})")
    ctx.finish_stage("stage_threshold_downstream", artifacts=[out_path], table=rows)
    return out_path
