"""Stage 3 — robustness across seeds.

Re-runs the ablation (smaller per-seed sample) for each seed in ROBUSTNESS_SEEDS
and records the trigger rate per seed, so the headline numbers come with a
mean +/- std instead of a single point. Per-seed JSONL is kept under
robustness/ for downstream inspection; the summary lands in robustness.csv.

This stage reports detection-trigger variance (cheap). Recovery-rate variance
would require judging every seed; that is deliberately left to a separate, larger
run to keep the default pipeline affordable — see README.
"""

import csv
import os

import numpy as np

import stage2_ablation
from drift_metrics import METRICS


def run(model, metric_name, ctx, cfg, layer, z):
    metric = METRICS[metric_name]
    sub = ctx.artifact("robustness")
    os.makedirs(sub, exist_ok=True)

    per_seed = []
    for seed in cfg.ROBUSTNESS_SEEDS:
        out_path = os.path.join(sub, f"ablation_seed{seed}.jsonl")
        # fresh file per seed
        open(out_path, "w").close()
        ctx.log(f"robustness: seed={seed} n={cfg.ROBUSTNESS_SAMPLES}")
        stats = stage2_ablation.run_ablation(
            model, metric, layer, z, cfg, seed, cfg.ROBUSTNESS_SAMPLES, out_path,
            ctx=ctx, dataset=ctx.dataset)
        per_seed.append({"seed": seed, **stats})

    rates = [r["trigger_rate"] for r in per_seed]
    summary = {
        "metric": metric_name,
        "mean_trigger_rate": float(np.mean(rates)) if rates else 0.0,
        "std_trigger_rate": float(np.std(rates)) if rates else 0.0,
        "n_seeds": len(rates),
    }

    csv_path = ctx.artifact("robustness.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "n_samples", "triggered", "trigger_rate"])
        w.writeheader()
        w.writerows(per_seed)

    ctx.log(f"robustness: trigger rate {summary['mean_trigger_rate']:.2f} "
            f"+/- {summary['std_trigger_rate']:.2f} over {summary['n_seeds']} seeds")
    ctx.finish_stage("stage3_robustness", artifacts=[csv_path], **summary)
    return summary
