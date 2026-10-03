"""Stage 5 — aggregate into the paper-ready results.

Pulls the selected layer/Z, robustness summary, and per-path recovery rates from
this run and writes:
    master_results.csv   one tidy row per (metric, path) with layer, z,
                         trigger rate +/- std, recovery rate + CI
    plots/recovery_by_path.png    bar chart of recovery rate per path (with CI)
    plots/threshold_sweep.png     TPR/FPR vs Z at the selected layer

It also APPENDS the run's rows to a cross-metric rollup at runs/master_results.csv
so that, once every metric version has run, that single file is the head-to-head
metric-ablation table for the paper (TODO #8, #15).

Plotting is best-effort: a headless box without a usable matplotlib backend still
gets the CSVs.
"""

import csv
import json
import os


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _plots(ctx, cfg, judge_rows, sweep_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        ctx.log(f"aggregate: skipping plots ({e})")
        return []

    plot_dir = ctx.artifact("plots")
    os.makedirs(plot_dir, exist_ok=True)
    made = []

    if judge_rows:
        labels = [r["path"].replace("path_", "").replace("_", " ") for r in judge_rows]
        rates = [float(r["rate"]) for r in judge_rows]
        lo = [float(r["rate"]) - float(r["ci_low"]) for r in judge_rows]
        hi = [float(r["ci_high"]) - float(r["rate"]) for r in judge_rows]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates, yerr=[lo, hi], capsize=4, color="#4C72B0")
        plt.ylabel("recovery rate")
        plt.title(f"Recovery by path — {ctx.metric}")
        plt.ylim(0, 1)
        plt.tight_layout()
        p = os.path.join(plot_dir, "recovery_by_path.png")
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(p)

    if sweep_rows:
        zs = [float(r["z"]) for r in sweep_rows]
        tpr = [float(r["tpr"]) for r in sweep_rows]
        fpr = [float(r["fpr"]) for r in sweep_rows]
        plt.figure(figsize=(7, 4))
        plt.plot(zs, tpr, "o-", label="TPR (target)")
        plt.plot(zs, fpr, "s--", label="FPR (control)")
        plt.xlabel("Z-threshold")
        plt.ylabel("trigger rate")
        plt.title(f"Threshold sweep — {ctx.metric}")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(plot_dir, "threshold_sweep.png")
        plt.savefig(p, dpi=150)
        plt.close()
        made.append(p)

    return made


def run(ctx, cfg):
    m = ctx.manifest
    metric = ctx.metric
    model = m.get("model")
    layer = m.get("selected_layer")
    z = m.get("selected_z")
    robust = m["stages"].get("stage3_robustness", {})
    mean_trig = robust.get("mean_trigger_rate")
    std_trig = robust.get("std_trigger_rate")

    # labeling-stage metrics (the formatting-vs-semantic numbers)
    lab = m["stages"].get("stage_labeling", {})
    auc = lab.get("auc_maxz_vs_wrong")
    false_trig = lab.get("false_trigger_rate_on_correct")
    genuine_frac = lab.get("genuine_error_fraction")
    n_genuine = lab.get("n_genuine_error_triggers")
    genuine_flip = {r["path"]: r["flip_rate"] for r in lab.get("per_path_genuine_flip", [])}

    judge_rows = _read_csv(ctx.artifact("judge_results.csv"))
    sweep_rows = _read_csv(ctx.artifact("threshold_sweep.csv"))

    base = {
        "model": model, "metric": metric, "layer": layer, "z": z,
        "mean_trigger_rate": mean_trig, "std_trigger_rate": std_trig,
        "auc_maxz_vs_wrong": auc, "false_trigger_rate": false_trig,
        "genuine_error_fraction": genuine_frac, "n_genuine": n_genuine,
    }
    rows = []
    if judge_rows:
        for jr in judge_rows:
            rows.append({**base, "path": jr["path"], "recovery_rate": jr["rate"],
                         "ci_low": jr["ci_low"], "ci_high": jr["ci_high"], "n": jr["n"],
                         "genuine_flip_rate": genuine_flip.get(jr["path"])})
    else:
        # judge skipped — still emit the labeling-derived per-path flip rates
        for path, flip in genuine_flip.items():
            rows.append({**base, "path": path, "recovery_rate": None,
                         "ci_low": None, "ci_high": None, "n": n_genuine,
                         "genuine_flip_rate": flip})

    fields = ["model", "metric", "layer", "z", "mean_trigger_rate", "std_trigger_rate",
              "auc_maxz_vs_wrong", "false_trigger_rate", "genuine_error_fraction", "n_genuine",
              "path", "recovery_rate", "ci_low", "ci_high", "n", "genuine_flip_rate"]

    master_path = ctx.artifact("master_results.csv")
    with open(master_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Append to the cross-metric rollup (create with header if new).
    rollup = os.path.join(cfg.RUNS_DIR, "master_results.csv")
    new = not os.path.exists(rollup)
    with open(rollup, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_dir"] + fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"run_dir": os.path.basename(ctx.dir), **r})

    plots = _plots(ctx, cfg, judge_rows, sweep_rows)

    artifacts = [master_path, rollup] + plots
    ctx.log(f"aggregate: wrote {len(rows)} rows -> {master_path} (+rollup {rollup})")
    ctx.finish_stage("stage5_aggregate", artifacts=artifacts, rows=len(rows))
    return master_path
