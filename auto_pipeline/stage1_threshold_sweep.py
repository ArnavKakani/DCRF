"""Stage 1 — threshold calibration.

Reuses the max-Z traces Stage 0 already computed (no new generation) and sweeps
Z_GRID at the selected layer, reporting trigger rate on the target task (TPR) and
on the controls (FPR) for each threshold. Picks the operating point: the highest Z
whose TPR >= TARGET_TPR while FPR <= MAX_FPR; if none qualifies, the Z that
maximises (TPR - FPR).

Produces:
    threshold_sweep.csv   z, tpr, fpr, separation, selected
Returns the selected Z (float).
"""

import csv

import numpy as np


def _read_traces(traces_path, layer):
    """Return (tgt, ctl) lists of (problem_idx, max_z) for the chosen layer."""
    tgt, ctl = [], []
    with open(traces_path) as f:
        for row in csv.DictReader(f):
            if int(row["layer"]) != layer:
                continue
            item = (int(row["problem_idx"]), float(row["max_z"]))
            if row["task"] == "target":
                tgt.append(item)
            elif row["task"].startswith("control"):
                ctl.append(item)
    return tgt, ctl


def _rates(tgt, ctl, z):
    tpr = float(np.mean([v > z for _, v in tgt])) if tgt else 0.0
    fpr = float(np.mean([v > z for _, v in ctl])) if ctl else 0.0
    return tpr, fpr


def run(metric, ctx, cfg, selected_layer, traces_path):
    tgt, ctl = _read_traces(traces_path, selected_layer)

    # Held-out split by problem-index parity: select Z on the calibration half,
    # validate on the held-out half (addresses "held-out validation" / sensitivity).
    fit_t = [x for x in tgt if x[0] % 2 == 0]
    val_t = [x for x in tgt if x[0] % 2 == 1]
    fit_c = [x for x in ctl if x[0] % 2 == 0]
    val_c = [x for x in ctl if x[0] % 2 == 1]

    sweep = []
    for z in cfg.Z_GRID:
        tpr, fpr = _rates(tgt, ctl, z)              # full-sample (for the sensitivity figure)
        ftpr, ffpr = _rates(fit_t, fit_c, z)        # calibration half
        vtpr, vfpr = _rates(val_t, val_c, z)        # held-out half
        sweep.append({"z": z, "tpr": tpr, "fpr": fpr, "separation": tpr - fpr,
                      "fit_tpr": ftpr, "fit_fpr": ffpr,
                      "val_tpr": vtpr, "val_fpr": vfpr})

    # Selection uses ONLY the calibration half.
    def ok(r):
        return r["fit_tpr"] >= cfg.TARGET_TPR and r["fit_fpr"] <= cfg.MAX_FPR
    qualifying = [r for r in sweep if ok(r)]
    if qualifying:
        chosen = max(qualifying, key=lambda r: r["z"])
        reason = f"highest Z with fit TPR>={cfg.TARGET_TPR} and FPR<={cfg.MAX_FPR}"
    else:
        chosen = max(sweep, key=lambda r: r["fit_tpr"] - r["fit_fpr"])
        reason = "no Z met fit TPR/FPR targets; took best fit (TPR - FPR)"

    for r in sweep:
        r["selected"] = (r["z"] == chosen["z"])

    sweep_path = ctx.artifact("threshold_sweep.csv")
    with open(sweep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["z", "tpr", "fpr", "separation",
                                          "fit_tpr", "fit_fpr", "val_tpr", "val_fpr", "selected"])
        w.writeheader()
        w.writerows(sweep)

    selected_z = float(chosen["z"])
    ctx.log(f"threshold: chose Z={selected_z} ({reason}); "
            f"fit TPR/FPR={chosen['fit_tpr']:.2f}/{chosen['fit_fpr']:.2f}, "
            f"held-out TPR/FPR={chosen['val_tpr']:.2f}/{chosen['val_fpr']:.2f}")
    ctx.set_selection(z=selected_z)
    ctx.finish_stage("stage1_threshold_sweep", artifacts=[sweep_path],
                     selected_z=selected_z, reason=reason,
                     val_tpr=chosen["val_tpr"], val_fpr=chosen["val_fpr"], table=sweep)
    return selected_z
