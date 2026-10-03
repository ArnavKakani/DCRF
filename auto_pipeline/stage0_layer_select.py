"""Stage 0 — layer selection (+ figure data).

Sweeps the candidate layer band in ONE generation per problem (detect_multilayer)
and scores how well each layer separates the reasoning task (should drift -> high
max-Z) from the control tasks (should stay healthy). The best-separating layer is
selected and flows downstream.

Along the way it records the data two paper figures need, so no model reload is
required later:
  * per-layer spike timing + the text-break index -> distance-to-hallucination (Fig 2)
  * the first target problem's full per-layer Z-trace -> the layer cascade (Fig 3)

Produces:
    layer_traces.csv   task, problem_idx, layer, max_z, spike_idx, d2h
    layer_select.csv   layer, tpr, fpr, separation, mean_d2h
    cascade_trace.json one example's per-layer Z-trace (for Fig 3)
Returns the selected layer (int).
"""

import csv
import json

import numpy as np

import runtime


def _scan(model, metric, prompts, task, cfg, rows, collect_first=False, ctx=None):
    cascade = None
    for pi, prompt in enumerate(prompts):
        for r in range(cfg.LAYER_SELECT_ROLLOUTS):
            seed = cfg.BASE_SEED + r
            grab = collect_first and pi == 0 and r == 0
            res = runtime.detect_multilayer(model, prompt, metric, cfg.LAYER_BAND, seed, cfg,
                                            ref_z=cfg.LAYER_SELECT_REF_Z, collect_traces=grab)
            brk = res["break_idx"]
            for L, rec in res["layers"].items():
                spike = rec["spike_idx"]
                d2h = (brk - spike) if (brk is not None and spike != -1) else ""
                rows.append((task, pi, L, rec["max_z"], spike, d2h))
            if grab:
                cascade = {
                    "metric": metric.name,
                    "break_idx": brk,
                    "ref_z": cfg.LAYER_SELECT_REF_Z,
                    "layers": {str(L): rec.get("trace", []) for L, rec in res["layers"].items()},
                }
    return cascade


def run(model, metric, ctx, cfg):
    rows = []  # (task, problem_idx, layer, max_z, spike_idx, d2h)

    target = runtime.load_problems(cfg, "target", cfg.LAYER_SELECT_SAMPLES,
                                   dataset=ctx.dataset, model=model)
    ctx.log(f"layer-select [{ctx.dataset}]: {len(target)} target problems "
            f"x {len(cfg.LAYER_BAND)} layers")
    cascade = _scan(model, metric, target, "target", cfg, rows, collect_first=True, ctx=ctx)

    for spec in cfg.CONTROL_DATASETS:
        ctrl = runtime.load_problems(cfg, "control", cfg.LAYER_SELECT_CONTROL_SAMPLES,
                                     control_spec=spec, model=model)
        ctx.log(f"layer-select: control {spec[0]} ({len(ctrl)} problems)")
        _scan(model, metric, ctrl, f"control:{spec[0]}", cfg, rows, ctx=ctx)

    traces_path = ctx.artifact("layer_traces.csv")
    with open(traces_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "problem_idx", "layer", "max_z", "spike_idx", "d2h"])
        w.writerows(rows)

    if cascade:
        cascade_path = ctx.artifact("cascade_trace.json")
        with open(cascade_path, "w") as f:
            json.dump(cascade, f)

    # Score each layer. We report the threshold-based TPR/FPR/separation (reported for
    # reference), but SELECT on z_gap — the continuous differential
    # mean(target max-Z) - mean(control max-Z). z_gap measures how strongly the layer
    # separates drift from healthy WITHOUT depending on the ref_z cutoff, so it can't
    # go degenerate the way TPR-FPR does when every problem lands on one side of 12.5.
    ref_z = cfg.LAYER_SELECT_REF_Z
    select_rows = []
    for L in cfg.LAYER_BAND:
        tgt = [r for r in rows if r[0] == "target" and r[2] == L]
        ctl = [r for r in rows if r[0].startswith("control") and r[2] == L]
        tpr = float(np.mean([r[3] > ref_z for r in tgt])) if tgt else 0.0
        fpr = float(np.mean([r[3] > ref_z for r in ctl])) if ctl else 0.0
        target_mean_z = float(np.mean([r[3] for r in tgt])) if tgt else 0.0
        control_mean_z = float(np.mean([r[3] for r in ctl])) if ctl else 0.0
        z_gap = target_mean_z - control_mean_z
        d2hs = [r[5] for r in tgt if r[5] != ""]
        mean_d2h = float(np.mean(d2hs)) if d2hs else ""
        select_rows.append({"layer": L, "tpr": tpr, "fpr": fpr,
                            "separation": tpr - fpr,
                            "target_mean_z": target_mean_z,
                            "control_mean_z": control_mean_z,
                            "z_gap": z_gap, "mean_d2h": mean_d2h})

    select_path = ctx.artifact("layer_select.csv")
    with open(select_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "tpr", "fpr", "separation",
                                          "target_mean_z", "control_mean_z", "z_gap",
                                          "mean_d2h"])
        w.writeheader()
        w.writerows(select_rows)

    # Pick the largest differential (z_gap), tie-broken by TPR then by the TPR-FPR margin.
    best = max(select_rows, key=lambda r: (r["z_gap"], r["tpr"], r["separation"]))
    selected = int(best["layer"])
    ctx.log(f"layer-select: chose layer {selected} (z_gap={best['z_gap']:.3f}, "
            f"TPR={best['tpr']:.2f}, FPR={best['fpr']:.2f}, sep={best['separation']:.2f})")
    ctx.set_selection(layer=selected)
    ctx.finish_stage("stage0_layer_select", artifacts=[traces_path, select_path],
                     selected_layer=selected, table=select_rows)
    return selected, traces_path
