"""Build the figures for the *controlled negative-result* paper.

Design choices:
  * every cell is keyed by (model, dataset, metric) -- GSM8K and SVAMP are both
    shown and always labelled; nothing is silently dropped.
  * Yi-6B-Chat is excluded from the main figures by default (its 0.04 baseline is
    an answer-extraction artifact); pass --include-yi to add it.
  * one figure per Results claim, named after the claim it supports.
  * reads only CSV/JSON artifacts under runs/ -- no torch, no model reload.

Every figure is written as .png and .pdf at 300 dpi.

Inputs  : runs/<cell>/{threshold_sweep,threshold_downstream,failure_types,
          labeling,judge_results}.csv  +  runs/{master_results,results_table}.csv
Outputs : <out>/*.png, <out>/*.pdf, <out>/tab_detection.tex, <out>/tab_recovery.tex

Usage (from the repository root):
    .venv/bin/python auto_pipeline/figures/make_paper_figures.py
    .venv/bin/python auto_pipeline/figures/make_paper_figures.py --out paper_revision/figs
    .venv/bin/python auto_pipeline/figures/make_paper_figures.py --only detection_auc,judge_artifact

Figures, mapped to the Results subsections of paper_revision/main.tex:
    detection_auc      §"does not predict failure"  -- AUC(max_z->wrong) per config,
                       chance line, annotated with n_genuine (statistical power)
    calibration        §"never reaches its operating target" -- fit TPR/FPR sweep vs
                       the unreachable TPR>=0.60/FPR<=0.20 target box
    typology           §"triggers are not reasoning failures" -- stacked failure types
                       per config (Qwen=formatting, Mistral=arithmetic)
    final_accuracy     §"beats neither doing nothing nor random" -- baseline / prompt /
                       full / random-position final accuracy per config
    genuine_flip       §"beats neither ... nor random" -- wrong->right flip on the
                       genuine-error subset: baseline vs full vs random-position
    intervention_cost  §"net-negative wherever it acts" -- final accuracy vs threshold
                       (more firing -> lower accuracy), GSM8K vs SVAMP labelled
    judge_artifact     §"a same-model coherence judge cannot score recovery" -- the
                       same-model coherence judge's "recovery" vs the true flip rate
    tables             tab_detection.tex + tab_recovery.tex (regenerate the two
                       main.tex tables from disk)
"""

import argparse
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS = os.path.join(HERE, "..", "..", "runs")
DEFAULT_OUT = os.path.join(HERE, "paper_out")
DPI = 300

MODEL_SHORT = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral-7B",
    "01-ai/Yi-6B-Chat": "Yi-6B",
}
DATASET_LABEL = {"gsm8k": "GSM8K", "svamp": "SVAMP"}
METRIC_LABEL = {"l2": "L2", "sinkhorn": "Sinkhorn", "random": "Random"}
PATH_LABEL = {
    "path_a_baseline": "baseline",
    "path_b_full_cure": "full",
    "path_c_prompt_only": "prompt-only",
    "path_d_rewind_only": "rewind-only",
}
# canonical left-to-right order of the six detector configs (powered first)
CONFIG_ORDER = [
    ("Mistral-7B", "gsm8k", "l2"),
    ("Mistral-7B", "gsm8k", "sinkhorn"),
    ("Qwen2.5-7B", "gsm8k", "l2"),
    ("Qwen2.5-7B", "gsm8k", "sinkhorn"),
    ("Qwen2.5-7B", "svamp", "l2"),
    ("Qwen2.5-7B", "svamp", "sinkhorn"),
]
POWERED_MIN_GENUINE = 30          # below this, AUC is noise; we grey it out
TARGET_TPR, TARGET_FPR = 0.60, 0.20   # the (unreachable) calibration target
C_FULL, C_RANDOM, C_BASE = "#4C72B0", "#55A868", "#BBBBBB"
C_PROMPT = "#DD8452"


def _plt():
    # NeurIPS requires Type 1 / embedded TrueType fonts; matplotlib defaults to Type 3.
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    return os.path.join(out, name + ".png")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


_DIR_RE = re.compile(r"-(gsm8k|svamp)-(l2|sinkhorn|random)$")


def load(runs_dir, include_yi=False):
    """Index every cell by (model_short, dataset, metric). No collisions."""
    cells = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        m = _DIR_RE.search(os.path.basename(d))
        mpath = os.path.join(d, "manifest.json")
        if not (m and os.path.isfile(mpath)):
            continue
        dataset, metric = m.group(1), m.group(2)
        man = json.load(open(mpath))
        model = MODEL_SHORT.get(man.get("model"), (man.get("model") or "?").split("/")[-1])
        if model == "Yi-6B" and not include_yi:
            continue
        cells[(model, dataset, metric)] = {
            "dir": d, "model": model, "dataset": dataset, "metric": metric,
            "layer": man.get("selected_layer"), "z": man.get("selected_z"),
        }
    master = _read_csv(os.path.join(runs_dir, "master_results.csv"))
    table = _read_csv(os.path.join(runs_dir, "results_table.csv"))
    # attach AUC / n_genuine / per-path flip from master_results
    for r in master:
        m = _DIR_RE.search(r["run_dir"])
        if not m:
            continue
        model = MODEL_SHORT.get(r["model"], r["model"].split("/")[-1])
        key = (model, m.group(1), m.group(2))
        c = cells.get(key)
        if not c:
            continue
        c["auc"] = _f(r.get("auc_maxz_vs_wrong"))
        c["n_genuine"] = int(_f(r.get("n_genuine")) or 0)
        c["false_trigger"] = _f(r.get("false_trigger_rate"))
        c.setdefault("flip", {})[r["path"]] = _f(r.get("genuine_flip_rate"))
    # attach final / baseline accuracy per method from results_table
    for r in table:
        model = MODEL_SHORT.get(r["model"], r["model"].split("/")[-1])
        key = (model, r["dataset"], r["metric"])
        c = cells.get(key)
        if not c:
            continue
        c.setdefault("acc", {})[r["method"]] = _f(r.get("final_accuracy"))
        c["baseline_acc"] = _f(r.get("baseline_accuracy"))
    return cells


def _configs(cells):
    """Detector configs present, in canonical order, then any extras (e.g. Yi)."""
    out = [c for c in CONFIG_ORDER if c in cells]
    out += [k for k in cells if k not in out and k[2] != "random"]
    return out


def _label(key, sep="\n"):
    model, dataset, metric = key
    return f"{model}{sep}{DATASET_LABEL.get(dataset, dataset)} / {METRIC_LABEL.get(metric, metric)}"


# ===========================================================================
# 1. Detection at chance, with statistical power
# ===========================================================================
def fig_detection_auc(cells, out):
    plt = _plt()
    keys = _configs(cells)
    aucs = [cells[k].get("auc") for k in keys]
    ns = [cells[k].get("n_genuine", 0) for k in keys]
    colors = [C_FULL if n >= POWERED_MIN_GENUINE else C_BASE for n in ns]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(keys)), [a or 0 for a in aucs], color=colors)
    ax.axhline(0.5, ls="--", c="black", lw=1)
    ax.text(len(keys) - 0.5, 0.51, "chance (0.5)", ha="right", va="bottom", fontsize=9)
    for i, (a, n) in enumerate(zip(aucs, ns)):
        ax.text(i, (a or 0) + 0.01, f"n={n}", ha="center", va="bottom", fontsize=9,
                fontweight="bold" if n >= POWERED_MIN_GENUINE else "normal")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([_label(k) for k in keys], fontsize=8)
    ax.set_ylabel(r"AUC(max $z$ $\rightarrow$ wrong answer)")
    ax.set_ylim(0, 1)
    ax.set_title("The drift signal does not predict failure\n"
                 "(blue = statistically powered; grey = too few failures to estimate)",
                 fontsize=12)
    return _save(fig, out, "fig_detection_auc")


# ===========================================================================
# 2. Calibration never reaches the operating target
# ===========================================================================
def fig_calibration(cells, out):
    plt = _plt()
    keys = _configs(cells)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.add_patch(plt.Rectangle((0, TARGET_TPR), TARGET_FPR, 1 - TARGET_TPR,
                               facecolor="#55A868", alpha=0.18, zorder=0))
    ax.text(TARGET_FPR / 2, (1 + TARGET_TPR) / 2, "target\nregion", ha="center",
            va="center", fontsize=9, color="#2e6b46")
    cmap = plt.get_cmap("tab10")
    drew = False
    for i, k in enumerate(keys):
        rows = _read_csv(os.path.join(cells[k]["dir"], "threshold_sweep.csv"))
        if not rows:
            continue
        fpr = [_f(r.get("fit_fpr")) for r in rows]
        tpr = [_f(r.get("fit_tpr")) for r in rows]
        ax.plot(fpr, tpr, "-o", ms=4, lw=1.3, alpha=0.8, color=cmap(i % 10),
                label=_label(k, sep=" "))
        sel = next((r for r in rows if str(r.get("selected")).lower() == "true"), None)
        if sel:
            ax.plot(_f(sel["fit_fpr"]), _f(sel["fit_tpr"]), "*", ms=16,
                    color=cmap(i % 10), markeredgecolor="black", zorder=5)
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false-positive rate on control tasks (fit)")
    ax.set_ylabel("true-positive rate on reasoning task (fit)")
    ax.set_title("No threshold meets the operating target\n"
                 r"(target: TPR $\geq$ 0.60, FPR $\leq$ 0.20; $\star$ = selected $\tau$)",
                 fontsize=12)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, out, "fig_calibration")


# ===========================================================================
# 3. Failure typology of triggered cases
# ===========================================================================
# Table 3 categories <- raw labels (matches retype_failures.py: drift/early stop = never states an answer)
FAILURE_TYPES = {"Wrong answer": ["arithmetic_logical"],
                 "No answer": ["semantic_drift", "premature_termination"],
                 "Template collapse": ["format_template_collapse"]}
FT_COLOR = {"Wrong answer": "#4C72B0", "No answer": "#8172B3", "Template collapse": "#C44E52"}


def fig_typology(cells, out):
    plt = _plt()
    keys = _configs(cells)
    counts = {k: {r["failure_type"]: _f(r["count"]) or 0
                  for r in _read_csv(os.path.join(cells[k]["dir"], "failure_types.csv"))}
              for k in keys}
    keys = [k for k in keys if counts[k]]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0] * len(keys)
    for ft, raw in FAILURE_TYPES.items():
        vals = [sum(counts[k].get(r, 0) for r in raw) for k in keys]
        ax.bar(range(len(keys)), vals, bottom=bottom, color=FT_COLOR[ft], label=ft)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([_label(k) for k in keys], fontsize=8)
    ax.set_ylabel("triggered baseline failures")
    ax.legend(fontsize=8)
    return _save(fig, out, "fig_typology")


# ===========================================================================
# 4. Final accuracy: baseline / prompt / full / random
# ===========================================================================
def fig_final_accuracy(cells, out):
    plt = _plt()
    keys = _configs(cells)
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.2
    series = [
        ("baseline", lambda c: c.get("baseline_acc"), C_BASE),
        ("prompt-only", lambda c: c.get("acc", {}).get("prompt_only"), C_PROMPT),
        ("full", lambda c: c.get("acc", {}).get("sdr_" + c["metric"]), C_FULL),
        ("random-position", lambda c: _random_acc(cells, c), C_RANDOM),
    ]
    for si, (lbl, fn, col) in enumerate(series):
        ax.bar([i + si * w for i in range(len(keys))], [fn(cells[k]) or 0 for k in keys],
               width=w, color=col, label=lbl)
    ax.set_xticks([i + 1.5 * w for i in range(len(keys))])
    ax.set_xticklabels([_label(k) for k in keys], fontsize=8)
    ax.set_ylabel("final task accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("The intervention beats neither doing nothing nor a random trigger",
                 fontsize=12)
    ax.legend(fontsize=9, ncol=4, loc="upper center")
    return _save(fig, out, "fig_final_accuracy")


def _random_acc(cells, c):
    rc = cells.get((c["model"], c["dataset"], "random"))
    return rc.get("acc", {}).get("random_trigger") if rc else None


# ===========================================================================
# 5. Genuine wrong->right flip: baseline vs full vs random-position
# ===========================================================================
def fig_genuine_flip(cells, out):
    plt = _plt()
    keys = _configs(cells)
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.26
    series = [
        ("baseline (no intervention)", "path_a_baseline", C_BASE, lambda c: c),
        ("full", "path_b_full_cure", C_FULL, lambda c: c),
        ("random-position", "path_b_full_cure", C_RANDOM,
         lambda c: cells.get((c["model"], c["dataset"], "random"))),
    ]
    for si, (lbl, path, col, pick) in enumerate(series):
        vals = []
        for k in keys:
            src = pick(cells[k])
            vals.append((src or {}).get("flip", {}).get(path) or 0)
        ax.bar([i + si * w for i in range(len(keys))], vals, width=w, color=col, label=lbl)
    for i, k in enumerate(keys):
        ax.text(i + w, -0.04, f"n={cells[k].get('n_genuine', 0)}", ha="center",
                va="top", fontsize=8)
    ax.set_xticks([i + w for i in range(len(keys))])
    ax.set_xticklabels([_label(k) for k in keys], fontsize=8)
    ax.set_ylabel(r"genuine wrong$\rightarrow$right flip rate")
    ax.set_ylim(0, max(0.4, ax.get_ylim()[1]))
    ax.set_title("Conditioning on the latent signal recovers no more failures than a random position",
                 fontsize=12)
    ax.legend(fontsize=9)
    return _save(fig, out, "fig_genuine_flip")


# ===========================================================================
# 6. Intervention is net-negative: accuracy vs threshold
# ===========================================================================
def fig_intervention_cost(cells, out):
    plt = _plt()
    keys = [k for k in _configs(cells)
            if os.path.exists(os.path.join(cells[k]["dir"], "threshold_downstream.csv"))]
    if not keys:
        return None
    cmap = _plt().get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, k in enumerate(keys):
        rows = _read_csv(os.path.join(cells[k]["dir"], "threshold_downstream.csv"))
        zs = [_f(r["z"]) for r in rows]
        acc = [_f(r.get("final_accuracy")) for r in rows]
        ax.plot(zs, acc, "-o", color=cmap(i % 10), label=_label(k, sep=" "))
        base = cells[k].get("baseline_acc")
        if base is not None:
            ax.axhline(base, ls=":", lw=1, color=cmap(i % 10), alpha=0.5)
    ax.set_xlabel(r"trigger threshold $\tau$  (left = fires often, right = fires rarely)")
    ax.set_ylabel("final task accuracy")
    ax.legend(fontsize=8)
    return _save(fig, out, "fig_intervention_cost")


# ===========================================================================
# 7. The coherence-judge artifact
# ===========================================================================
def fig_judge_artifact(cells, out):
    plt = _plt()
    # use the statistically powered cells (most genuine errors)
    keys = sorted(_configs(cells), key=lambda k: -cells[k].get("n_genuine", 0))[:2]
    keys = [k for k in keys if os.path.exists(os.path.join(cells[k]["dir"], "judge_results.csv"))]
    if not keys:
        return None
    paths = ["path_a_baseline", "path_b_full_cure", "path_c_prompt_only", "path_d_rewind_only"]
    fig, axes = plt.subplots(1, len(keys), figsize=(8, 3.2),
                             squeeze=False, constrained_layout=True)
    for ax, k in zip(axes[0], keys):
        judge = {r["path"]: _f(r["rate"])
                 for r in _read_csv(os.path.join(cells[k]["dir"], "judge_results.csv"))}
        flip = {r["path"]: _f(r["flip_rate"])
                for r in _read_csv(os.path.join(cells[k]["dir"], "labeling.csv"))}
        w = 0.38
        ax.bar([i - w / 2 for i in range(len(paths))], [judge.get(p) or 0 for p in paths],
               width=w, color="#C44E52", label="judge: sound")
        ax.bar([i + w / 2 for i in range(len(paths))], [flip.get(p) or 0 for p in paths],
               width=w, color=C_FULL, label="answer: wrong to right")
        ax.set_xticks(range(len(paths)))
        ax.set_xticklabels([PATH_LABEL[p] for p in paths], fontsize=9, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_ylabel("rate", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(_label(k, sep=" "), fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
    return _save(fig, out, "fig_judge_artifact")


# ===========================================================================
# Tables (regenerate the two main.tex tables from disk)
# ===========================================================================
def make_tables(cells, out):
    keys = _configs(cells)
    det = [r"\begin{tabular}{llcccc}", r"\toprule",
           r"\bf Model & \bf Dataset & \bf Metric & \bf AUC & \bf $n_{\text{genuine}}$ & \bf False-trigger \\",
           r"\midrule"]
    for k in sorted(keys, key=lambda k: cells[k].get("n_genuine", 0)):
        c = cells[k]
        auc = f"{c.get('auc'):.2f}" if c.get("auc") is not None else "--"
        ft = f"{c.get('false_trigger'):.2f}" if c.get("false_trigger") is not None else "--"
        bold = c.get("n_genuine", 0) >= POWERED_MIN_GENUINE
        auc_s = f"\\bf {auc}" if bold else auc
        n_s = f"\\bf {c.get('n_genuine',0)}" if bold else str(c.get("n_genuine", 0))
        det.append(f"{k[0]} & {DATASET_LABEL.get(k[1])} & {METRIC_LABEL.get(k[2])} & "
                   f"{auc_s} & {n_s} & {ft} \\\\")
    det += [r"\bottomrule", r"\end{tabular}"]
    p1 = os.path.join(out, "tab_detection.tex")
    open(p1, "w").write("\n".join(det) + "\n")

    rec = [r"\begin{tabular}{llccccc}", r"\toprule",
           r"\bf Model & \bf Dataset & \bf Metric & \bf Baseline & \bf Prompt-only & \bf Full & \bf Random \\",
           r"\midrule"]
    for k in keys:
        c = cells[k]
        def g(v):
            return f"{v:.2f}" if v is not None else "--"
        rec.append(f"{k[0]} & {DATASET_LABEL.get(k[1])} & {METRIC_LABEL.get(k[2])} & "
                   f"{g(c.get('baseline_acc'))} & {g(c.get('acc',{}).get('prompt_only'))} & "
                   f"{g(c.get('acc',{}).get('sdr_'+k[2]))} & {g(_random_acc(cells, c))} \\\\")
    rec += [r"\bottomrule", r"\end{tabular}"]
    p2 = os.path.join(out, "tab_recovery.tex")
    open(p2, "w").write("\n".join(rec) + "\n")
    return f"{p1}, {p2}"


FIGURES = {
    "detection_auc": fig_detection_auc,
    "calibration": fig_calibration,
    "typology": fig_typology,
    "final_accuracy": fig_final_accuracy,
    "genuine_flip": fig_genuine_flip,
    "intervention_cost": fig_intervention_cost,
    "judge_artifact": fig_judge_artifact,
    "tables": make_tables,
}


def main():
    ap = argparse.ArgumentParser(description="Build negative-result paper figures")
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", default="all", help="comma-separated subset of: " + ",".join(FIGURES))
    ap.add_argument("--include-yi", action="store_true", help="include Yi-6B-Chat (excluded by default)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cells = load(args.runs, include_yi=args.include_yi)
    if not cells:
        print(f"no usable cells under {args.runs}")
        return
    print(f"loaded {len(cells)} cells: " + ", ".join("/".join(k) for k in sorted(cells)))
    which = list(FIGURES) if args.only == "all" else [s.strip() for s in args.only.split(",")]
    for name in which:
        fn = FIGURES.get(name)
        if not fn:
            print(f"  [skip] unknown figure '{name}'")
            continue
        try:
            res = fn(cells, args.out)
            print(f"  [ok] {name} -> {res}" if res else f"  [skip] {name}: no data")
        except Exception as e:
            print(f"  [warn] {name} failed: {e}")
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
