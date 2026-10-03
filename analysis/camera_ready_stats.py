"""Camera-ready statistics for the MATH-AI paper, computed from runs/ only.
Produces: AUC on the 80 problems disjoint from layer selection (with bootstrap CIs),
greedy-continuation-control accuracies and paired full-minus-control differences,
initially-correct answers turned wrong, the matched-timing comparison on the 35 shared
Sinkhorn cases, judge population counts, trigger-position statistics, and the
fig_trigger_positions figure written to figs/.
Run from anywhere: uv run --with matplotlib,numpy,scikit-learn python analysis/camera_ready_stats.py
"""
import json, re, csv, glob, os
import numpy as np
from sklearn.metrics import roc_auc_score
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
R = ROOT + "/runs/"
# pull parser source text out of runtime.py without importing it (it imports torch)
src = open(ROOT + "/auto_pipeline/runtime.py").read()
ns = {"re": re}
for pat in [r"^_NUM_RE = .*?\n", r"^def _to_float\(.*?(?=\n\n\n)", r"^_ANSWER_RE = re\.compile\(\n.*?\)\n",
            r"^def extract_final_number\(.*?(?=\n\n\n)", r"^def answers_match\(.*?(?=\n\n\n)"]:
    exec(re.search(pat, src, re.S | re.M).group(0), ns)
efn, am = ns["extract_final_number"], ns["answers_match"]
assert efn("so 3+4 = 7. The answer is 1,234.") == 1234.0 and am(2.0, 2.0) and not am(None, 2)

def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
M = {"l2": R+"20260624-230642-Mistral-7B-Instruct-v0.1-gsm8k-l2",
     "sinkhorn": R+"20260625-001738-Mistral-7B-Instruct-v0.1-gsm8k-sinkhorn",
     "random": R+"20260625-011259-Mistral-7B-Instruct-v0.1-gsm8k-random"}
rng = np.random.default_rng(0)
B = 2000

def strip(t):  # sensitivity: score only the generated part after the prompt
    return t.split("[/INST]", 1)[-1] if t else t

print("== 1. AUC max_z -> baseline wrong")
for k in ["l2", "sinkhorn"]:
    T = jl(M[k] + "/trajectories.jsonl")
    nulls = sum(t["max_z"] is None for t in T)
    y = np.array([not t["baseline_correct"] for t in T], int)
    s = np.array([t["max_z"] if t["max_z"] is not None else -np.inf for t in T], float)
    uid = np.array([t["uid"] for t in T])
    m = uid >= 20
    a100, a80 = roc_auc_score(y, s), roc_auc_score(y[m], s[m])
    y8, s8 = y[m], s[m]; bs = []
    for _ in range(B):
        i = rng.integers(0, len(y8), len(y8))
        if y8[i].min() != y8[i].max(): bs.append(roc_auc_score(y8[i], s8[i]))
    print(f"{k}: n={len(T)} null_maxz={nulls} wrong={y.sum()} AUC100={a100:.4f} | n80={m.sum()} wrong80={y8.sum()} AUC80={a80:.4f} CI95=[{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}] (valid resamples {len(bs)})")

print("== 2. path A / B accuracy")
for k in ["l2", "sinkhorn", "random"]:
    T = {t["uid"]: t for t in jl(M[k] + "/trajectories.jsonl")}
    A = {a["uid"]: a for a in jl(M[k] + "/ablation.jsonl")}
    for mode, f in [("fulltext", lambda x: x), ("after[/INST]", strip)]:
        uids = sorted(T)
        def corr(u, p):
            if u in A: return am(efn(f(A[u][p])), A[u]["gold"])
            return bool(T[u]["baseline_correct"])
        ca = np.array([corr(u, "path_a_baseline") for u in uids], int)
        cb = np.array([corr(u, "path_b_full_cure") for u in uids], int)
        base = np.array([T[u]["baseline_correct"] for u in uids], int)
        trig = np.array([u in A for u in uids])
        d = cb - ca
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
        broke = int(((base == 1) & trig & (cb == 0)).sum())
        brokeA = int(((base == 1) & trig & (ca == 0)).sum())
        mism = sum(A[u]["baseline_correct"] != T[u]["baseline_correct"] for u in A)
        print(f"{k} [{mode}]: triggered={trig.sum()} | A trig={ca[trig].sum()}/{trig.sum()}={ca[trig].mean():.3f} B trig={cb[trig].sum()}/{trig.sum()}={cb[trig].mean():.3f} | "
              f"A all={ca.mean():.3f} B all={cb.mean():.3f} sampled baseline all={base.mean():.3f} | B-A={d.mean():+.3f} CI95=[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] | "
              f"init-correct&triggered={int(((base==1)&trig).sum())}, B turns wrong={broke}, A turns wrong={brokeA} | abl/traj baseline_correct mismatches={mism}")

print("== 3. matched timing")
S = jl(M["sinkhorn"] + "/ablation.jsonl"); Rd = {a["uid"]: a for a in jl(M["random"] + "/ablation.jsonl")}
for mode, f in [("fulltext", lambda x: x), ("after[/INST]", strip)]:
    W = [a for a in S if not a["baseline_correct"]]
    common = [a for a in W if a["uid"] in Rd]
    c = lambda a, p: am(efn(f(a[p])), a["gold"])
    mA = sum(c(a, "path_a_baseline") for a in common); mB = sum(c(a, "path_b_full_cure") for a in common)
    rA = sum(c(Rd[a["uid"]], "path_a_baseline") for a in common); rB = sum(c(Rd[a["uid"]], "path_b_full_cure") for a in common)
    rbw = sum(not Rd[a["uid"]]["baseline_correct"] for a in common)
    rsp = [Rd[a["uid"]]["spike_idx"] for a in common]
    n = len(common)
    print(f"[{mode}] sinkhorn triggered&wrong={len(W)} common with random={n} | monitor A={mA}/{n} B={mB}/{n} | random A={rA}/{n} B={rB}/{n} | "
          f"lift B-A monitor={(mB-mA)/n:+.3f} random={(rB-rA)/n:+.3f} | monitor A-randomA={(mA-rA)/n:+.3f} monitor B-randomB={(mB-rB)/n:+.3f} | random-run baseline wrong on these uids={rbw} | random spike_idx median={np.median(rsp)}")

print("== 4. judge populations Mistral L2")
A = jl(M["l2"] + "/ablation.jsonl")
print("triggered records:", len(A))
for r in csv.DictReader(open(M["l2"] + "/judge_results.csv")): print(" judge csv:", dict(r))
for p in ["path_a_baseline", "path_b_full_cure", "path_c_prompt_only", "path_d_rewind_only"]:
    W = [a for a in A if not a["baseline_correct"]]
    print(f" {p}: correct over all 99 = {sum(am(efn(a[p]), a['gold']) for a in A)}; wrong->right over {len(W)} baseline-wrong = {sum(am(efn(a[p]), a['gold']) for a in W)}")

print("== 5. trigger positions")
cells = [("Qwen2.5 GSM8K L2", "20260624-085847-Qwen2.5-7B-Instruct-gsm8k-l2"), ("Qwen2.5 GSM8K Sinkhorn", "20260624-113828-Qwen2.5-7B-Instruct-gsm8k-sinkhorn"),
         ("Qwen2.5 SVAMP L2", "20260624-154807-Qwen2.5-7B-Instruct-svamp-l2"), ("Qwen2.5 SVAMP Sinkhorn", "20260624-175251-Qwen2.5-7B-Instruct-svamp-sinkhorn"),
         ("Mistral GSM8K L2", "20260624-230642-Mistral-7B-Instruct-v0.1-gsm8k-l2"), ("Mistral GSM8K Sinkhorn", "20260625-001738-Mistral-7B-Instruct-v0.1-gsm8k-sinkhorn")]
data = []
for name, d in cells:
    T = jl(R + d + "/trajectories.jsonl")
    sp = np.array([t["spike_idx"] for t in T if t["triggered"]], float)
    na = len(jl(R + d + "/ablation.jsonl"))
    data.append((name, sp))
    print(f"{name}: n={len(sp)} (ablation rows {na}) min={sp.min():.0f} median={np.median(sp):.1f} mean={sp.mean():.2f} max={sp.max():.0f} frac<=20={np.mean(sp<=20):.3f} ({int((sp<=20).sum())}) frac<=15={np.mean(sp<=15):.3f}")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42; plt.rcParams["font.size"] = 8
fig, axes = plt.subplots(2, 3, figsize=(9, 4.5), sharex=True)
mx = max(sp.max() for _, sp in data)
bins = np.arange(0, mx + 10, 10)
for ax, (name, sp) in zip(axes.T.ravel(), data):
    ax.hist(sp, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(15, ls="--", color="k", lw=0.9)
    ax.set_title(f"{name} (n={len(sp)})", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[1]: ax.set_xlabel("trigger token index")
for ax in axes[:, 0]: ax.set_ylabel("problems")
fig.tight_layout()
out = ROOT + "/figs/fig_trigger_positions"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print("wrote", out)
