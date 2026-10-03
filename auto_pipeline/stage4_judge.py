"""Stage 4 — judge the recovery paths.

Grades each of the four paths in the Stage 2 JSONL for semantic soundness and
reports a per-path recovery rate with a Wilson 95% CI. Optionally runs a second,
differently-worded judge (cfg.SECOND_JUDGE, TODO #14) and reports agreement.

Produces:
    judge_results.csv   path, n, recovered, rate, ci_low, ci_high
Returns the per-path summary list.
"""

import csv
import json
import math

import runtime

PATHS = ["path_a_baseline", "path_b_full_cure", "path_c_prompt_only", "path_d_rewind_only"]


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _load_records(jsonl_path):
    records = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        pass  # stage2 was skipped / produced nothing — judge nothing rather than crash
    return records


def run(model, ctx, cfg, ablation_path):
    records = _load_records(ablation_path)
    ctx.log(f"judge: scoring {len(records)} records x {len(PATHS)} paths"
            + (" x2 judges" if cfg.SECOND_JUDGE else ""))

    tally = {p: {"n": 0, "rec": 0} for p in PATHS}
    agree = {"both": 0, "total": 0}

    for rec in records:
        q = rec.get("question", "")
        for p in PATHS:
            resp = rec.get(p, "")
            score = runtime.judge(model, q, resp, cfg, persona="default")
            if not math.isnan(score):
                tally[p]["n"] += 1
                tally[p]["rec"] += int(score >= 0.5)
            if cfg.SECOND_JUDGE:
                s2 = runtime.judge(model, q, resp, cfg, persona="strict")
                if not math.isnan(score) and not math.isnan(s2):
                    agree["total"] += 1
                    agree["both"] += int((score >= 0.5) == (s2 >= 0.5))

    summary = []
    for p in PATHS:
        n, k = tally[p]["n"], tally[p]["rec"]
        lo, hi = _wilson(k, n)
        summary.append({"path": p, "n": n, "recovered": k,
                        "rate": (k / n if n else 0.0), "ci_low": lo, "ci_high": hi})

    csv_path = ctx.artifact("judge_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "n", "recovered", "rate", "ci_low", "ci_high"])
        w.writeheader()
        w.writerows(summary)

    extra = {}
    if cfg.SECOND_JUDGE and agree["total"]:
        extra["judge_agreement"] = agree["both"] / agree["total"]
        ctx.log(f"judge: inter-judge agreement {extra['judge_agreement']:.2f}")

    for r in summary:
        ctx.log(f"judge: {r['path']} recovery {r['rate']:.2f} "
                f"[{r['ci_low']:.2f},{r['ci_high']:.2f}] (n={r['n']})")
    ctx.finish_stage("stage4_judge", artifacts=[csv_path], table=summary, **extra)
    return summary
