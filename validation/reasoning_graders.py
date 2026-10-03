"""Reasoning-quality graders — grade the REASONING ITSELF, not just the final answer.

Three complementary alternatives to a single primed same-model coherence judge,
plus an aggregator that averages whichever you run into one reasoning score.

  (1) step_verifier   — DETERMINISTIC, no model. Parses the arithmetic in the
                        reasoning ("a op b = c") and recomputes each step; for code,
                        runs unit tests. Catches "right answer, wrong/lucky reasoning."
                        Cost: low (no model, no bias).
  (2) independent_judge — a DIFFERENT model family than the generator, with a NEUTRAL
                        rubric (validity of reasoning, not coherence; no primed phrases),
                        sampled K times -> mean + confidence interval.
                        Cost: medium. Removes same-model circularity + priming.
  (3) panel           — 2-3 INDEPENDENT judges, mean + inter-rater agreement.
                        Cost: higher. Reduces single-judge variance.

`grade_reasoning(...)` runs whichever graders you enable and returns each score plus a
combined `average` in [0,1]. `batch(...)` runs over a jsonl and writes per-item + summary.

Design rules:
  - never the same model that generated the text          (avoids circularity)
  - never prime the judge to reward the method's phrasing (avoids rigging)
  - grade validity, not coherence                         (avoids fluent-but-wrong)
  - multiple samples / multiple judges                    (quantifies uncertainty)
  - reference-free by default                             (judge the reasoning, not the answer)

Self-contained: copies the few helpers it needs; does NOT import the pipeline.

CLI:
    # free, deterministic only:
    python validation/reasoning_graders.py responses.jsonl --response-key path_b_full_cure --step-only
    # add one independent judge (5 samples):
    python validation/reasoning_graders.py responses.jsonl --judge meta-llama/Llama-3.1-8B-Instruct
    # full panel:
    python validation/reasoning_graders.py responses.jsonl \
        --panel meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.1,google/gemma-2-9b-it
Input jsonl rows need at least: {question, <response-key>, gold(optional)}.
"""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile

# --------------------------------------------------------------------------- #
# shared number helpers (copied from runtime.py so this file stands alone)
# --------------------------------------------------------------------------- #
_NUM = r"-?\d[\d,]*\.?\d*"
_ANSWER_RE = re.compile(
    r"(?:answer\s*(?:is|:)|final answer|####|\\boxed\{|=)\s*\$?(" + _NUM + r")", re.I)
_NUM_RE = re.compile(_NUM)


def _to_float(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_final_number(text):
    if not text:
        return None
    marked = _ANSWER_RE.findall(text)
    if marked:
        return _to_float(marked[-1])
    nums = _NUM_RE.findall(text)
    return _to_float(nums[-1]) if nums else None


def answers_match(pred, gold, tol=1e-4):
    return pred is not None and gold is not None and abs(pred - gold) <= tol


def _wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n -> (lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


# =========================================================================== #
# (1) PROGRAMMATIC STEP VERIFIER  — deterministic, no model
# =========================================================================== #
# Matches "<num> <op> <num> [<op> <num> ...] = <num>", e.g. "12 + 7 = 19",
# "3 x 4 = 12", "1,200 / 4 = 300". Operators: + - * / and x × ÷ aliases.
_STEP_RE = re.compile(
    r"(" + _NUM + r"(?:\s*[-+*/xX×÷]\s*" + _NUM + r")+)\s*=\s*(" + _NUM + r")")


def _safe_arith(expr):
    """Evaluate a pure-arithmetic string safely (digits and + - * / ( ) only)."""
    expr = (expr.replace(",", "").replace("$", "")
                .replace("×", "*").replace("x", "*").replace("X", "*").replace("÷", "/"))
    if not re.fullmatch(r"[0-9.+\-*/() ]+", expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})  # no names/builtins exposed
    except (ZeroDivisionError, SyntaxError, ValueError, TypeError):
        return None


def step_verify(question, response, gold=None, rel_tol=1e-3, abs_tol=1e-4):
    """Deterministic arithmetic-step check.

    Returns {score, n_steps, n_correct, final_number, final_correct, lucky_guess}.
    score = fraction of arithmetic equalities that actually hold. None if no steps.
    lucky_guess = final answer matches gold BUT step score < 1 (right answer, bad work).
    """
    n_steps = n_correct = 0
    for lhs, rhs in _STEP_RE.findall(response or ""):
        lv = _safe_arith(lhs)
        rv = _to_float(rhs)
        if lv is None or rv is None:
            continue
        n_steps += 1
        if abs(lv - rv) <= max(abs_tol, rel_tol * max(abs(lv), abs(rv))):
            n_correct += 1
    score = (n_correct / n_steps) if n_steps else None
    final = extract_final_number(response)
    final_correct = answers_match(final, gold) if gold is not None else None
    lucky = bool(final_correct and score is not None and score < 1.0)
    return {"score": score, "n_steps": n_steps, "n_correct": n_correct,
            "final_number": final, "final_correct": final_correct, "lucky_guess": lucky}


def run_code_unit_tests(code, tests, timeout=10):
    """For code tasks: exec `code` then `tests` (asserts) in a subprocess.

    Returns {score, passed, error}. score=1.0 if the test block runs without error.
    """
    src = code + "\n\n" + tests + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        ok = r.returncode == 0
        return {"score": 1.0 if ok else 0.0, "passed": ok,
                "error": (r.stderr or "").strip()[-500:]}
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "passed": False, "error": "timeout"}


# =========================================================================== #
# (2)/(3) LLM JUDGES  — independent model family, neutral rubric, multi-sample
# =========================================================================== #
# Neutral rubric: grades VALIDITY of reasoning, ignores style, rewards no phrase,
# leaks no gold answer (reference-free intrinsic validity).
JUDGE_SYSTEM = (
    "You are a careful grader of step-by-step mathematical reasoning. Judge ONLY "
    "whether the reasoning is logically valid and every calculation is correct. "
    "Ignore writing style, tone, length, and formatting. Do not reward or penalize "
    "any particular phrases. Use this scale:\n"
    "5 = every step follows logically and all arithmetic is correct\n"
    "4 = essentially sound, at most a trivial slip\n"
    "3 = one clear error or logical gap\n"
    "2 = several errors\n"
    "1 = mostly invalid, contradictory, or incoherent reasoning\n"
    "Output ONLY a single integer from 1 to 5.")

_SCORE_RE = re.compile(r"[1-5]")


class LLMJudge:
    """Loads one HF judge model; scores reasoning K times for a mean + CI.

    Use a DIFFERENT model family than the one that generated the text.
    """

    def __init__(self, model_id, dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.torch = torch
        print(f"[judge] loading {model_id} ...")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=getattr(torch, dtype))

    def _one(self, question, response, temperature):
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",
                 "content": f"Problem: {question}\n\nReasoning to grade:\n{response}"}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=4,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                top_p=0.95, pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        m = _SCORE_RE.search(gen)
        return (int(m.group()) - 1) / 4.0 if m else None   # 1..5 -> 0..1

    def score(self, question, response, k=5, temperature=0.7):
        """Return {score, std, ci_low, ci_high, n, raw}. score = mean over K samples."""
        raw = [s for s in (self._one(question, response, temperature) for _ in range(k))
               if s is not None]
        if not raw:
            return {"score": None, "std": None, "ci_low": None, "ci_high": None,
                    "n": 0, "raw": []}
        mean = statistics.fmean(raw)
        std = statistics.pstdev(raw) if len(raw) > 1 else 0.0
        half = 1.96 * std / math.sqrt(len(raw))
        return {"score": mean, "std": std, "ci_low": max(0.0, mean - half),
                "ci_high": min(1.0, mean + half), "n": len(raw), "raw": raw}

    def free(self):
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def panel_agreement(per_judge_item_scores):
    """Mean pairwise agreement across judges on a binarized (>=0.5) decision.

    per_judge_item_scores: list (judges) of list (items) of score-or-None.
    Returns mean fraction of items where each judge pair gives the same binary label.
    """
    J = len(per_judge_item_scores)
    if J < 2:
        return None
    agrees, pairs = [], 0
    for a in range(J):
        for b in range(a + 1, J):
            sa, sb = per_judge_item_scores[a], per_judge_item_scores[b]
            both = [(x, y) for x, y in zip(sa, sb) if x is not None and y is not None]
            if both:
                agrees.append(sum((x >= 0.5) == (y >= 0.5) for x, y in both) / len(both))
                pairs += 1
    return statistics.fmean(agrees) if pairs else None


# =========================================================================== #
# Aggregator
# =========================================================================== #
def grade_reasoning(question, response, gold=None, judge=None, panel_judges=None,
                    k=5, weights=None):
    """Run enabled graders and return each + a combined `average` in [0,1].

    judge: an LLMJudge instance (grader 2) or None.
    panel_judges: list[LLMJudge] (grader 3) or None. If given, supersedes `judge`
                  for the model-judge component (avoids double-counting).
    weights: dict {"step": w1, "model_judge": w2} for the combined average.
    """
    out = {"step_verifier": step_verify(question, response, gold)}

    model_judge_score = None
    if panel_judges:
        per = [j.score(question, response, k=k) for j in panel_judges]
        scores = [p["score"] for p in per if p["score"] is not None]
        out["panel"] = {
            "score": statistics.fmean(scores) if scores else None,
            "per_judge": {j.model_id: p for j, p in zip(panel_judges, per)},
            "models": [j.model_id for j in panel_judges],
        }
        model_judge_score = out["panel"]["score"]
    elif judge:
        out["independent_judge"] = judge.score(question, response, k=k)
        out["independent_judge"]["model"] = judge.model_id
        model_judge_score = out["independent_judge"]["score"]

    # combined average over available components (step validity + model judge)
    w = weights or {"step": 0.5, "model_judge": 0.5}
    parts, wsum = 0.0, 0.0
    if out["step_verifier"]["score"] is not None:
        parts += w["step"] * out["step_verifier"]["score"]
        wsum += w["step"]
    if model_judge_score is not None:
        parts += w["model_judge"] * model_judge_score
        wsum += w["model_judge"]
    out["average"] = (parts / wsum) if wsum else None
    return out


# =========================================================================== #
# Batch runner
# =========================================================================== #
def batch(rows, response_key, judge_ids=None, panel=False, k=5, out_path=None):
    """Grade a list of dict rows. Loads judge models sequentially (memory-safe).

    Strategy: run the free deterministic verifier on every row first; then, if
    judges are requested, load each judge ONCE and score all rows (so we never
    hold two big models at the same time), then aggregate.
    """
    # 1) deterministic step verifier (free) for all rows
    step = [step_verify(r.get("question", ""), r.get(response_key, ""), r.get("gold"))
            for r in rows]

    # 2) model judges, one model at a time over all rows
    per_judge = {}            # model_id -> list[per-item score dict]
    if judge_ids:
        for mid in judge_ids:
            j = LLMJudge(mid)
            per_judge[mid] = [j.score(r.get("question", ""), r.get(response_key, ""), k=k)
                              for r in rows]
            j.free()

    # 3) assemble per-item + combined average
    items = []
    judge_item_scores = [[pj[i]["score"] for i in range(len(rows))]
                         for pj in per_judge.values()]
    for i, r in enumerate(rows):
        rec = {"i": i, "question": (r.get("question", "") or "")[:80],
               "gold": r.get("gold"), "step_verifier": step[i]}
        comps, wsum = 0.0, 0.0
        if step[i]["score"] is not None:
            comps += 0.5 * step[i]["score"]; wsum += 0.5
        if per_judge:
            ms = [pj[i]["score"] for pj in per_judge.values() if pj[i]["score"] is not None]
            mj = statistics.fmean(ms) if ms else None
            rec["model_judge"] = {mid: per_judge[mid][i] for mid in per_judge}
            rec["model_judge_mean"] = mj
            if mj is not None:
                comps += 0.5 * mj; wsum += 0.5
        rec["average"] = (comps / wsum) if wsum else None
        items.append(rec)

    # 4) dataset-level summary
    def agg(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        m = statistics.fmean(xs)
        half = 1.96 * (statistics.pstdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
        return {"mean": m, "ci_low": max(0.0, m - half), "ci_high": min(1.0, m + half),
                "n": len(xs)}

    summary = {
        "n_rows": len(rows),
        "step_verifier": agg([s["score"] for s in step]),
        "lucky_guess_rate": agg([1.0 if s["lucky_guess"] else 0.0 for s in step]),
        "final_correct_rate": agg([1.0 if s["final_correct"] else 0.0
                                   for s in step if s["final_correct"] is not None]),
        "combined_average": agg([it["average"] for it in items]),
    }
    if per_judge:
        for mid, pj in per_judge.items():
            summary[f"judge::{mid}"] = agg([p["score"] for p in pj])
        if panel or len(per_judge) > 1:
            summary["panel_mean"] = agg([it.get("model_judge_mean") for it in items])
            summary["panel_inter_rater_agreement"] = panel_agreement(judge_item_scores)

    result = {"summary": summary, "items": items}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_path}")
    return result


def _print_summary(summary):
    print("\n" + "=" * 64 + "\nREASONING-QUALITY SUMMARY\n" + "=" * 64)
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"{k:34s}  {v['mean']*100:6.1f}%  [{v['ci_low']*100:.1f}, "
                  f"{v['ci_high']*100:.1f}]  (n={v['n']})")
        else:
            print(f"{k:34s}  {v}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Grade reasoning quality (3 methods + average)")
    ap.add_argument("input", help="jsonl with {question, <response-key>, gold?}")
    ap.add_argument("--response-key", default="response",
                    help="field holding the reasoning text (e.g. path_b_full_cure)")
    ap.add_argument("--step-only", action="store_true", help="deterministic verifier only (free)")
    ap.add_argument("--judge", default=None, help="single independent judge model id")
    ap.add_argument("--panel", default=None, help="comma-separated judge model ids")
    ap.add_argument("--k", type=int, default=5, help="samples per judge")
    ap.add_argument("--out", default=None, help="write full per-item json here")
    a = ap.parse_args()

    rows = []
    with open(a.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"{len(rows)} rows; grading field '{a.response_key}'")

    judge_ids = None
    if not a.step_only:
        if a.panel:
            judge_ids = [m.strip() for m in a.panel.split(",") if m.strip()]
        elif a.judge:
            judge_ids = [a.judge]

    res = batch(rows, a.response_key, judge_ids=judge_ids,
                panel=bool(a.panel), k=a.k, out_path=a.out)
    _print_summary(res["summary"])


if __name__ == "__main__":
    main()
