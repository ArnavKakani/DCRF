"""Shared post-hoc analysis: turn trajectories + ablation paths into per-method
accuracy numbers. Used by stage_results and stage_threshold_downstream so the
"final accuracy / recovery / false-trigger" definitions live in exactly one place.

A "method" applies an intervention only where the detector triggered:
  final outcome(problem) = correct(path) if triggered else baseline_correct
so final accuracy is over the WHOLE dataset, not just triggered cases — which is
what makes the results table interpretable (it answers "what is the end-to-end
accuracy if I deploy this method").
"""

import runtime


def index_by_uid(records):
    """Index ablation records by their stable per-run uid (NOT the question text,
    which is not unique across SVAMP/MultiArith templated problems)."""
    return {r.get("uid"): r for r in records}


def method_summary(traj, abl_idx, path_key):
    """Accuracy of intervening with `path_key` at the detector's trigger points."""
    n = len(traj)
    if n == 0:
        return {"n": 0, "trigger_rate": 0.0, "baseline_accuracy": None,
                "final_accuracy": None, "recovery_on_failures": None,
                "false_trigger_on_correct": None}

    baseline_correct = sum(1 for r in traj if r.get("baseline_correct"))
    triggered = [r for r in traj if r.get("triggered")]
    final_correct = 0
    fail_total = fail_fixed = 0
    for r in traj:
        if r.get("triggered"):
            a = abl_idx.get(r.get("uid"))
            ok = bool(a) and runtime.answers_match(
                runtime.extract_final_number(a.get(path_key, "")), r.get("gold"))
            final_correct += int(ok)
            if not r.get("baseline_correct"):
                fail_total += 1
                fail_fixed += int(ok)
        else:
            final_correct += int(r.get("baseline_correct"))

    false_trig = sum(1 for r in triggered if r.get("baseline_correct"))
    return {
        "n": n,
        "trigger_rate": len(triggered) / n,
        "baseline_accuracy": baseline_correct / n,
        "final_accuracy": final_correct / n,
        "recovery_on_failures": (fail_fixed / fail_total) if fail_total else None,
        "false_trigger_on_correct": (false_trig / baseline_correct) if baseline_correct else None,
    }


def baseline_summary(traj):
    """The no-intervention baseline: final accuracy == baseline accuracy."""
    n = len(traj)
    if n == 0:
        return {"n": 0, "trigger_rate": 0.0, "baseline_accuracy": None,
                "final_accuracy": None, "recovery_on_failures": None,
                "false_trigger_on_correct": 0.0}
    correct = sum(1 for r in traj if r.get("baseline_correct"))
    return {"n": n, "trigger_rate": 0.0, "baseline_accuracy": correct / n,
            "final_accuracy": correct / n, "recovery_on_failures": None,
            "false_trigger_on_correct": 0.0}
