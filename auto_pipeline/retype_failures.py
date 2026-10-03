"""Re-run ONLY the failure typology over existing run directories (no model).

Why: the pre-2026-09-01 `classify_failure` scanned the full decoded path text
(prompt included) for chat-template markers, so every ChatML-model (Qwen, Yi)
failure was labeled "format_template_collapse" and every Mistral failure fell
through to "arithmetic_logical" regardless of content. This script rewrites
`failure_types.csv` and the `stage_labeling.failure_types` block in each
manifest using the corrected classifier, leaving every other artifact untouched.

Usage:  python auto_pipeline/retype_failures.py [runs_dir]
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg                      # noqa: E402
from failure_typology import FAILURE_TYPES, classify_failure   # noqa: E402


def main(runs_dir):
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        abl_path = os.path.join(d, "ablation.jsonl")
        man_path = os.path.join(d, "manifest.json")
        if not (os.path.isfile(abl_path) and os.path.isfile(man_path)):
            continue
        abl = [json.loads(l) for l in open(abl_path) if l.strip()]
        failures = [r for r in abl if not r.get("baseline_correct")]
        counts = {t: 0 for t in FAILURE_TYPES}
        for r in failures:
            counts[classify_failure(r, cfg)] += 1
        nf = max(len(failures), 1)
        with open(os.path.join(d, "failure_types.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["failure_type", "count", "fraction"])
            for t in FAILURE_TYPES:
                w.writerow([t, counts[t], counts[t] / nf])
        man = json.load(open(man_path))
        lab = man.setdefault("stages", {}).setdefault("stage_labeling", {})
        lab["failure_types"] = counts
        lab["n_failures_typed"] = len(failures)
        lab["failure_types_retyped"] = "2026-09-01 corrected classifier (prompt stripped, EOS ignored)"
        with open(man_path, "w") as f:
            json.dump(man, f, indent=2, default=str)
        print(f"{os.path.basename(d)[:58]:58s} n={len(failures):3d} {counts}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "runs"))
