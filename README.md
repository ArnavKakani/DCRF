# Detecting and Correcting Reasoning Failures: A Controlled Negative Result for a Residual-Stream Monitor

Code and artifacts for the paper by Arnav Kakani and Mrinal Agarwal, MATH-AI workshop at NeurIPS 2026.

A residual-stream monitor is a detector that tracks how far a language model's hidden state moves away from its state at the start of a solution. When the distance crosses a threshold, the generation is rolled back a few tokens and re-prompted, in the hope of recovering from a drifting chain of thought. The paper asks whether such a monitor detects reasoning failures better than chance and whether its intervention fixes them, using answer correctness against gold labels, a matched random-trigger control, and a prompt-only control. The main evaluation uses two open 7B models (Qwen2.5-7B-Instruct on GSM8K and SVAMP, Mistral-7B-Instruct-v0.1 on GSM8K); Yi-6B-Chat on GSM8K is kept only for the failure typology. The monitor provides no consistent accuracy improvement over the no-intervention and random-trigger controls, and a same-model coherence judge reports high "recovery" for every path, including doing nothing. The paper reports this as a negative result and documents the conditions under which it holds.

## What is in the repo

```
README.md
requirements.txt              Python dependencies (transformers 4.45.2 pinned)
auto_pipeline/                the evaluation pipeline
  config.py                   every constant: models, metrics, layer band, Z grid, seeds, budgets
  run_all.py                  driver, one (model, dataset, metric) cell at a time
  runtime.py                  model loading, prompting, detection, recovery paths, answer extraction
  drift_metrics.py            l2, sinkhorn, cosine, residual_norm score definitions
  stage0_layer_select.py      layer selection by z_gap
  stage1_threshold_sweep.py   threshold sweep on a held-out fit/validation split
  stage2_ablation.py          detection and the four recovery paths
  stage_labeling.py           AUC, formatting-versus-genuine split, flip rates, failure types
  stage_threshold_downstream.py  detection and recovery at each Z, with final accuracy
  stage3_robustness.py        trigger rate across five seeds
  stage4_judge.py             same-model soundness judge
  stage_results.py            per-method accuracy tables
  stage5_aggregate.py         cross-run rollup
  analysis.py, failure_typology.py, retype_failures.py   shared helpers and the failure classifier
  figures/make_paper_figures.py   rebuilds figures and tables from runs/, no model needed
  METHODOLOGY.md, README.md   stage-by-stage description
pipelines/                    two standalone ablation scripts (L2, Sinkhorn) and drift_metrics.py
validation/reasoning_graders.py   self-contained step-verifier and judge graders
analysis/camera_ready_stats.py    statistics and the trigger-position figure, computed from runs/
runs/                         eleven run directories plus master_results.csv and results_table.csv
figs/                         the four figures the paper uses (pdf, png) and tab_detection.tex, tab_recovery.tex
paper/                        submission_mathai.tex, supplementary.tex, checklist_filled.tex,
                              neurips_2026.sty, references.bib, si_config_excerpt.py,
                              si_drift_metrics.py, figs/ (the four figure PDFs the tex reads)
```

## How to reproduce

### Environment

- Python 3 and the packages in `requirements.txt`: `pip install -r requirements.txt`. The file pins `transformers==4.45.2` and `huggingface_hub==0.25.2`; `torch` and the rest are unpinned.
- A CUDA GPU that holds a 7B model in bfloat16 (the weights alone are about 14 GB). Models are loaded one at a time. Per the timestamps in each run's `manifest.json`, the eleven runs in `runs/` took between 0.5 and 2.7 hours each on single NVIDIA H100 and A100 instances rented from Lambda, one model at a time.
- `transformer-lens` (`transformer_lens`) hooks the residual stream, so each model must be one it supports. `geomloss` (with `pykeops`) provides the Sinkhorn distance.
- The models and datasets download from Hugging Face on first use. The `openai` package is imported for the optional judge stage.

### Running the pipeline

Run from the repository root:

```
python auto_pipeline/run_all.py --model "mistralai/Mistral-7B-Instruct-v0.1" \
    --dataset gsm8k --metric l2,sinkhorn,random
```

Flags, from the argparse in `auto_pipeline/run_all.py`:

- `--model`: model id or comma-separated ids, or `all` (the list in `config.MODELS`).
- `--dataset`: target dataset or comma-separated datasets, or `all` (`config.TARGET_DATASETS`).
- `--metric`: `all`, `both`, or a comma-separated subset of `l2,sinkhorn,cosine,residual_norm,random`. `random` is the random-trigger control.
- `--stages`: comma-separated stage names to run; the default is every stage.
- `--profile`: `smoke` (wiring check), `default` (N=100, what the paper used) or `large`.
- `--resume`: an existing run directory to resume, for a single model and a single metric.
- `--gpus`: GPU indices (`0,1`) or a count (`2`), for parallel runs.

The seed is 42 (`config.BASE_SEED`). Each cell writes one directory under `runs/`.

### What a run directory contains

`runs/<timestamp>-<model>-<dataset>-<metric>/`

- `manifest.json`: model, dataset, git SHA, full config snapshot, per-stage status and timing, selected layer and Z, labeling and failure-type summaries. Method names such as `"method": "sdr_l2"` appear here as data keys.
- `pipeline.log`: the run's log.
- `layer_select.csv`: per-layer separation statistics and `z_gap` (stage 0).
- `layer_traces.csv`: per-problem, per-layer maximum z and spike timing (stage 0).
- `cascade_trace.json`: one example's per-layer z trace (stage 0).
- `threshold_sweep.csv`: TPR and FPR per Z on the fit split and the selected Z (stage 1).
- `trajectories.jsonl`: one row per problem with uid, `max_z`, `spike_idx`, `triggered`, gold, `baseline_correct` and the trigger token (stage 2).
- `ablation.jsonl`: the four recovery paths (A baseline continuation, B rewind and re-prompt, C prompt only, D rewind only) for each triggered problem (stage 2).
- `spike_traces.jsonl`: per-token z traces of triggered cases (stage 2).
- `labeling.csv`: per-path wrong-to-right flips on the genuine-error subset (labeling).
- `failure_types.csv`: failure typology counts (labeling).
- `threshold_downstream.csv` and `threshold_downstream/`: trigger rate, recovery rate, false-trigger rate and final accuracy at each Z, with per-Z `ablation_z*.jsonl` and `trajectories_z*.jsonl` (Qwen GSM8K L2 only).
- `robustness.csv` and `robustness/ablation_seed42..46.jsonl`: trigger rate across five seeds (Qwen GSM8K L2 only).
- `judge_results.csv`: per-path judged and parseable rates for the same-model judge.
- `results_methods.csv`: per-method final accuracy for this run. The method key for the full intervention is `sdr_<metric>`, kept as a data value.
- `master_results.csv`: this run's rows of the cross-run rollup.
- `plots/`: quick-look diagnostic images.

Not every directory has every file. Random-trigger runs have no layer selection or threshold sweep. Only the Qwen GSM8K L2 run has the robustness and threshold-downstream outputs. The Yi sinkhorn run is incomplete and has no labeling, judge or results files. `runs/master_results.csv` and `runs/results_table.csv` aggregate across directories. Sampled generation text is not stored in the run artifacts, and per-case judge verdicts were not stored.

### Regenerating figures and tables without a model

These read only `runs/` and need no GPU or torch:

```
python auto_pipeline/figures/make_paper_figures.py --out figs
python analysis/camera_ready_stats.py
```

`make_paper_figures.py` writes the intervention-cost, typology and judge figures, three further figures the paper does not include (`fig_detection_auc`, `fig_final_accuracy`, `fig_genuine_flip`), and `tab_detection.tex` and `tab_recovery.tex`. Yi-6B-Chat is skipped by default; add `--include-yi` to include it, which the typology figure needs to show the Yi row. The two tables regenerate identically without `--include-yi`; the committed `figs/` tables contain no Yi rows. Table 1 in the paper is typed by hand from these values.

`analysis/camera_ready_stats.py` needs `numpy`, `scikit-learn` and `matplotlib`. It prints the AUC on the 80 problems disjoint from layer selection with bootstrap intervals, the path A and path B accuracies and paired differences, the matched-timing comparison, the judge populations and the trigger-position statistics, and writes `figs/fig_trigger_positions.{pdf,png}`. It reads the answer-extraction helpers out of `auto_pipeline/runtime.py` as text, so it does not import torch.

### Building the paper

```
tectonic paper/submission_mathai.tex
tectonic paper/supplementary.tex
```

The main file reads figures from `paper/figs/`, which holds copies of the four figure PDFs.

## Where each number comes from

| Paper item | Source |
|---|---|
| Table 1: AUC, n_wrong, n_gen, false-trigger rate | `runs/master_results.csv` (`auc_maxz_vs_wrong`, `n_genuine`, `false_trigger_rate`), written by `stage5_aggregate.py`; same values in `figs/tab_detection.tex` |
| Table 1: accuracy of baseline, prompt-only, full | `runs/results_table.csv` and per-run `results_methods.csv` (`baseline_cot`, `prompt_only`, `sdr_<metric>`); `figs/tab_recovery.tex` |
| Table 1: random-trigger accuracy | the `*-random` run directories, `random_trigger` row of `results_methods.csv` |
| Mistral AUC 0.514 and 0.531, 10,000-shuffle p-values | `trajectories.jsonl` (`max_z`, `baseline_correct`); AUC values printed by `analysis/camera_ready_stats.py`; the permutation test is not in a committed script |
| AUC on the 80 disjoint problems, 43 wrong | `analysis/camera_ready_stats.py`, section 1 |
| Sampled baseline versus greedy continuation, paired intervals, correct-to-wrong counts | `analysis/camera_ready_stats.py`, section 2, from `ablation.jsonl` |
| Matched-timing comparison on 35 shared cases | `analysis/camera_ready_stats.py`, section 3 |
| Wilson intervals on Mistral accuracy, truncation counts | derived from `results_methods.csv` and `ablation.jsonl`; no committed script |
| Table 2: layer, z_gap, threshold, fit TPR and FPR | `layer_select.csv`, `threshold_sweep.csv`, `manifest.json` (`selected_layer`, `selected_z`) |
| Figure: intervention cost and trigger rate across Z | Qwen GSM8K L2 `threshold_downstream.csv` and `threshold_downstream/` |
| Trigger rate across five seeds | Qwen GSM8K L2 `robustness.csv` |
| Table 3 and typology figure | per-run `failure_types.csv`, folded by `make_paper_figures.py`; classifier in `failure_typology.py` and `retype_failures.py`; Yi row needs `--include-yi` |
| Counts of correct continuations (6 of 50, 5 of 29) | `labeling.csv` and `ablation.jsonl`; no committed script |
| Judge figure and judge rates | `judge_results.csv` joined with `labeling.csv` in `make_paper_figures.py`; judge code in `stage4_judge.py` |
| Trigger-position figure and medians | `trajectories.jsonl` field `spike_idx` where `triggered` is true; `analysis/camera_ready_stats.py`, section 5 |
| Chat template raising Qwen GSM8K accuracy from 0.40 to 0.90 | not in `runs/`; all reported runs use the chat template, and the 0.40 figure comes from an earlier raw-prompt run that is not included |
| Setup constants (temperature, top-k, budgets, seed, window sizes) | `auto_pipeline/config.py` (excerpt in `paper/si_config_excerpt.py`) and each `manifest.json` |
| Drift score definitions | `auto_pipeline/drift_metrics.py` (verbatim in `paper/si_drift_metrics.py`) |

## Licenses

- GSM8K and SVAMP: MIT.
- Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.1: Apache-2.0.
- Yi-6B-Chat: the Yi license (see the model card).
- transformer_lens and geomloss: MIT.

This repository is released under the MIT License (see `LICENSE`). The run artifacts derive from GSM8K and SVAMP (MIT) and from model outputs under the model licenses listed above.

## Citation

```
@inproceedings{kakani2026dcrf,
  title     = {Detecting and Correcting Reasoning Failures: A Controlled Negative Result for a Residual-Stream Monitor},
  author    = {Kakani, Arnav and Agarwal, Mrinal},
  booktitle = {The 6th Workshop on Mathematical Reasoning and AI (MATH-AI) at NeurIPS 2026},
  year      = {2026}
}
```
