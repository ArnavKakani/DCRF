# auto_pipeline — Methodology

A precise, paper-ready reference for the evaluation pipeline: what
every stage computes, the detection algorithm, the metrics, the recovery ablation, the
scoring, and the methodological choices that make the result trustworthy. Numbers/knobs
here are the source of truth (config.py snapshots into every run's `manifest.json`).

---

## 1. Purpose

The pipeline tests whether a residual-stream drift signal can **detect**
chain-of-thought (CoT) reasoning failure early and whether an inference-time
intervention can **recover** it, under controlled conditions. It runs the full chain once per **(model × dataset × metric)**
cell and writes a self-contained, reproducible run directory per cell.

Design commitments:
- **In-distribution prompting** via each model's chat template (not raw text).
- **Outcome scoring by answer-correctness vs gold** (not an LLM coherence judge).
- **A matched random-trigger control** for every recovery claim.
- **Threshold-free ranking** of the detector via AUC, with confidence intervals.
- **Separation of formatting collapse from genuine reasoning error.**

---

## 2. Unit of work and run layout

`run_all.py` loops `model → dataset → metric` (models in `config.MODELS` order; one model
held in memory at a time). Each cell runs the stage chain:

```
stage0_layer_select → stage1_threshold_sweep → stage2_ablation → stage_labeling
  → stage_threshold_downstream → stage3_robustness → stage4_judge
  → stage_results → stage5_aggregate
```

Output: `runs/<timestamp>-<model>-<dataset>-<metric>/` with `manifest.json` (git SHA,
full config snapshot, per-stage status/timings/artifacts) and all CSV/JSONL artifacts.
`--stages` runs a subset; `--profile {smoke,default,large}` scales sample sizes;
`--model/--dataset/--metric` restrict the matrix. Stage modules never hardcode a layer or
threshold — those flow from Stage 0/1 into everything downstream.

---

## 3. Prompting and scoring (the trust foundation)

**Prompting.** With `USE_CHAT_TEMPLATE=True`, every problem is rendered through the
model's own chat template (`build_chat_prompt` → `tokenizer.apply_chat_template(...,
add_generation_prompt=True)`), with a system message (`GSM8K_SYSTEM`) instructing the
model to "end with 'The answer is N'." Tokenization uses `prepend_bos=False` (the template
already carries the model's leading tokens). This is the fix that lifts the Qwen2.5 GSM8K
baseline from an off-distribution **0.40** (raw `System:/User:` text) to **0.90**, and
removes the formatting-collapse confound. `USE_CHAT_TEMPLATE=False`
selects raw-prompt rendering for comparison.

**Scoring.** Correctness is programmatic: `extract_final_number` prefers an explicit
answer marker ("answer is", "=", "####", `\boxed{}`) and falls back to the last number;
`answers_match` compares to the parsed gold within tolerance. **No LLM judge is used for
the headline metrics.** (A separate optional soundness judge exists in stage4 but is not
the basis of any accuracy claim.)

---

## 4. Detection algorithm (`runtime.detect`)

For one problem at one layer with one metric, generate up to
`MAX_NEW_TOKENS_DETECTION` tokens and monitor the residual stream:

1. **Sampling** (induces the drift we then try to catch): temperature `0.85`, top-k `50`,
   seed `BASE_SEED=42`. Recovery generations are greedy.
2. At each generated token `i`, capture the residual state at the monitored layer:
   `h_raw = cache["blocks.{L}.hook_resid_post"][:, -1, :]`; normalize
   `h_norm = h_raw / (‖h_raw‖₂ + 1e-8)`.
3. **Healthy window** (`HEALTHY_WINDOW=10`): the first 10 tokens define the per-problem
   reference; `ref = metric.build_reference(healthy states)` is built at `i = 9`.
4. **Post-healthy:** push `h_norm` into a sliding `recent` deque (`RECENT_WINDOW=5`). A
   set metric scores only once the window is full; a point metric scores every step.
   `dist = metric.score(h_norm, h_raw, recent, ref)`.
5. **Tare** (`TARE_WINDOW_SIZE=5`): the first 5 post-healthy distances set
   `baseline_mean` and `baseline_std = max(std, 1e-5)`. This makes the detector
   instance-specific and avoids a global threshold.
6. **z-score:** `z_i = (dist_i − baseline_mean) / baseline_std`. The full per-token
   `z_trace` is recorded; `max_z` is tracked.
7. **Trigger:** the first step where `z_i > z_threshold` sets `spike_idx`, `trigger_z`,
   `trigger_token`, and `tokens_at_trigger` (the token state recovery rewinds from).
8. **Text-break index:** `break_idx` = the first generated step emitting a
   formatting/template token (`is_formatting_trigger` vs `FORMATTING_MARKERS`). The lead
   time **D2H = break_idx − spike_idx** measures how far the latent spike precedes the
   visible break.

`detect_multilayer` scores every candidate layer from a *single* generation (used by
Stage 0, far cheaper than re-generating per layer).

---

## 5. Drift metrics (`drift_metrics.py`)

Each metric implements `build_reference(healthy_norm, healthy_raw)` and
`score(h_norm, h_raw, recent_norm, ref)`; `detect` z-standardizes the score.

- **`l2`** (point): `ref` = mean of the 10 healthy normalized states; score =
  `Σ(h_norm − ref)²` (squared Euclidean to the healthy mean direction). `uses_set=False`.
- **`sinkhorn`** (set-vs-set OT): `ref` = the **cloud** of healthy normalized states (μ);
  score = entropy-regularized Wasserstein-2 between the recent-window cloud (ν_t) and μ,
  via `geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.1, backend="tensorized")`.
  `uses_set=True`. Comparing a single current vector to a single mean anchor
  instead would degenerate to ≈ squared L2, so this is the only setting in which OT
  differs from `l2`.
- **`cosine`, `residual_norm`** (point baselines; available, dropped from the default set
  because `cosine ≈ l2` and to bound compute).
- **`random`** (`is_control=True`): no geometry — triggers at a uniformly random
  post-tare position (`random_trigger_index`), skips layer/threshold selection. This is
  the matched-position control: any recovery method must beat it.

---

## 6. Stage 0 — layer selection (`stage0_layer_select`)

Sweep `LAYER_BAND = [18, 20, 22, 24, 26]` over `LAYER_SELECT_SAMPLES=20` target problems
and `LAYER_SELECT_CONTROL_SAMPLES=20` problems from each `CONTROL_DATASETS` entry
(`cnn_dailymail`, `mbpp`), scoring all layers per generation. Per layer:

- `tpr` = fraction of target problems with `max_z > LAYER_SELECT_REF_Z (12.5)`.
- `fpr` = fraction of control problems with `max_z > 12.5`.
- `target_mean_z`, `control_mean_z`, and **`z_gap = target_mean_z − control_mean_z`**.

The selected layer **maximizes `z_gap`** (tie-broken by tpr) — the continuous separation
between drift (should be high on the reasoning task) and healthy (should stay low on the
controls). z_gap is **threshold-free**, so it cannot go degenerate the way `tpr − fpr`
does when every problem lands on one side of the 12.5 cutoff. `layer_select.csv` reports
`layer, tpr, fpr, separation(tpr−fpr), target_mean_z, control_mean_z, z_gap, mean_d2h`;
`layer_traces.csv` and `cascade_trace.json` hold the per-token traces for the figures.

---

## 7. Stage 1 — threshold calibration (`stage1_threshold_sweep`)

At the chosen layer, score the z-traces from Stage 0 against each Z in
`Z_GRID = [2, 4, 6, 8, 10, 12.5, 15]` on a held-out **fit/val split**. Select the
**highest** Z whose fit trigger-rate ≥ `TARGET_TPR (0.60)` and control trigger-rate ≤
`MAX_FPR (0.20)`; if none qualifies, fall back to the best fit `(TPR − FPR)`.
`threshold_sweep.csv` records `z, tpr, fpr, separation, fit/val_tpr, fit/val_fpr,
selected`. This is the data that *justifies* the operating Z rather than asserting it.

---

## 8. Stage 2 — recovery ablation (`stage2_ablation`)

On `ABLATION_SAMPLES=100` test problems: run `detect` (full trajectory) → record the
baseline answer/correctness; if the problem **triggered**, generate four recovery paths
from `tokens_at_trigger`, all **greedy** (`MAX_NEW_TOKENS_RECOVERY=256`):

| Path | Intervention |
|---|---|
| **A — baseline** | continue generating, no intervention (do-nothing) |
| **B — full cure** | rewind `REWIND_TOKENS=15` tokens **and** inject `COGNITIVE_RESET_TEXT` ("Wait, let me recalculate…") |
| **C — prompt only** | inject the reset text without rewinding |
| **D — rewind only** | rewind 15 tokens, no reset text |

Records carry a stable per-run `uid` (joined downstream), `gold`, `baseline_correct`,
`spike_idx`, `break_idx`, `trigger_token`, and the four path texts. Isolating the four
paths separates the effect of *rewinding* from *reprompting* and tests whether the
intervention beats doing nothing.

---

## 9. Labeling — the headline metrics (`stage_labeling`)

From the ablation + trajectory records (joined on `uid`):

- **`auc_maxz_vs_wrong`** — AUC of `max_z` predicting a wrong final answer
  (threshold-free; 0.5 = chance). The cleanest test of "does the drift signal carry
  information about reasoning failure."
- **`false_trigger_rate`** — fraction of *correct* answers the detector fires on.
- **Formatting vs genuine split** — `is_formatting_trigger` (token at/near the spike is a
  chat/role/end-of-text marker per `FORMATTING_MARKERS`) separates template collapse from
  genuine reasoning error; `genuine_error_fraction`, `n_genuine`.
- **`genuine_flip_rate`** per path — of the *genuine* errors, the fraction the path flips
  wrong→right (the only recovery number that is not contaminated by false triggers).
- **5-way failure typology** (`classify_failure`): `arithmetic_logical`, `semantic_drift`,
  `format_template_collapse`, `premature_termination`, `other_unclear` →
  `failure_types.csv`; before/after `examples.md`.

---

## 10. Downstream, robustness, judge

- **`stage_threshold_downstream`** — re-runs detection+recovery at each Z in `Z_GRID` on
  `THRESHOLD_DOWNSTREAM_SAMPLES=20` problems → trigger rate, recovery rate, false-trigger
  rate, and **final task accuracy per Z** (`threshold_downstream.csv`). Shows the
  real-world cost/benefit of the threshold, not just its trigger statistics. (This is the
  most expensive stage — ~7×20 detect+recover runs per cell.)
- **`stage3_robustness`** — repeats detection over `ROBUSTNESS_SEEDS=[42..46]` ×
  `ROBUSTNESS_SAMPLES=30` to report trigger-rate **mean/variance**.
- **`stage4_judge`** — an optional same-model "soundness" pass (Wilson CIs). Retained for
  completeness and explicitly **not** used for any accuracy claim (a same-model coherence
  judge rewards fluent text regardless of correctness).

---

## 11. Results and aggregation

- **`stage_results`** → `results_table.csv`: per-method final accuracy
  (`baseline_cot`, `prompt_only`, `sdr_<metric>` or `random_trigger`). `final_accuracy`
  uses the path output on triggered problems and the baseline otherwise
  (`analysis.method_summary`, joined on `uid`).
- **`stage5_aggregate`** → `runs/master_results.csv`: one row per (cell × path) with
  `auc_maxz_vs_wrong`, `mean/std_trigger_rate`, `false_trigger_rate`,
  `genuine_error_fraction`, `n_genuine`, `recovery_rate` + Wilson CI, and
  `genuine_flip_rate` — the cross-everything head-to-head table.

Figures rebuild from `runs/` with **no GPU** via `figures/make_figures.py` (Fig 1 spikes,
Fig 2 D2H/FPR-vs-layer tradeoff, Fig 3 cascade, plus metric/recovery/threshold/
formatting-split/failure-type plots and the LaTeX tables).

---

## 12. Datasets and controls

- **Targets** (`DATASET_ADAPTERS`): `gsm8k` (`gsm8k/main/test`), `svamp`
  (`ChilleD/SVAMP`), `multiarith` (`ChilleD/MultiArith`). Each adapter supplies
  `question` and `gold`; items with unparseable gold are dropped so accuracy denominators
  aren't biased.
- **Negative controls** (`CONTROL_DATASETS`, used only for FPR in layer/threshold
  selection): `cnn_dailymail` (summarization), `mbpp` (code) — tasks where the model
  should *not* exhibit reasoning drift, so any triggering is a false positive.

---

## 13. Configuration (the knobs that define a run)

All in `config.py`, snapshotted into each `manifest.json`. Key values:
`MODELS`, `METRICS=["l2","sinkhorn","random"]`, `TARGET_DATASETS=["gsm8k","svamp"]`,
`USE_CHAT_TEMPLATE=True`, `LAYER_BAND`, `Z_GRID`, `TARGET_TPR=0.60`, `MAX_FPR=0.20`,
`ABLATION_SAMPLES=100`, `THRESHOLD_DOWNSTREAM_SAMPLES=20`, `ROBUSTNESS_SEEDS/SAMPLES`,
`HEALTHY_WINDOW=10`, `RECENT_WINDOW=5`, `TARE_WINDOW_SIZE=5`,
`MAX_NEW_TOKENS_DETECTION=400`, `MAX_NEW_TOKENS_RECOVERY=256`, `REWIND_TOKENS=15`,
`COGNITIVE_RESET_TEXT`, `TEMPERATURE=0.85`, `TOP_K=50`, `BASE_SEED=42`,
`FORMATTING_MARKERS`, `GSM8K_SYSTEM`. Profiles: `smoke` (tiny wiring check), `default`
(the values above), `large` (N≥1000 for significance — heavy).

---

## 14. Reproducibility

- Every run dir carries `manifest.json` = git SHA + full config snapshot + per-stage
  status/timings/artifacts → the exact parameters that produced any result are
  recoverable from the run directory.
- Fixed seeds (`BASE_SEED=42`) for detection sampling; greedy recovery is deterministic.
- One model loaded at a time; `--resume` continues a partial run; `--stages` re-runs a
  subset.

---

## 15. Methodological limitations

- **Statistical power vs baseline accuracy.** AUC ranks the *wrong* answers; a model with
  a high baseline (e.g. Qwen2.5 at 0.90) yields few failures per 100, so its AUC has wide
  CIs. Weaker models (more failures) give a better-powered estimate — report AUC alongside
  `n_genuine` and CIs, not as a point value.
- **Detection is O(n²) per token** (a fresh `run_with_cache` over the growing prefix) —
  the dominant cost; `--profile large` is expensive.
- **Set-metric horizon.** The set-vs-set `sinkhorn` waits for a full `RECENT_WINDOW`, so
  it starts scoring ~4 tokens later than the point metrics; note or equalize when
  comparing metrics.
- **Scope.** Short math CoT (GSM8K/SVAMP). A different failure definition (e.g. long-horizon
  agentic drift) or a different signal might behave differently; the conclusions are about
  the detector and intervention *as defined and measured here*.
