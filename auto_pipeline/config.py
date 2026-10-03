"""Central configuration for the residual-stream drift monitor evaluation pipeline.

A run is fully described by this file plus the chosen --metric. run_all.py
snapshots every UPPER_CASE name here into each run's manifest.json, so the exact
parameters that produced a result are always recoverable from the run directory.

Tune here, not in the stage modules. Nothing in the pipeline hardcodes a layer or
threshold: Stage 0 selects the layer and Stage 1 calibrates the Z-threshold, and
both flow downstream from the values in this file's *grids*.
"""

import os

# ---------------------------------------------------------------------------
# Models. The pipeline runs the full chain for each (model x metric) pair, so
# adding a model here is the second-model generalization study (issue #12). Only
# one model is held in memory at a time. MODEL_ID is the default / first model.
# ---------------------------------------------------------------------------
# ORDER MATTERS: run_all.py (sequential) executes models top-to-bottom, so if a run
# is cut short we still have the priority model (Qwen2.5, continuity) complete first.
#
# Working set = ungated HF repos that the rented transformer_lens build supports. The
# earlier plan's models had to be dropped on the rented stack:
#   - Qwen/Qwen3-8B                       : not in this transformer_lens build's list
#   - meta-llama/Llama-3.1-8B-Instruct    : GATED repo (401 without HF auth + license)
#   - google/gemma-2-9b-it                : GATED repo (Gemma license / HF auth)
# Substituted with ungated, tlens-supported, different-FAMILY models so the
# multi-model generalization study (issue #12) still holds (lower baselines than
# Qwen2.5's 0.90 also mean MORE failures -> better-powered AUC).
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",            # run on gsm8k+svamp
    "mistralai/Mistral-7B-Instruct-v0.1",  # Mistral family (open)
    "01-ai/Yi-6B-Chat",                    # Yi family (open)
    "microsoft/Phi-3-mini-4k-instruct",    # Phi family (open; strong at math)
]
# Re-enable on a stack with HF auth + a newer transformer_lens:
#   "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"
MODEL_ID = MODELS[0]  # default/first model

# Prompt rendering. True = render every prompt through the model's OWN chat template
# (correct, in-distribution; this is the fix that lifts Qwen2.5 GSM8K baseline 0.40 -> 0.90
# and removes the formatting-collapse artifact). This is the regime used for the reported runs.
# Set False to use raw-text prompting instead (formatting collapse inflates trigger rates).
USE_CHAT_TEMPLATE = True

# ---------------------------------------------------------------------------
# Metrics ("versions") to run. Each key indexes drift_metrics.METRICS. The full
# stage chain runs once per metric. "random" is the matched-position control
# (rollback+reprompt at a random token) and skips layer/threshold selection.
# ---------------------------------------------------------------------------
METRICS = ["l2", "sinkhorn", "random"]   # cosine dropped (≈ l2; saves a full chain/cell)

# ---------------------------------------------------------------------------
# Stage 0 — layer selection (real sweep, pick best separator)
# ---------------------------------------------------------------------------
LAYER_BAND = [18, 20, 22, 24, 26]   # candidate monitoring layers
LAYER_SELECT_SAMPLES = 20           # GSM8K problems used to score each layer (TPR)
LAYER_SELECT_CONTROL_SAMPLES = 20   # control-task problems per control dataset (FPR)
LAYER_SELECT_REF_Z = 12.5           # reference threshold used to score separation
LAYER_SELECT_ROLLOUTS = 1           # rollouts per (problem, layer); >1 reduces noise

# ---------------------------------------------------------------------------
# Stage 1 — threshold calibration. Picks the operating Z from Z_GRID at the
# layer Stage 0 chose: the highest Z whose target trigger-rate >= TARGET_TPR
# while control trigger-rate <= MAX_FPR. Falls back to best (TPR - FPR).
# ---------------------------------------------------------------------------
Z_GRID = [2.0, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0]   # threshold grid for the downstream sweep
TARGET_TPR = 0.60
MAX_FPR = 0.20

# Threshold sweep WITH downstream metrics (recovery + final accuracy per Z). This
# re-runs detection+recovery at every Z, so it uses a smaller sample to stay cheap.
THRESHOLD_DOWNSTREAM_SAMPLES = 20   # budget run: 20x7=140 vs 50x7=350 detect+recover/cell

# ---------------------------------------------------------------------------
# Stage 2 / 3 — recovery ablation + robustness
# ---------------------------------------------------------------------------
ABLATION_SAMPLES = 100              # GSM8K test problems for the main ablation
ROBUSTNESS_SEEDS = [42, 43, 44, 45, 46]
ROBUSTNESS_SAMPLES = 30            # smaller per-seed sample to bound compute
# Generation budgets — set to 250/150. (The chat-template path can truncate long solutions at 250;
# if you flip USE_CHAT_TEMPLATE=True, consider raising these to 400/256.)
MAX_NEW_TOKENS_DETECTION = 400   # pinned to the values recorded in runs/*/manifest.json
MAX_NEW_TOKENS_RECOVERY = 256    # (the 2026-06-24/25 runs used 400/256, not 250/150)
REWIND_TOKENS = 15
COGNITIVE_RESET_TEXT = "\nWait, let me recalculate this step to be sure:\n"

# ---------------------------------------------------------------------------
# Detection geometry (shared by all metrics)
# ---------------------------------------------------------------------------
HEALTHY_WINDOW = 10                 # first N tokens define the "healthy" reference
RECENT_WINDOW = 5                   # sliding window size for set-vs-set (sinkhorn)
TARE_WINDOW_SIZE = 5                # tokens used to estimate the per-problem baseline

# ---------------------------------------------------------------------------
# Sampling during the detection phase (induces the drift we then catch)
# ---------------------------------------------------------------------------
TEMPERATURE = 0.85
TOP_K = 50
BASE_SEED = 42

# ---------------------------------------------------------------------------
# Labeling stage — separate formatting collapse from genuine reasoning error.
# A trigger is "formatting" if the token at the spike (or within the rewind
# window) is a chat-template / role / end-of-text marker rather than content.
# Without this split, trigger rates are inflated by formatting glitches.
# ---------------------------------------------------------------------------
FORMATTING_MARKERS = [
    "<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|end|>", "<|eot_id|>",
    "endoftext", "im_start", "im_end",
    "Human:", "User:", "Assistant:", "System:", "\nHuman", "\nUser", "\nAssistant",
]

# ---------------------------------------------------------------------------
# Figure data. The pipeline saves the per-token Z-traces needed to rebuild the
# paper figures WITHOUT reloading the model (the old plot scripts re-ran the 7B).
# ---------------------------------------------------------------------------
FIGURE_TRACE_LIMIT = 12     # how many triggered spike traces to save (Fig 1 overlay)

# ---------------------------------------------------------------------------
# Stage 4 — judge
# ---------------------------------------------------------------------------
SECOND_JUDGE = False               # set True to also run an independent judge pass (TODO #14)
JUDGE_MAX_NEW_TOKENS = 8

# ---------------------------------------------------------------------------
# Target datasets — the reasoning tasks. The full chain runs once per
# (model x dataset x metric). Adding a dataset here is the generalisation study.
# Each key must have an adapter in runtime.DATASET_ADAPTERS (question/gold/prompt).
# ---------------------------------------------------------------------------
TARGET_DATASETS = ["gsm8k", "svamp"]
# Available generalisation datasets (add to TARGET_DATASETS to run):
#   "svamp"       — ChilleD/SVAMP   (arithmetic word problems)
#   "multiarith"  — ChilleD/MultiArith
TARGET_DATASET = ("gsm8k", "main", "test")  # kept for back-compat / control loader

# Negative-control datasets used only for the false-positive rate in layer/threshold
# selection. Tuples are (hf_name, hf_config_or_None, split).
CONTROL_DATASETS = [
    ("cnn_dailymail", "3.0.0", "test"),
    ("mbpp", None, "test"),
]

# ---------------------------------------------------------------------------
# Prompt templates ({q} is replaced with the problem text)
# ---------------------------------------------------------------------------
# GSM8K_SYSTEM is the system message used with the model's chat template (the
# correct, in-distribution path). The "end with 'The answer is N'" instruction
# anchors answer extraction. GSM8K_PROMPT is the legacy raw-text fallback kept
# only for base models without a chat template (model=None path).
GSM8K_SYSTEM = ("Solve the math problem step by step using basic arithmetic. "
                "End your response with 'The answer is N' where N is the final number.")
GSM8K_PROMPT = "System: Solve step-by-step using basic arithmetic.\nUser: {q}\nResponse:"
CNN_PROMPT = "Summarize: {q}\nSummary:"
MBPP_PROMPT = "System: You are an expert Python programmer.\nUser: {q}\nCode:"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PKG_DIR)          # repository root
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")    # all run history lands here
