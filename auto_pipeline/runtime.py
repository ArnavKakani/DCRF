"""Shared runtime for the residual-stream drift monitor evaluation pipeline.

One model load, one detection engine, one recovery routine, and the run-history
machinery — all stages import from here so nothing is duplicated and the 7B model
is loaded exactly once per `run_all.py` invocation (the old loose scripts reloaded
it ~14x).

Public surface:
    load_model(cfg)                          -> HookedTransformer (cached)
    load_problems(cfg, which, n)             -> list[str] of formatted prompts
    detect(model, prompt, metric, layer, z, seed, cfg, full_trajectory=False)
    detect_multilayer(model, prompt, metric, layers, seed, cfg)   -> {layer: max_z}
    recovery_paths(model, current_tokens, cfg)                    -> dict[str,str]
    sample_trajectory(model, prompt, seed, cfg)                   -> token tensor
    judge(model, question, response, cfg)    -> 1.0 / 0.0 / nan
    RunContext(metric, cfg)                   -> per-run dir + manifest + log
"""

import os
import re
import gc
import json
import datetime
import subprocess
from collections import deque

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from drift_metrics import METRICS  # noqa: F401  (re-exported for convenience)

_MODEL = None
_MODEL_ID = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(cfg, model_id=None):
    """Load a model and cache it (single slot).

    When a different model is requested, the previous one is released first so the
    multi-model loop never holds two 7B models in memory at once.
    """
    global _MODEL, _MODEL_ID
    model_id = model_id or cfg.MODEL_ID
    if _MODEL is not None and _MODEL_ID == model_id:
        return _MODEL
    if _MODEL is not None:
        print(f"[runtime] releasing {_MODEL_ID}...")
        del _MODEL
        _MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    device = pick_device()
    print(f"[runtime] loading {model_id} on {device} (bfloat16)...")
    _MODEL = HookedTransformer.from_pretrained(model_id, device=device, dtype=torch.bfloat16)
    _MODEL_ID = model_id
    return _MODEL


def _device_of(model):
    return model.cfg.device


# ---------------------------------------------------------------------------
# Prompting — render inputs through the model's OWN chat template
# ---------------------------------------------------------------------------
# The instruct models we study (Qwen / Llama / Gemma / Mistral) were tuned on a
# chat format (<|im_start|>, <|start_header_id|>, <bos><start_of_turn>, [INST]…).
# Feeding raw "System:/User:/Response:" text runs them OFF-DISTRIBUTION, which
# collapses accuracy and makes the model emit role/termination tokens mid-answer
# (the "formatting" confound). build_chat_prompt fixes that by using the
# tokenizer's apply_chat_template; base models without a template fall back to
# the legacy raw string so nothing breaks.
def build_chat_prompt(model, user_content, system_content=None):
    """Return a generation-ready prompt string in the model's native chat format."""
    tok = getattr(model, "tokenizer", None)
    if tok is not None and getattr(tok, "chat_template", None):
        msgs = []
        if system_content:
            msgs.append({"role": "system", "content": system_content})
        msgs.append({"role": "user", "content": user_content})
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass  # some templates reject a system role — fall through to raw
    sys_prefix = f"{system_content}\n" if system_content else ""
    return f"{sys_prefix}{user_content}\n"


def _to_input_tokens(model, prompt):
    """Tokenize a (chat-templated) prompt for generation.

    prepend_bos=False because the chat template already carries the model's
    leading special tokens; letting to_tokens add another BOS double-counts it
    for Llama/Gemma. Qwen's template has no BOS, so this is correct there too.
    Used everywhere a generation prompt is tokenized so prompt-length bookkeeping
    (random-trigger position, recovery rewind) stays consistent.
    """
    return model.to_tokens(prompt, prepend_bos=False)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# Per-dataset adapters: how to pull the question text and gold answer. gold/_to_float
# are defined further down; resolved at call time, so order here is fine.
DATASET_ADAPTERS = {
    "gsm8k": {
        "hf": ("gsm8k", "main", "test"),
        "question": lambda it: it["question"],
        "gold": lambda it: gsm8k_gold(it.get("answer", "")),
    },
    "svamp": {
        "hf": ("ChilleD/SVAMP", None, "test"),
        "question": lambda it: (str(it.get("Body", "")) + " " + str(it.get("Question", ""))).strip(),
        "gold": lambda it: _to_float(str(it.get("Answer", ""))),
    },
    "multiarith": {
        "hf": ("ChilleD/MultiArith", None, "test"),
        "question": lambda it: it.get("question", ""),
        "gold": lambda it: _to_float(str(it.get("final_ans", it.get("answer", "")))),
    },
}


def _load_hf(spec, n):
    name, conf, split = spec
    ds = load_dataset(name, conf, split=split) if conf else load_dataset(name, split=split)
    n = n or len(ds)
    return ds.select(range(min(n, len(ds))))


def _format_question(cfg, model, q):
    """Render a math question as a prompt — chat-templated when enabled."""
    if model is not None and getattr(cfg, "USE_CHAT_TEMPLATE", True):
        return build_chat_prompt(model, q, cfg.GSM8K_SYSTEM)
    return cfg.GSM8K_PROMPT.format(q=q)  # raw prompting (USE_CHAT_TEMPLATE=False)


def load_target_items(cfg, n, dataset="gsm8k", model=None):
    """Target problems as dicts {prompt, question, gold} for any registered dataset.

    Pass `model` so prompts render through its chat template (correct, in-distribution);
    omit it only for back-compat with the legacy raw-string format.
    """
    ad = DATASET_ADAPTERS[dataset]
    items = []
    for it in _load_hf(ad["hf"], n):
        q = ad["question"](it)
        gold = ad["gold"](it)
        if gold is None:
            continue  # unparseable gold would bias the accuracy denominators
        items.append({"prompt": _format_question(cfg, model, q), "question": q, "gold": gold})
    return items


def load_target_prompts(cfg, n, dataset="gsm8k", model=None):
    return [x["prompt"] for x in load_target_items(cfg, n, dataset, model)]


def _format_target(cfg, item):
    return cfg.GSM8K_PROMPT.format(q=item["question"])


def _format_control(cfg, spec, item):
    name = spec[0]
    if name == "cnn_dailymail":
        return cfg.CNN_PROMPT.format(q=item["article"][:500])
    if name == "mbpp":
        return cfg.MBPP_PROMPT.format(q=item["text"])
    # generic fallback: first string-valued field
    for v in item.values():
        if isinstance(v, str):
            return v
    raise ValueError(f"no prompt template for dataset {name}")


def load_problems(cfg, which="target", n=None, control_spec=None, dataset="gsm8k", model=None):
    """Return a list of formatted prompt strings.

    which="target"  -> the reasoning task `dataset` (registered adapter)
    which="control" -> one of cfg.CONTROL_DATASETS (pass control_spec)
    Pass `model` to render through its chat template (correct, in-distribution).
    """
    if which == "target":
        return load_target_prompts(cfg, n, dataset, model)

    spec = control_spec or cfg.CONTROL_DATASETS[0]
    name, conf, split = spec
    ds = load_dataset(name, conf, split=split) if conf else load_dataset(name, split=split)
    n = n or len(ds)
    ds = ds.select(range(min(n, len(ds))))
    prompts = [_format_control(cfg, spec, it) for it in ds]
    if model is not None and getattr(cfg, "USE_CHAT_TEMPLATE", True):
        prompts = [build_chat_prompt(model, p) for p in prompts]  # FPR in-distribution
    return prompts


# ---------------------------------------------------------------------------
# Answer parsing & trigger-cause classification (for the labeling stage)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _to_float(s):
    try:
        return float(s.replace(",", "").replace("$", ""))
    except (ValueError, AttributeError):
        return None


def gsm8k_gold(answer_field):
    """Parse the gold answer from a GSM8K 'answer' field (the '#### N' tail)."""
    if not answer_field:
        return None
    if "####" in answer_field:
        return _to_float(answer_field.split("####")[-1].strip())
    nums = _NUM_RE.findall(answer_field)
    return _to_float(nums[-1]) if nums else None


_ANSWER_RE = re.compile(
    r"(?:answer\s*(?:is|:)|final answer|####|\\boxed\{|=)\s*\$?(-?\d[\d,]*\.?\d*)", re.I)


def extract_final_number(text):
    """Final numeric answer from a generated solution.

    Prefers an explicit answer marker ("answer is", "= ", "####", "\\boxed{}")
    so we don't grab an intermediate computation; falls back to the last number.
    """
    if not text:
        return None
    marked = _ANSWER_RE.findall(text)
    if marked:
        return _to_float(marked[-1])
    nums = _NUM_RE.findall(text)
    return _to_float(nums[-1]) if nums else None


def answers_match(pred, gold, tol=1e-4):
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tol


def is_formatting_trigger(trigger_token, context_text="", cfg=None):
    """True if the trigger looks like chat-template / termination collapse, not content.

    Checks the triggering token and a small surrounding context against the
    FORMATTING_MARKERS list — this is what separates the formatting confound from
    a genuine reasoning-step token.
    """
    markers = cfg.FORMATTING_MARKERS if cfg else []
    blob = f"{trigger_token or ''} {context_text or ''}"
    return any(mk in blob for mk in markers)


# ---------------------------------------------------------------------------
# Sampling helpers (shared so detection and the random control sample identically)
# ---------------------------------------------------------------------------
def _sample_next(logits, cfg):
    scaled = logits[0, -1] / cfg.TEMPERATURE
    top_k_logits, top_k_idx = torch.topk(scaled, cfg.TOP_K)
    probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    return top_k_idx[choice].item()


def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Detection engine — single layer (used by Stage 2 ablation)
# ---------------------------------------------------------------------------
def detect(model, prompt, metric, layer, z_threshold, seed, cfg,
           full_trajectory=False, collect_trace=False):
    """Run the detection phase for one problem at one layer with one metric.

    Returns a dict:
        triggered, spike_idx, trigger_z, trigger_token, max_z, baseline_mean,
        tokens, tokens_at_trigger, break_idx, gen_token_strs, z_trace
    `tokens` is the token state at the trigger (full_trajectory=False, seeds
    recovery) or at the end (full_trajectory=True). `z_trace` (collect_trace=True)
    is [{idx, z}] per scored step — the data the Fig-1 spike overlay needs.
    `break_idx` is the first generated position that emits a formatting/template
    token (where the text visibly breaks), for the distance-to-hallucination figure.
    """
    _seed_all(seed)
    device = _device_of(model)
    tokens = _to_input_tokens(model, prompt)
    hook = f"blocks.{layer}.hook_resid_post"

    healthy_norm, healthy_raw = [], []
    ref = None
    recent = deque(maxlen=cfg.RECENT_WINDOW)
    initial = []
    baseline_mean = baseline_std = None
    spike_idx, trigger_z, max_z = -1, None, 0.0
    trigger_token, tokens_at_trigger = None, None
    z_trace = []
    gen_token_strs = []
    break_idx = None

    for i in range(cfg.MAX_NEW_TOKENS_DETECTION):
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x == hook)
        next_token = _sample_next(logits, cfg)
        tok_str = model.to_string(next_token)
        gen_token_strs.append(tok_str)
        if break_idx is None and is_formatting_trigger(tok_str, "", cfg):
            break_idx = i

        h_raw = cache[hook][:, -1, :].float()
        h_norm = h_raw / (h_raw.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        if i < cfg.HEALTHY_WINDOW:
            healthy_norm.append(h_norm)
            healthy_raw.append(h_raw)
            if i == cfg.HEALTHY_WINDOW - 1:
                ref = metric.build_reference(healthy_norm, healthy_raw)
        else:
            recent.append(h_norm)
            ready = (not metric.uses_set) or (len(recent) == cfg.RECENT_WINDOW)
            if ready:
                dist = metric.score(h_norm, h_raw, list(recent), ref)
                if len(initial) < cfg.TARE_WINDOW_SIZE:
                    initial.append(dist)
                    if len(initial) == cfg.TARE_WINDOW_SIZE:
                        baseline_mean = float(np.mean(initial))
                        baseline_std = max(float(np.std(initial)), 1e-5)
                if baseline_mean is not None:
                    z = (dist - baseline_mean) / baseline_std
                    if collect_trace:
                        z_trace.append({"idx": i, "z": float(z)})
                    if z > max_z:
                        max_z = z
                    if z > z_threshold and spike_idx == -1:
                        # record the trigger: the token being emitted at the spike,
                        # and the token state BEFORE it (what recovery rewinds from).
                        spike_idx, trigger_z = i, z
                        trigger_token = tok_str
                        tokens_at_trigger = tokens.clone()
                        if not full_trajectory:
                            break

        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == model.tokenizer.eos_token_id:
            break

    return {
        "triggered": spike_idx != -1,
        "spike_idx": spike_idx,
        "trigger_z": trigger_z,
        "trigger_token": trigger_token,
        "max_z": max_z,
        "baseline_mean": baseline_mean,
        "tokens": tokens,
        "tokens_at_trigger": tokens_at_trigger if tokens_at_trigger is not None else tokens,
        "break_idx": break_idx,
        "gen_token_strs": gen_token_strs,
        "z_trace": z_trace,
    }


# ---------------------------------------------------------------------------
# Detection engine — all candidate layers in ONE generation (Stage 0/1)
# ---------------------------------------------------------------------------
def detect_multilayer(model, prompt, metric, layers, seed, cfg, ref_z=None, collect_traces=False):
    """Score every candidate layer from a single generation pass.

    Returns a dict:
        {"layers": {L: {"max_z", "spike_idx", "trace"(if collect_traces)}},
         "break_idx": first generated position emitting a formatting token}
    `spike_idx` is the first step layer L crosses ref_z (drives the
    distance-to-hallucination figure: D2H = break_idx - spike_idx). Far cheaper
    than re-generating per layer. Not used for the random control (no geometry).
    """
    _seed_all(seed)
    ref_z = ref_z if ref_z is not None else cfg.LAYER_SELECT_REF_Z
    device = _device_of(model)
    tokens = _to_input_tokens(model, prompt)
    hooks = {L: f"blocks.{L}.hook_resid_post" for L in layers}
    hook_set = set(hooks.values())
    break_idx = None

    state = {
        L: {"healthy_norm": [], "healthy_raw": [], "ref": None,
            "recent": deque(maxlen=cfg.RECENT_WINDOW), "initial": [],
            "mean": None, "std": None, "max_z": 0.0, "spike_idx": -1, "trace": []}
        for L in layers
    }

    for i in range(cfg.MAX_NEW_TOKENS_DETECTION):
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x in hook_set)
        next_token = _sample_next(logits, cfg)
        if break_idx is None and is_formatting_trigger(model.to_string(next_token), "", cfg):
            break_idx = i

        for L in layers:
            s = state[L]
            h_raw = cache[hooks[L]][:, -1, :].float()
            h_norm = h_raw / (h_raw.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            if i < cfg.HEALTHY_WINDOW:
                s["healthy_norm"].append(h_norm)
                s["healthy_raw"].append(h_raw)
                if i == cfg.HEALTHY_WINDOW - 1:
                    s["ref"] = metric.build_reference(s["healthy_norm"], s["healthy_raw"])
            else:
                s["recent"].append(h_norm)
                ready = (not metric.uses_set) or (len(s["recent"]) == cfg.RECENT_WINDOW)
                if ready:
                    dist = metric.score(h_norm, h_raw, list(s["recent"]), s["ref"])
                    if len(s["initial"]) < cfg.TARE_WINDOW_SIZE:
                        s["initial"].append(dist)
                        if len(s["initial"]) == cfg.TARE_WINDOW_SIZE:
                            s["mean"] = float(np.mean(s["initial"]))
                            s["std"] = max(float(np.std(s["initial"])), 1e-5)
                    if s["mean"] is not None:
                        z = (dist - s["mean"]) / s["std"]
                        if collect_traces:
                            s["trace"].append({"idx": i, "z": float(z)})
                        if z > s["max_z"]:
                            s["max_z"] = z
                        if z > ref_z and s["spike_idx"] == -1:
                            s["spike_idx"] = i

        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == model.tokenizer.eos_token_id:
            break

    layer_out = {}
    for L in layers:
        s = state[L]
        rec = {"max_z": s["max_z"], "spike_idx": s["spike_idx"]}
        if collect_traces:
            rec["trace"] = s["trace"]
        layer_out[L] = rec
    return {"layers": layer_out, "break_idx": break_idx}


# ---------------------------------------------------------------------------
# Trajectory sampling (for the random-position control)
# ---------------------------------------------------------------------------
def sample_trajectory(model, prompt, seed, cfg):
    """Sample a full detection-length trajectory (no detection). Returns tokens."""
    _seed_all(seed)
    device = _device_of(model)
    tokens = _to_input_tokens(model, prompt)
    for _ in range(cfg.MAX_NEW_TOKENS_DETECTION):
        with torch.no_grad():
            logits = model(tokens)
        next_token = _sample_next(logits, cfg)
        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == model.tokenizer.eos_token_id:
            break
    return tokens


def random_trigger_index(prompt_len, total_len, cfg, rng):
    """Uniform random trigger position within the post-healthy generated span."""
    lo = prompt_len + cfg.HEALTHY_WINDOW + cfg.TARE_WINDOW_SIZE
    hi = total_len - 1
    if hi <= lo:
        return None
    return int(rng.integers(lo, hi))


# ---------------------------------------------------------------------------
# Recovery — the four ablation paths (identical for every metric)
# ---------------------------------------------------------------------------
def recovery_paths(model, current_tokens, cfg):
    """Generate the four ablation paths from the token state at the trigger.

    A: baseline (do nothing)   B: rewind + reprompt (full cure)
    C: reprompt only           D: rewind only
    All greedy, so the only variable is the intervention.
    """
    gen = lambda toks: model.generate(
        toks, max_new_tokens=cfg.MAX_NEW_TOKENS_RECOVERY,
        do_sample=False, prepend_bos=False, return_type="tokens",
    )
    dec = model.tokenizer.decode

    out_a = gen(current_tokens)

    safe = max(current_tokens.shape[1] - cfg.REWIND_TOKENS, 1)
    clean = current_tokens[:, :safe]
    reset = model.to_tokens(cfg.COGNITIVE_RESET_TEXT, prepend_bos=False)

    out_b = gen(torch.cat([clean, reset], dim=1))
    out_c = gen(torch.cat([current_tokens, reset], dim=1))
    out_d = gen(clean)

    return {
        "path_a_baseline": dec(out_a[0]),
        "path_b_full_cure": dec(out_b[0]),
        "path_c_prompt_only": dec(out_c[0]),
        "path_d_rewind_only": dec(out_d[0]),
    }


# ---------------------------------------------------------------------------
# Judge — the model grades a response for semantic soundness
# ---------------------------------------------------------------------------
def judge(model, question, response, cfg, persona="default"):
    """Return 1.0 (sound) / 0.0 (drifted) / nan (unparseable).

    persona lets Stage 4 run a second, differently-worded judge (TODO #14).
    """
    if persona == "strict":
        sys_line = ("You are a strict grader. Output 1.0 only if the final answer is "
                    "fully correct and the arithmetic is sound; otherwise 0.0.")
    else:
        sys_line = ("You are an expert grading assistant. Did the reasoning stay sound, "
                    "or did it drift into hallucination / unnecessary advanced math? "
                    "Score 1.0 if sound, 0.0 if severe drift.")
    judge_prompt = build_chat_prompt(
        model,
        f"Question: {question}\nResponse: {response}",
        f"{sys_line}\nOutput ONLY the float 1.0 or 0.0. No other words.",
    )
    toks = _to_input_tokens(model, judge_prompt)
    out = model.generate(
        toks, max_new_tokens=cfg.JUDGE_MAX_NEW_TOKENS,
        do_sample=False, prepend_bos=False, return_type="tokens",
    )
    content = model.tokenizer.decode(out[0, toks.shape[1]:]).strip()
    if "1.0" in content or content.startswith("1"):
        return 1.0
    if "0.0" in content or content.startswith("0"):
        return 0.0
    return float("nan")


# ---------------------------------------------------------------------------
# Run history — manifest, per-stage status, logging
# ---------------------------------------------------------------------------
def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _config_snapshot(cfg):
    return {k: getattr(cfg, k) for k in dir(cfg)
            if k.isupper() and not k.startswith("_")}


def now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class RunContext:
    """Owns one run directory: runs/<timestamp>-<metric>/ with manifest + log.

    The manifest records git SHA, the full config snapshot, and per-stage status /
    timings / artifact paths — so the entire history of a run is recoverable from
    the directory alone. Supports resume: a stage already marked "ok" is skipped.
    """

    def __init__(self, metric, cfg, run_dir=None, model_id=None, dataset="gsm8k"):
        self.cfg = cfg
        self.metric = metric
        self.dataset = dataset
        self.model_id = model_id or cfg.MODEL_ID
        short_model = self.model_id.split("/")[-1]
        if run_dir is None:
            run_dir = os.path.join(cfg.RUNS_DIR,
                                   f"{now_stamp()}-{short_model}-{dataset}-{metric}")
        self.dir = run_dir
        os.makedirs(self.dir, exist_ok=True)
        self.log_path = os.path.join(self.dir, "pipeline.log")
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "model": self.model_id,
                "dataset": dataset,
                "metric": metric,
                "started": now_stamp(),
                "git_sha": _git_sha(),
                "config": _config_snapshot(cfg),
                "selected_layer": None,
                "selected_z": None,
                "stages": {},
            }
            self._save()

    def artifact(self, name):
        return os.path.join(self.dir, name)

    def _save(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def log(self, msg):
        line = f"[{now_stamp()}] {msg}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def stage_done(self, name):
        return self.manifest["stages"].get(name, {}).get("status") == "ok"

    def start_stage(self, name):
        self.manifest["stages"][name] = {"status": "running", "started": now_stamp()}
        self._save()
        self.log(f"=== stage {name} START ===")

    def finish_stage(self, name, status="ok", artifacts=None, **extra):
        rec = self.manifest["stages"].setdefault(name, {})
        rec.update({"status": status, "finished": now_stamp(),
                    "artifacts": artifacts or [], **extra})
        self._save()
        self.log(f"=== stage {name} {status.upper()} ===")

    def set_selection(self, layer=None, z=None):
        if layer is not None:
            self.manifest["selected_layer"] = layer
        if z is not None:
            self.manifest["selected_z"] = z
        self._save()
