"""Five-way failure typology of a triggered baseline failure (torch-free).

Kept separate from stage_labeling so it can be re-run over saved artifacts on a
machine without the model stack (see retype_failures.py).

The path texts in ablation.jsonl are full decoded sequences (prompt + continuation),
so the classifier (a) strips the prompt before scanning for collapse markers -- the
ChatML prompt itself contains "<|im_start|>", which is in FORMATTING_MARKERS -- and
(b) removes the model's legitimate end-of-turn token so a normal EOS is never
counted as a template collapse. The pre-2026-09-01 classifier did neither and
labeled every ChatML-model failure "format_template_collapse".
"""
import re

FAILURE_TYPES = ["arithmetic_logical", "semantic_drift",
                 "format_template_collapse", "premature_termination", "other_unclear"]

_GEN_START_SEPARATORS = ["<|im_start|>assistant\n", "[/INST]", "<|assistant|>\n",
                         "<start_of_turn>model\n", "Response:"]
_EOS_TOKENS = ["<|im_end|>", "</s>", "<|endoftext|>", "<|eot_id|>", "<|end|>",
               "<end_of_turn>"]
_ANSWER_PHRASE = re.compile(r"answer\s*(?:is|:)", re.I)


def generated_text(text):
    """Isolate the model's generated continuation from a full decoded path text."""
    text = text or ""
    for sep in _GEN_START_SEPARATORS:
        i = text.rfind(sep)
        if i >= 0:
            return text[i + len(sep):].strip()
    return text.strip()


def strip_eos(gen):
    for t in _EOS_TOKENS:
        gen = gen.replace(t, "")
    return gen.strip()


def _has_marker(trigger_token, gen, markers):
    blob = f"{trigger_token or ''} {gen or ''}"
    return any(mk in blob for mk in markers)


def _is_repetitive(gen, thresh=0.5):
    words = gen.split()
    if len(words) < 12:
        return False
    return (len(set(words)) / len(words)) < thresh   # low unique-word ratio = looping


def classify_failure(rec, cfg):
    """Bucket one baseline failure into one of 5 failure categories."""
    gen = strip_eos(generated_text(rec.get("path_a_baseline")))
    if _has_marker(rec.get("trigger_token"), gen, cfg.FORMATTING_MARKERS):
        return "format_template_collapse"   # role/template tag hallucinated mid-answer
    if not _ANSWER_PHRASE.search(gen):
        # never stated an answer: looping, or cut off by the token budget
        return "semantic_drift" if _is_repetitive(gen) else "premature_termination"
    if rec.get("baseline_answer") is not None:
        return "arithmetic_logical"      # well-formed, stated a (wrong) answer
    return "other_unclear"
