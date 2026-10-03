"""Ablation — L2 (point-to-point) version.

The drift signal is an explicit
squared-L2 distance between the current normalized residual state and the
healthy anchor mean. This is what a geomloss `sinkhorn_loss` call
degenerates to (single vector vs single vector), computed directly without the
optimal-transport solver.

See pipelines/02_ablation_sinkhorn.py for the set-vs-set OT counterpart.
All metric logic is in pipelines/drift_metrics.py.

Output: results/ablation_l2.jsonl  (records tag metric="l2")
"""

import torch, json, numpy as np, os
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm

from drift_metrics import l2_distance

SAVE_PATH = "./results/ablation_l2.jsonl"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

LAYER = 22
Z_THRESHOLD = 12.5
HEALTHY_WINDOW = 10
TARE_WINDOW_SIZE = 5
MAX_NEW_TOKENS_DETECTION = 250

REWIND_TOKENS = 15
COGNITIVE_RESET_TEXT = "\nWait, let me recalculate this step to be sure:\n"

torch.manual_seed(42)
np.random.seed(42)

print(f"Loading {MODEL_ID} for L2 Ablation Study...")
model = HookedTransformer.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)


def run_ablation(question):
    formatted_prompt = f"System: Solve step-by-step using basic arithmetic.\nUser: {question}\nResponse:"
    current_tokens = model.to_tokens(formatted_prompt)

    stats_log = []
    spike_idx, baseline_mean, baseline_std, trigger_z = -1, None, None, None

    anchor_states = []
    v_anchor = None
    initial_distances = []

    for i in range(MAX_NEW_TOKENS_DETECTION):
        with torch.no_grad():
            logits, cache = model.run_with_cache(current_tokens, names_filter=lambda x: f"blocks.{LAYER}.hook_resid_post" in x)

        temperature = 0.85
        top_k = 50
        scaled_logits = logits[0, -1] / temperature
        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
        sampling_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
        next_token_idx = torch.multinomial(sampling_probs, num_samples=1)
        next_token = top_k_indices[next_token_idx].item()

        token_str = model.to_string(next_token)

        h_t = cache[f"blocks.{LAYER}.hook_resid_post"][:, -1, :].float()
        h_t_norm = h_t / (h_t.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        if i < HEALTHY_WINDOW:
            anchor_states.append(h_t_norm)
            if i == HEALTHY_WINDOW - 1:
                v_anchor = torch.stack(anchor_states).mean(dim=0)
                v_anchor = v_anchor / (v_anchor.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        else:
            dist = l2_distance(h_t_norm, v_anchor)
            stats_log.append({"token": token_str, "dist": dist, "idx": i})

            if len(initial_distances) < TARE_WINDOW_SIZE:
                initial_distances.append(dist)
                if len(initial_distances) == TARE_WINDOW_SIZE:
                    baseline_mean = np.mean(initial_distances)
                    baseline_std = max(np.std(initial_distances), 1e-5)

            if baseline_mean is not None:
                z_score = (dist - baseline_mean) / baseline_std
                if z_score > Z_THRESHOLD and spike_idx == -1:
                    spike_idx = i
                    trigger_z = z_score
                    print(f"\n[L2] DYNAMIC SPIKE at {i}: Z-Score = {z_score:.2f} above baseline")
                    break

        current_tokens = torch.cat([current_tokens, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
        if next_token == model.tokenizer.eos_token_id: break

    if spike_idx == -1: return None

    print(f"Triggered! Generating Ablation Paths...")

    out_a = model.generate(current_tokens, max_new_tokens=150, do_sample=False, prepend_bos=False, return_type="tokens")

    safe_slice_idx = max(current_tokens.shape[1] - REWIND_TOKENS, 1)
    clean_tokens = current_tokens[:, :safe_slice_idx]
    reset_tokens = model.to_tokens(COGNITIVE_RESET_TEXT, prepend_bos=False)
    path_b_tokens = torch.cat([clean_tokens, reset_tokens], dim=1)
    out_b = model.generate(path_b_tokens, max_new_tokens=150, do_sample=False, prepend_bos=False, return_type="tokens")

    path_c_tokens = torch.cat([current_tokens, reset_tokens], dim=1)
    out_c = model.generate(path_c_tokens, max_new_tokens=150, do_sample=False, prepend_bos=False, return_type="tokens")

    out_d = model.generate(clean_tokens, max_new_tokens=150, do_sample=False, prepend_bos=False, return_type="tokens")

    return {
        "metric": "l2",
        "question": question,
        "spike_idx": spike_idx,
        "trigger_z": trigger_z,
        "baseline_mean": baseline_mean,
        "path_a_baseline": model.tokenizer.decode(out_a[0]),
        "path_b_full_cure": model.tokenizer.decode(out_b[0]),
        "path_c_prompt_only": model.tokenizer.decode(out_c[0]),
        "path_d_rewind_only": model.tokenizer.decode(out_d[0])
    }


dataset = load_dataset("gsm8k", "main", split="test").select(range(100))
print(f"Running L2 Ablation Study... Saving to: {SAVE_PATH}")

for prob in tqdm(dataset):
    try:
        data = run_ablation(prob['question'])
        if data:
            with open(SAVE_PATH, "a") as f:
                f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Error: {e}")
