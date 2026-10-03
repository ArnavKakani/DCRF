# pipelines

Two standalone ablation scripts and the shared metric module. The automated stages in `../auto_pipeline/` were built from them. They are kept for reference and are not needed to regenerate any paper figure.

| script | drift signal |
|---|---|
| `01_ablation_l2.py` | Point-to-point squared L2 between the current normalized residual state and the healthy-window mean. |
| `02_ablation_sinkhorn.py` | Set-to-set entropic optimal transport between the healthy-window states and a sliding window of recent states. |
| `drift_metrics.py` | The two distance functions used by both scripts. |

Each script writes a JSONL file to a local `./results/` directory, created on first run: `ablation_l2.jsonl` and `ablation_sinkhorn.jsonl`. Run them from the repository root with a GPU and the environment in `../requirements.txt`:

```
python pipelines/01_ablation_l2.py
python pipelines/02_ablation_sinkhorn.py
```

The Sinkhorn distance is on a different numeric scale from the L2 distance, so the `Z_THRESHOLD` constant in the Sinkhorn script has to be recalibrated for a new model.
