# auto_pipeline

Evaluation pipeline for a residual-stream drift monitor on short math chain-of-thought. One command runs the full stage chain once per (model, dataset, metric) cell and writes a self-contained run directory under `runs/`.

See `METHODOLOGY.md` for what each stage computes and `../README.md` for the repository overview, the run command and the run-directory layout.

## Files

| file | role |
|---|---|
| `config.py` | Every knob: models, metrics, layer band, Z grid, sample counts, seeds, windows, datasets, `USE_CHAT_TEMPLATE`. It is snapshotted into each run's `manifest.json`. |
| `drift_metrics.py` | Drift score geometries (`l2`, `sinkhorn`, `cosine`, `residual_norm`) behind one `build_reference` / `score` interface. |
| `runtime.py` | Shared engine: model loading, chat-template prompting, detection, the four recovery paths, answer extraction, dataset loading, judge, `RunContext`. Imports torch. |
| `stage0_layer_select.py` | Sweeps the layer band and selects a layer by `z_gap`. |
| `stage1_threshold_sweep.py` | Sweeps `Z_GRID` on a held-out fit/validation split and selects the operating Z. |
| `stage2_ablation.py` | Detection plus the four recovery paths; writes `ablation.jsonl` and `trajectories.jsonl`. |
| `stage_labeling.py` | AUC of max_z against wrong answers, formatting-versus-genuine split, false-trigger rate, flip rates, failure typology. No new model runs. |
| `stage_threshold_downstream.py` | Re-runs detection and recovery at each Z and reports final accuracy per Z. |
| `stage3_robustness.py` | Repeats detection across seeds and reports the trigger-rate mean and standard deviation. |
| `stage4_judge.py` | Same-model soundness judge (not used for any accuracy number). |
| `stage_results.py` | Per-method final-accuracy table for the run (`results_methods.csv`) and the cross-run `runs/results_table.csv`. |
| `stage5_aggregate.py` | Cross-run rollup `runs/master_results.csv` and plots. |
| `analysis.py`, `failure_typology.py`, `retype_failures.py` | Shared accuracy arithmetic and the failure-type classifier. |
| `run_all.py` | Driver: `--model`, `--dataset`, `--metric`, `--stages`, `--profile`, `--resume`, `--gpus`. |
| `figures/make_paper_figures.py` | Rebuilds the paper figures and tables from `runs/` with no model and no GPU. |

The string `sdr_<metric>` appears as a method key in `results_methods.csv`, `results_table.csv` and `manifest.json`. It is a data value that the table and figure code read, so it is kept unchanged.
