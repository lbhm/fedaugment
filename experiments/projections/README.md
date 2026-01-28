# Embedding Projections Experiments

Benchmark framework for evaluating embedding alignment strategies on join and union discovery tasks.

## Data Flow

```text
data/embeddings/{dataset}/data/*.fa  →  train_projection_models.py
(Pre-computed embeddings)               (Train projection models)
                                                    ↓
                                        data/checkpoints/{group}/
                                                    ↓
                    ┌───────────────────────────────┴───────────────────────────────┐
                    ↓                                                               ↓
           evaluate_aligned.py                                           evaluate_centralized.py
      (Aligned table augmentation)                                       (Single-view baseline)
                    ↓                                                               ↓
  logs/augmentations/{dataset}/{model}/                         logs/augmentations/{dataset}/centralized/
                    │                                                               │
                    └───────────────────────────────┬───────────────────────────────┘
                                                    ↓
                                   analysis/analyze_augmentations.ipynb
                                     (Aggregate & visualize results)


                                       evaluate_view_robustness.py
                                       (Per-view precision/recall)
                                                    ↓
                                 logs/view-robustness/{dataset}/{model}/
                                       (Per-view metrics parquet)
                                                    ↓
                                      analysis/view_robustness.ipynb
                                        (Box plots visualization)
```

## Files

| File                                | Purpose                                           |
|-------------------------------------|---------------------------------------------------|
| `config.py`                         | CL hyperparameter presets and other configs       |
| `evaluate_aligned.py`               | Join/union discovery evaluation                   |
| `evaluate_centralized.py`           | Single-view baseline evaluation                   |
| `evaluate_view_robustness.py`       | Per-view precision/recall analysis                |
| `train_projection_models.py`        | Training script for projection models             |

## Usage

Run the respective Python scripts via `uv`:

```bash
# Train models
uv run python -m experiments.projections.train_projection_models \
    --experiment-group my_experiment --dataset freyja \
    --views 2 4 6 8 10 --models cl_default la2m_default

# Evaluate aligned embeddings
uv run python -m experiments.projections.evaluate_aligned \
    --experiment-group my_experiment

# View robustness analysis
uv run python -m experiments.projections.evaluate_view_robustness \
    --experiment-group my_experiment --datasets freyja
```
