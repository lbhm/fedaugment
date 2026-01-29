# Projection Model Experiments

This directory contains scripts for training and evaluating embedding projection models that align multiple embedding views into a common vector space for table augmentation tasks.

## Overview

The projection experiments follow this pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

data/embeddings/webtable/split/train/           train_projection_models.py
    ├── mpnet-dj_adpt.fa/          ─────────►   (Train projection models)
    ├── distilroberta-dj_adpt.fa/                        │
    ├── gte_base-dj_adpt.fa/                             │
    └── ... (N views)                                    ▼
                                                data/checkpoints/{group}/
                                                    └── {model_name}/
                                                        └── last.ckpt

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PHASE                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────┬────────────────────────────────────┐
    │                                    │                                    │
    ▼                                    ▼                                    ▼
evaluate_aligned.py              evaluate_centralized.py       evaluate_view_robustness.py
(Multi-view aligned)             (Single-view baseline)        (Per-view analysis)
    │                                    │                                    │
    ▼                                    ▼                                    ▼
logs/augmentations/              logs/augmentations/           logs/view-robustness/
  {dataset}/{group}/               {dataset}/centralized/        {dataset}/{model}/
    └── *.csv                        └── *.csv                     └── metrics.parquet

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ANALYSIS PHASE                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

    logs/augmentations/          logs/view-robustness/
           │                            │
           ▼                            ▼
analysis/augmentations.ipynb    analysis/view_robustness.ipynb
  (Aggregate results,             (Box plots for per-view
   generate tables/plots)          precision and recall)
```

## Files

| File | Purpose | Paper Section |
|------|---------|---------------|
| `train_projection_models.py` | Train projection models (CL, LA2M, Vec2Vec, etc.) | §5.3 |
| `evaluate_aligned.py` | Evaluate join/union discovery with aligned embeddings | §6.1 |
| `evaluate_centralized.py` | Single-view baseline (no projection) | §6.1 |
| `evaluate_view_robustness.py` | Per-view precision/recall breakdown | §6.2 |
| `train_utils.py` | Shared training utilities | - |
| `config.py` (parent dir) | Hyperparameter configurations | - |

## Usage

### Training

```bash
# Train CL and LA2M models with 12 views
uv run python -m experiments.projections.train_projection_models \
    --experiment-group 12_views \
    --dataset webtable \
    --models cl_optim la2m_default la2m_nopca v2v \
    --views 12 \
    --seed 123

# Train with different view counts for scaling analysis
for v in 2 4 6 8 10 12; do
    uv run python -m experiments.projections.train_projection_models \
        --experiment-group all_views \
        --dataset webtable \
        --models cl_optim \
        --views $v \
        --seed 123
done
```

### Evaluation

```bash
# Evaluate aligned embeddings on test datasets
uv run python -m experiments.projections.evaluate_aligned \
    --experiment-group 12_views \
    --datasets freyja omnimatch_city_test omnimatch_culture_test santos-union

# Evaluate single-view baselines
uv run python -m experiments.projections.evaluate_centralized \
    --datasets freyja omnimatch_city_test omnimatch_culture_test santos-union

# Per-view robustness analysis
uv run python -m experiments.projections.evaluate_view_robustness \
    --experiment-group 12_views \
    --datasets freyja omnimatch_city_test \
    --ignore union_  # Skip union baselines
```

## Model Configurations

Models are defined in `experiments/config.py`:

| Model | Type | Description |
|-------|------|-------------|
| `cl_optim` | Neural Network | Contrastive learning with NT-Xent loss, early stopping |
| `cl_default` | Neural Network | Deeper CL architecture (exploratory) |
| `cl_noval` | Neural Network | CL without validation (for efficiency benchmarks) |
| `la2m_default` | Clustering | LA2M with PCA reduction to 128 dimensions |
| `la2m_nopca` | Clustering | LA2M without PCA (full dimensionality) |
| `v2v` | GAN | Vec2Vec adversarial alignment |
| `union_plus` | Baseline | Zero-padding to max dimension |
| `union_minus` | Baseline | Truncation to min dimension |

## Output Format

### Augmentation Results (`logs/augmentations/`)

CSV files with columns:
- `query_table`, `query_column`: Query identifier
- `precision@{k}`, `recall@{k}`: Metrics at various k values
- `map`: Mean Average Precision
- `mrr`: Mean Reciprocal Rank

### View Robustness (`logs/view-robustness/`)

Parquet files with per-view metrics for analyzing which embedding models perform best.

## Checkpoints

Trained models are saved to `data/checkpoints/{experiment_group}/{model_name}/`:

```
data/checkpoints/12_views/
├── webtable-cl_optim-v=12-{view_spec}/
│   ├── last.ckpt              # Final model checkpoint
│   └── version-1-*.ckpt       # Best checkpoint (by val_loss)
├── webtable-la2m_default-v=12-{view_spec}/
│   └── last.ckpt
└── ...
```

## Troubleshooting

### "Embeddings not found"

Ensure embeddings exist for all views:

```bash
ls data/embeddings/webtable/split/train/
# Should show: mpnet-dj_adpt.fa/, distilroberta-dj_adpt.fa/, etc.
```

### Out of Memory during training

1. Reduce batch size: Edit `batch_size` in model config
2. Use fewer views: `--views 4` instead of `--views 12`
3. Enable mixed precision: Set `precision: "bf16-mixed"` in trainer config

### Slow evaluation

The HNSW index construction can be slow for large datasets. Consider:
1. Increasing `ef_construction` for better quality (slower build)
2. Decreasing `M` for faster build (lower quality)
