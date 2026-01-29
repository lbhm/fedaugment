# Experiments

This directory contains all scripts and configurations for reproducing the experiments in the FedAugment paper.

## Directory Structure

```
experiments/
├── config.py                 # Centralized configuration (models, datasets, hyperparameters)
├── utils.py                  # Shared utility functions
├── run_all.sh                # Master script to run the complete pipeline
│
├── embeddings/               # Embedding generation
│   ├── embed_dataset.py      # Generate embeddings for a single dataset
│   └── embed_all_datasets.sh # Batch embedding generation for all datasets
│
├── curation/                 # Data curation for training set selection
│   ├── curate_webtable.py    # Apply curation algorithms (FFT, grid, random)
│   ├── embed_webtable.py     # Generate embeddings for curated subsets
│   └── train_webtable.py     # Train models on curated data
│
├── projections/              # Projection model training and evaluation
│   ├── train_projection_models.py    # Train alignment models
│   ├── evaluate_aligned.py           # Evaluate aligned embeddings
│   ├── evaluate_centralized.py       # Single-view baseline evaluation
│   └── evaluate_view_robustness.py   # Per-view precision/recall analysis
│
├── efficiency/               # Performance measurements
│   └── measure_projections.py        # Training time and memory benchmarks
│
└── misc/                     # Miscellaneous utilities
    ├── curation_figure_data.py       # Data for curation visualizations
    └── pytorch_check.py              # Verify PyTorch installation
```

## Quick Start

### Full Pipeline

To run all experiments from scratch:

```bash
# Ensure datasets are in place
ls data/datasets/

# Run everything
bash experiments/run_all.sh
```

**Note:** The full pipeline requires significant resources (500+ GB RAM, A100 GPU) and takes several days to complete.

### Individual Steps

#### 1. Generate Embeddings

```bash
# Single dataset
uv run python -m experiments.embeddings.embed_dataset \
    --dataset freyja \
    --models mpnet distilroberta \
    --strategies dj_adpt

# All datasets
bash experiments/embeddings/embed_all_datasets.sh
```

#### 2. Train Projection Models

```bash
uv run python -m experiments.projections.train_projection_models \
    --experiment-group my_exp \
    --dataset webtable \
    --models cl_optim la2m_default \
    --views 8 12 \
    --seed 123
```

#### 3. Evaluate

```bash
# Aligned embeddings
uv run python -m experiments.projections.evaluate_aligned \
    --experiment-group my_exp \
    --datasets freyja omnimatch_city_test

# Centralized baseline
uv run python -m experiments.projections.evaluate_centralized \
    --datasets freyja omnimatch_city_test
```

## Configuration

All experiment configurations are centralized in `config.py`:

### Embedding Models

```python
EMBEDDING_MODEL_REGISTRY = {
    "mpnet": EmbeddingModelSpec(...),      # 768-dim, fast
    "distilroberta": EmbeddingModelSpec(...),  # 768-dim
    "qwen3_8b": EmbeddingModelSpec(...),   # 4096-dim, high quality
    # ... see config.py for full list
}
```

### Projection Models

```python
PROJECTION_MODEL_REGISTRY = {
    "cl_optim": ProjectionModelSpec(...),   # Best performer
    "la2m_default": ProjectionModelSpec(...),
    "v2v": ProjectionModelSpec(...),
    # ... see config.py for full list
}
```

### Datasets

```python
DATASET_REGISTRY = {
    "freyja": DatasetConfig(augmentation_mode="join", ...),
    "santos-union": DatasetConfig(augmentation_mode="union", ...),
    # ... see config.py for full list
}
```

## Output Structure

```
logs/
├── augmentations/
│   └── {dataset}/
│       ├── {experiment_group}/     # Aligned model results
│       │   └── {model}-v={views}-{views_spec}.csv
│       └── centralized/            # Single-view baselines
│           └── {model}-{strategy}.csv
│
├── view-robustness/
│   └── {dataset}/
│       └── {model}/
│           └── metrics.parquet     # Per-view precision/recall
│
├── efficiency/
│   └── {experiment_group}/
│       └── timing_results.json
│
└── projection-training/
    └── {experiment_group}/
        └── {model}/
            ├── config.json
            └── results.json
```

## Experiment Groups

Experiments are organized into groups for easier management:

| Group | Description | Views |
|-------|-------------|-------|
| `12_views` | Main experiments with 12 embedding views | 12 |
| `08_views` | Experiments with 8 views (for webtable-union) | 8 |
| `all_views` | Scaling analysis across view counts | 2-12 |

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in `config.py`
- Use fewer views (`--views 4` instead of `--views 12`)
- Enable gradient checkpointing

### Missing Embeddings

Ensure embeddings exist before training:

```bash
ls data/embeddings/{dataset}/datasets/
```

### CUDA Errors

Verify PyTorch installation:

```bash
uv run python -m experiments.misc.pytorch_check
```
