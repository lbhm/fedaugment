# Experiments

This directory contains all scripts and configurations for reproducing the experiments in the FedAugment paper.

## Directory Structure

```bash
experiments/
├── config.py                 # Centralized configuration (models, datasets, hyperparameters)
├── utils.py                  # Shared utility functions
├── run_all.sh                # Main script to run all experiments
│
├── embeddings/               # Embedding generation
│   ├── embed_dataset.py      # Generate embeddings for a single dataset
│   └── embed_all_datasets.sh # Batch embedding generation for all datasets
│
├── curation/                 # Data curation for training set selection
│   ├── curate_webtable.py    # Apply curation algorithms
│   ├── embed_webtable.py     # Generate embeddings for curated subsets
│   └── train_webtable.py     # Train models on curated data
│
├── projections/                      # Projection model training and evaluation
│   ├── train_projection_models.py    # Train alignment models
│   ├── evaluate_aligned.py           # Evaluate aligned embeddings
│   ├── evaluate_centralized.py       # Centralized augmentation evaluation
│   └── evaluate_view_robustness.py   # Per-view precision/recall analysis
│
├── efficiency/                       # Performance measurements
│   └── measure_projections.py        # Training time and memory benchmarks
│
└── misc/                             # Miscellaneous utilities
    ├── curation_figure_data.py       # Data for curation visualizations
    └── pytorch_check.py              # Verify PyTorch installation
```

## Running Experiments

To run all experiments from scratch:

```bash
# Ensure datasets are in place
ls data/datasets/

# Run everything
bash experiments/run_all.sh
```

**Note:** The full experiment set requires significant resources (500+ GB RAM, A100 GPU) and takes several days to complete.

To run individual experiments, see the commands in `run_all.sh` for examples.
