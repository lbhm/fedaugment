# Projection Model Experiments

This directory contains scripts for training and evaluating embedding projection models that align multiple embedding views into a common vector space for table augmentation tasks.

## Overview

The projection experiments follow this pipeline:

```text
+---------------------------------------------------------------------------------+
|                                 TRAINING PHASE                                  |
+---------------------------------------------------------------------------------+

data/embeddings/webtable/split/train/           train_projection_models.py
    +-- mpnet-dj_adpt.fa/          --------->   (Train projection models)
    +-- distilroberta-dj_adpt.fa/                           |
    +-- gte_base-dj_adpt.fa/                                |
    +-- ... (N views)                                       v
                                                data/checkpoints/{group}/
                                                    +-- {model_name}/
                                                        +-- last.ckpt

+---------------------------------------------------------------------------------+
|                                EVALUATION PHASE                                 |
+---------------------------------------------------------------------------------+

         +---------------------------+-------------------------------+
         |                           |                               |
         v                           v                               v
evaluate_aligned.py       evaluate_centralized.py       evaluate_view_robustness.py
(Multi-view aligned)      (Single-view baseline)            (Per-view analysis)
         |                           |                               |
         v                           v                               v
logs/augmentations/       logs/augmentations/           logs/view-robustness/
  {dataset}/{group}/        {dataset}/centralized/        {dataset}/{model}/
    *.csv                     *.csv                         *_per_view.csv

+---------------------------------------------------------------------------------+
|                                 ANALYSIS PHASE                                  |
+---------------------------------------------------------------------------------+

     logs/augmentations/             logs/view-robustness/
             |                                 |
             v                                 v
analysis/augmentations.ipynb     analysis/view_robustness.ipynb
    (Aggregate results,             (Box plots for per-view
   generate tables/plots)            precision and recall)
```

## Files

| File                          | Purpose                                               |
|-------------------------------|-------------------------------------------------------|
| `train_projection_models.py`  | Train projection models (CL, LA2M, Vec2Vec, etc.)     |
| `evaluate_aligned.py`         | Evaluate join/union discovery with aligned embeddings |
| `evaluate_centralized.py`     | Single-view baseline (no projection)                  |
| `evaluate_view_robustness.py` | Per-view precision/recall breakdown                   |
| `train_utils.py`              | Shared training utilities                             |
| `config.py` (parent dir)      | Hyperparameter configurations                         |

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
