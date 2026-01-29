#!/bin/bash

set -euxo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"

# Verify `data/datasets` exists and is a directory (follows symlinks)
if [[ ! -d "data/datasets" ]]; then
    echo "Error: data/datasets is not a directory or a symlink to a directory." >&2
    echo "" >&2
    echo "Please download the required datasets first:" >&2
    echo "  1. Freyja:         https://mydisk.cs.upc.edu/s/QHJbKcyeacxq35f" >&2
    echo "  2. OmniMatch:      https://zenodo.org/records/15705578" >&2
    echo "  3. Santos (Small): https://zenodo.org/records/7758091" >&2
    echo "  4. LakeBench:      https://github.com/DB-121143/LakeBench" >&2
    echo "" >&2
    echo "Then place them in data/datasets/ or create a symlink:" >&2
    echo "  ln -s /path/to/your/datasets data" >&2
    echo "" >&2
    echo "See README.md for the expected directory structure." >&2
    exit 1
fi

##############
# Embeddings #
##############

./embeddings/embed_all_datasets.sh

############
# Curation #
############

uv run python -m experiments.curation.curate_webtable
uv run python -m experiments.curation.embed_webtable
uv run python -m experiments.curation.train_webtable

# Symlink for the curated training subset
ln -s data/datasets/webtable/train-set data/datasets/webtable/sample-050/random-mpnet-dj_adapt-k=630493

###############
# Projections #
###############

# Training (main models with 8 and 12 views)
uv run python -m experiments.projections.train_projection_models --experiment-group 12_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v union_minus union_plus --views 12 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group 08_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v union_minus union_plus --views 8 --seed 123

# Training (performance over the number of views)
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 2 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 3 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 4 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 5 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 6 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 7 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 9 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 10 --seed 123
uv run python -m experiments.projections.train_projection_models --experiment-group all_views --dataset webtable --models cl_optim la2m_default la2m_nopca v2v --views 11 --seed 123
# Copy the existing 8- and 12-view checkpoints into the all_views group
find data/checkpoints/12_views -mindepth 1 -maxdepth 1 -type d \( -name '*la2m_*' -o -name '*cl_optim*' -o -name '*v2v*' \) -exec cp -r {} data/checkpoints/all_views/ \;
find data/checkpoints/08_views -mindepth 1 -maxdepth 1 -type d \( -name '*la2m_*' -o -name '*cl_optim*' -o -name '*v2v*' \) -exec cp -r {} data/checkpoints/all_views/ \;

# Eval for all but webtable-union
uv run python -m experiments.projections.evaluate_centralized --datasets freyja omnimatch_city_test omnimatch_culture_test santos-union
uv run python -m experiments.projections.evaluate_aligned --datasets freyja omnimatch_city_test omnimatch_culture_test santos-union --experiment-group 12_views
uv run python -m experiments.projections.evaluate_view_robustness --datasets freyja omnimatch_city_test omnimatch_culture_test santos-union --experiment-group 12_views --ignore union_

# Eval for webtable-union
uv run python -m experiments.projections.evaluate_centralized --datasets webtable-union
uv run python -m experiments.projections.evaluate_aligned --datasets webtable-union --experiment-group 08_views
uv run python -m experiments.projections.evaluate_view_robustness --datasets webtable-union --experiment-group 08_views --ignore union_ v2v

# Eval for all view counts (to measure performance over the number of views)
uv run python -m experiments.projections.evaluate_aligned --datasets freyja omnimatch_city_test omnimatch_culture_test santos-union --experiment-group all_views

##############
# Efficiency #
##############

# NOTE: We train another model without validation overhead here since early stopping anyway does not trigger during cl_optim training with many views
uv run python -m experiments.projections.train_projection_models --experiment-group 12_views --dataset webtable --models cl_noval --views 12 --seed 123
uv run python -m experiments.efficiency.measure_projections --experiment-group 12_views
