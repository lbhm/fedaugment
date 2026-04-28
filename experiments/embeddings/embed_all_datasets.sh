#!/bin/bash

set -euo pipefail
ulimit -Sn 10000
cd "$(git rev-parse --show-toplevel)"

# Data paths - create ./data symlink to your storage (see README)
DATA_ROOT="data/datasets"
EMB_ROOT="data/embeddings"

MODELS=(mini_l6 mini_l12 mpnet distilroberta gtr_t5 gte_base kalm_instruct_v2_5 qwen3_06b qwen3_4b qwen3_8b llama_nemotron_8b kalm_gemma3_12b)

# Dataset configs (format: name|data_path|query_path)
DATASETS=(
  # "omnimatch_city_train|$DATA_ROOT/omnimatch_city_train/datasets/pq|$DATA_ROOT/omnimatch_city_train/datasets/pq"
  "omnimatch_city_test|$DATA_ROOT/omnimatch_city_test/datasets/pq|$DATA_ROOT/omnimatch_city_test/datasets/pq"
  # "omnimatch_culture_train|$DATA_ROOT/omnimatch_culture_train/datasets/pq|$DATA_ROOT/omnimatch_culture_train/datasets/pq"
  "omnimatch_culture_test|$DATA_ROOT/omnimatch_culture_test/datasets/pq|$DATA_ROOT/omnimatch_culture_test/datasets/pq"
  "freyja|$DATA_ROOT/freyja/datasets/pq|$DATA_ROOT/freyja/datasets/pq"
  "santos_small|$DATA_ROOT/santos_small/datasets/pq|$DATA_ROOT/santos_small/queries/pq"
)

echo "=== Embedding with models: ${MODELS[*]} ==="
for ds in "${DATASETS[@]}"; do
  IFS='|' read -r name data_path query_path <<< "$ds"

  echo "--- $name (data) ---"
  uv run python -m experiments.embeddings.embed_dataset \
    --data-path "$data_path" \
    --output-path "$EMB_ROOT/$name/datasets" \
    --models "${MODELS[@]}" \
    --strategy dj_adapted

  # Process queries separately if they differ from data or create symlink
  if [ "$data_path" != "$query_path" ]; then
    echo "--- $name (queries) ---"
    uv run python -m experiments.embeddings.embed_dataset \
      --data-path "$query_path" \
      --output-path "$EMB_ROOT/$name/queries" \
      --models "${MODELS[@]}" \
      --strategy dj_adapted
  else
    target="$EMB_ROOT/$name/queries"
    if [ ! -e "$target" ]; then
      ln -s datasets "$target"
      echo "Created symlink: $target -> datasets"
    fi
  fi
done

# Embed the curated training set as well as the full val and test split for webtable
for split in train-set split/val split/test; do
  echo "--- Webtable $split ---"
  uv run python -m experiments.embeddings.embed_dataset \
    --data-path "$DATA_ROOT/webtable/$split" \
    --output-path "$EMB_ROOT/webtable/$split" \
    --models "${MODELS[@]}" \
    --strategy dj_adapted
done

# We exclude the large LLMs for WebTable due to resource constraints
WEBTABLE_MODELS=(mini_l6 mini_l12 mpnet distilroberta gtr_t5 gte_base kalm_instruct_v2_5 qwen3_06b)
echo "--- webtable (data) ---"
uv run python -m experiments.embeddings.embed_dataset \
  --data-path "$DATA_ROOT/webtable/datasets/pq" \
  --output-path "$EMB_ROOT/webtable/datasets" \
  --models "${WEBTABLE_MODELS[@]}" \
  --strategy dj_adapted
echo "--- webtable (queries) ---"
uv run python -m experiments.embeddings.embed_dataset \
  --data-path "$DATA_ROOT/webtable/queries/union-pq" \
  --output-path "$EMB_ROOT/webtable/queries" \
  --models "${WEBTABLE_MODELS[@]}" \
  --strategy dj_adapted

# DeepJoin/Starmie finetuning evaluation
# For the additional finetuning experiments, you need to download the DeepJoin model checkpoint from
# https://github.com/BIT-DataLab/LakeBench/tree/main/join/Deepjoin/output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27
# and update `DEEPJOIN_CHECKPOINT` accordingly.
# To finetune a Starmie model and generate embeddings, please see the script at
# https://github.com/leonardgeissler/starmie/blob/fedaugment-integration/train_and_export_for_fedaugment.py
# and copy the resulting embeddings to $EMB_ROOT.
DEEPJOIN_CHECKPOINT="data/checkpoints/deepjoin/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27"

# Embed Freyja and Omnimatch with the DeepJoin model checkpoint
DATASETS=(
  "omnimatch_city_test|$DATA_ROOT/omnimatch_city_test/datasets/pq|$DATA_ROOT/omnimatch_city_test/datasets/pq"
  "omnimatch_culture_test|$DATA_ROOT/omnimatch_culture_test/datasets/pq|$DATA_ROOT/omnimatch_culture_test/datasets/pq"
  "freyja|$DATA_ROOT/freyja/datasets/pq|$DATA_ROOT/freyja/datasets/pq"
)

echo "=== Embedding with DeepJoin checkpoint: $DEEPJOIN_CHECKPOINT ==="
for ds in "${DATASETS[@]}"; do
  IFS='|' read -r name data_path query_path <<< "$ds"

  echo "--- $name (data) ---"
  uv run python -m experiments.embeddings.embed_dataset \
    --data-path "$data_path" \
    --output-path "$EMB_ROOT/$name/datasets" \
    --strategy dj_adapted \
    --local-checkpoints "$DEEPJOIN_CHECKPOINT" \
      --encoder-batch-size 1024

  if [ "$data_path" != "$query_path" ]; then
    echo "--- $name (queries) ---"
    uv run python -m experiments.embeddings.embed_dataset \
      --data-path "$query_path" \
      --output-path "$EMB_ROOT/$name/queries" \
      --strategy dj_adapted \
      --local-checkpoints "$DEEPJOIN_CHECKPOINT" \
      --encoder-batch-size 1024
  fi
done

echo "=== Model embedding done ==="
