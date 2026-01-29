"""Evaluate all checkpoints in an experiment group on a join or union discovery tasks.

Usage:
  uv run python -m experiments.projections.evaluate_aligned \
      --experiment-group my_experiment --datasets freyja santos_small
"""

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

from fedaugment.augmentations import BaseEvaluator, JoinDiscoveryEvaluator, UnionDiscoveryEvaluator
from fedaugment.config import HNSWConfig
from fedaugment.projections import project_embedding_collection
from fedaugment.projections.dataset import NPYDataset
from fedaugment.utils import experiment_setup

from ..config import ALIGNED_SUBDIR, DATASET_REGISTRY, DEFAULT_K_VALS
from ..utils import find_checkpoints, find_embedding_files, load_checkpoint_metadata


def evaluate_checkpoint(
    checkpoint: Path, dataset: str, output_dir: Path, k_vals: list[int]
) -> None:
    """Evaluate a single checkpoint on a dataset.

    Args:
        checkpoint: Path to checkpoint file
        dataset: Name of dataset to evaluate on
        output_dir: Directory to save results
        k_vals: K values for precision@k and recall@k
    """
    ckpt_name = checkpoint.parent.name

    # Load dataset configuration
    logger.info("Evaluating {} on dataset {}", ckpt_name, dataset)
    dataset_config = DATASET_REGISTRY[dataset]
    if not dataset_config:
        logger.warning("Unknown dataset: {}", dataset)
        return

    # Load checkpoint metadata
    ckpt_metadata = load_checkpoint_metadata(checkpoint)

    # Get embedding files
    candidate_files = find_embedding_files(
        dataset_config.candidate_emb_dir, ckpt_metadata.pipeline_names, ckpt_metadata.num_views
    )
    if not candidate_files:
        missing = set(ckpt_metadata.pipeline_names) if ckpt_metadata.pipeline_names else set()
        logger.warning("Skipping {} - missing candidate views: {}", ckpt_name, missing)
        return

    # Get query files
    query_files = find_embedding_files(
        dataset_config.query_emb_dir, ckpt_metadata.pipeline_names, ckpt_metadata.num_views
    )
    if not query_files:
        missing = set(ckpt_metadata.pipeline_names) if ckpt_metadata.pipeline_names else set()
        logger.warning("Skipping {} - missing query views: {}", ckpt_name, missing)
        return

    # Project embeddings
    aligned_dir = dataset_config.emb_dir / ALIGNED_SUBDIR
    aligned_dir.mkdir(parents=True, exist_ok=True)
    aligned_emb = aligned_dir / f"{ckpt_name}.fa"

    if not (aligned_emb / "embeddings.npy").exists():
        logger.info("Projecting: {}", ckpt_name)
        project_embedding_collection(
            dataset=NPYDataset(candidate_files),
            output_path=aligned_emb,
            checkpoint_path=checkpoint,
            split="uniform",
        )

    # Run evaluation
    evaluator: BaseEvaluator
    if dataset_config.augmentation_mode == "join":
        evaluator = JoinDiscoveryEvaluator(
            candidate_dataset=NPYDataset([aligned_emb], verify_column_ids=False),
            query_dataset=NPYDataset(query_files),
            query_path=dataset_config.queries_path,
            ground_truth_path=dataset_config.ground_truth_path,
            checkpoint_path=checkpoint,
            hnsw_config=HNSWConfig(),
        )
    else:
        evaluator = UnionDiscoveryEvaluator(
            candidate_dataset=NPYDataset([aligned_emb], verify_column_ids=False),
            query_dataset=NPYDataset(query_files),
            query_path=dataset_config.queries_path,
            ground_truth_path=dataset_config.ground_truth_path,
            checkpoint_path=checkpoint,
            hnsw_config=HNSWConfig(),
        )
    results = evaluator.evaluate(k_vals=k_vals, view_idx="iterate")

    # Handle empty results
    if len(results) == 0:
        logger.warning("Skipping {} - no valid queries (all skipped)", ckpt_name)
        return

    # Add metadata
    results = results.with_columns(
        [
            pl.lit(ckpt_name).alias("ckpt_name"),
            pl.lit(dataset).alias("dataset"),
            pl.lit(ckpt_metadata.num_views).alias("num_views"),
        ]
    )

    # Save individual result
    results.write_csv(output_dir / f"{ckpt_name}.csv")
    logger.success("Finished evaluating {} queries for {}", len(results), ckpt_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment group")
    parser.add_argument("--experiment-group", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/augmentations"))
    parser.add_argument("--k-vals", type=int, nargs="+", default=DEFAULT_K_VALS)
    parser.add_argument("--ignore", nargs="*", default=[])
    args = parser.parse_args()

    experiment_setup(
        "INFO", log_file_name="aug_aligned_{time:YYYY-MM-DD_HH-mm-ss}.log", filter_duplicates=False
    )

    checkpoints = find_checkpoints(args.experiment_group, ignore=args.ignore)
    if not checkpoints:
        logger.error(
            "No checkpoints found in experiment group '{}' after filtering", args.experiment_group
        )
        return
    logger.info("Found {} checkpoint(s)", len(checkpoints))

    for ckpt in checkpoints:
        for dataset in args.datasets:
            output_dir: Path = args.output_dir / dataset / args.experiment_group
            output_dir.mkdir(parents=True, exist_ok=True)
            evaluate_checkpoint(ckpt, dataset, output_dir, args.k_vals)


if __name__ == "__main__":
    main()
