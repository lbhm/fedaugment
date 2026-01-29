"""Evaluate view robustness: precision@k and recall@k per view with box plots.

Projects all embeddings through each view separately and compares translation quality.

Usage:
  uv run python -m experiments.projections.evaluate_view_robustness \
      --experiment-group my_experiment --datasets freyja
"""

import argparse
import shutil
from pathlib import Path

import polars as pl
from loguru import logger

from fedaugment.augmentations import BaseEvaluator, JoinDiscoveryEvaluator, UnionDiscoveryEvaluator
from fedaugment.config import HNSWConfig
from fedaugment.projections import project_embedding_collection
from fedaugment.projections.dataset import NPYDataset
from fedaugment.utils import experiment_setup

from ..config import ALIGNED_PER_VIEW_SUBDIR, DATASET_REGISTRY, DEFAULT_K_VALS, DatasetConfig
from ..utils import (
    CheckpointMetadata,
    find_checkpoints,
    find_embedding_files,
    load_checkpoint_metadata,
    safe_exp_name,
)


def evaluate_single_view(
    checkpoint: Path,
    candidate_files: list[Path],
    query_files: list[Path],
    dataset_config: DatasetConfig,
    view_idx: int,
    metadata: CheckpointMetadata,
    k_vals: list[int],
    aligned_dir: Path,
    ckpt_name: str,
    safe_ckpt_name: str,
    delete_aligned: bool = True,
) -> pl.DataFrame | None:
    """Evaluate a single view by projecting all embeddings through it.

    Args:
        checkpoint: Path to checkpoint
        candidate_files: List of candidate embedding files
        query_files: List of query embedding files
        dataset_config: Dataset configuration
        view_idx: Index of the view to evaluate
        metadata: Checkpoint metadata
        k_vals: K values for evaluation
        aligned_dir: Directory for aligned embeddings
        ckpt_name: Name of the model checkpoint
        safe_ckpt_name: Filesystem-safe checkpoint name
        delete_aligned: Whether to delete aligned embeddings after evaluation

    Returns:
        DataFrame with evaluation results or None
    """
    view_name = (
        metadata.pipeline_names[view_idx] if metadata.pipeline_names else f"view_{view_idx}"
    )
    # Use a filesystem-safe (possibly shortened) experiment name for paths
    aligned_emb = aligned_dir / f"{safe_ckpt_name}_v{view_idx}.fa"

    # Create single-view split: all embeddings through this view
    split = [0.0] * metadata.num_views
    split[view_idx] = 1.0

    if not (aligned_emb / "embeddings.npy").exists():
        npy_dataset = NPYDataset(candidate_files)
        project_embedding_collection(
            dataset=npy_dataset, output_path=aligned_emb, checkpoint_path=checkpoint, split=split
        )
    else:
        logger.warning(
            "Aligned embeddings for {} view {} already exist, skipping projection",
            ckpt_name,
            view_name,
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
    results = evaluator.evaluate(k_vals=k_vals, view_idx=view_idx)

    if len(results) == 0:
        logger.warning("Skipping {} view {} - no results", ckpt_name, view_name)
        return None

    results = results.with_columns(
        [
            pl.lit(view_name).alias("view"),
            pl.lit(view_idx).alias("view_idx"),
            pl.lit(ckpt_name).alias("ckpt_name"),
        ]
    )

    if delete_aligned:
        # Cleanup aligned embeddings to save space
        shutil.rmtree(aligned_emb)

    return results


def evaluate_view_robustness(
    checkpoint: Path, dataset: str, output_dir: Path, k_vals: list[int]
) -> None:
    """Evaluate each view separately and collect per-view metrics.

    Args:
        checkpoint: Path to checkpoint file
        dataset: Name of dataset to evaluate on
        output_dir: Directory to save results
        k_vals: K values for precision@k and recall@k
    """
    ckpt_name = checkpoint.parent.name
    safe_ckpt_name = safe_exp_name(ckpt_name)

    # Skip if combined per-view results already exist
    combined_outfile = output_dir / f"{safe_ckpt_name}_per_view.csv"
    if combined_outfile.exists():
        logger.warning("Skipping {} - results already exist", ckpt_name)
        return

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

    aligned_dir = dataset_config.emb_dir / ALIGNED_PER_VIEW_SUBDIR
    aligned_dir.mkdir(parents=True, exist_ok=True)

    all_view_results: list[pl.DataFrame] = []
    for view_idx in range(ckpt_metadata.num_views):
        logger.info(
            "Evaluating projection quality of view {}/{}",
            ckpt_name,
            view_idx + 1,
            ckpt_metadata.num_views,
        )
        result = evaluate_single_view(
            checkpoint=checkpoint,
            candidate_files=candidate_files,
            query_files=query_files,
            dataset_config=dataset_config,
            view_idx=view_idx,
            metadata=ckpt_metadata,
            k_vals=k_vals,
            aligned_dir=aligned_dir,
            ckpt_name=ckpt_name,
            safe_ckpt_name=safe_ckpt_name,
        )
        if result is not None:
            all_view_results.append(result)

    combined = pl.concat(all_view_results)
    combined.write_csv(combined_outfile)
    logger.success("Finished evaluating {} views for {}", ckpt_metadata.num_views, ckpt_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate view robustness")
    parser.add_argument("--experiment-group", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/view-robustness"))
    parser.add_argument("--k-vals", type=int, nargs="+", default=DEFAULT_K_VALS)
    parser.add_argument("--ignore", nargs="*", default=[])
    args = parser.parse_args()

    experiment_setup(
        "INFO",
        log_file_name="aug_robustness_{time:YYYY-MM-DD_HH-mm-ss}.log",
        filter_duplicates=False,
    )

    checkpoints = find_checkpoints(args.experiment_group, ignore=args.ignore)
    if not checkpoints:
        logger.error(
            "No checkpoints found in experiment group '{}' after filtering", args.experiment_group
        )
        return
    logger.info("Found {} checkpoint(s)", len(checkpoints))

    for ckpt in checkpoints:
        logger.info("Evaluating checkpoint: {}", ckpt.parent.name)
        for dataset in args.datasets:
            output_dir: Path = args.output_dir / dataset / args.experiment_group
            output_dir.mkdir(parents=True, exist_ok=True)
            evaluate_view_robustness(ckpt, dataset, output_dir, args.k_vals)


if __name__ == "__main__":
    main()
