import argparse
import sys
from pathlib import Path

from loguru import logger

from fedaugment.augmentations import BaseEvaluator, JoinDiscoveryEvaluator, UnionDiscoveryEvaluator
from fedaugment.config import HNSWConfig
from fedaugment.projections.dataset import NPYDataset
from fedaugment.utils import experiment_setup

from ..config import DATASET_REGISTRY, DEFAULT_K_VALS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()))
    args = parser.parse_args()

    experiment_setup(
        "INFO",
        log_file_name="aug_centralized_{time:YYYY-MM-DD_HH-mm-ss}.log",
        filter_duplicates=False,
    )

    for dataset_id in args.datasets:
        dataset_config = DATASET_REGISTRY[dataset_id]
        if not dataset_config:
            logger.warning("Unknown dataset: {}", dataset_id)
            sys.exit(1)

        logger.info("Using embeddings from {}", dataset_config.candidate_emb_dir)
        logger.info("Using queries from {}", dataset_config.query_emb_dir)

        embeddings = sorted(
            [
                p
                for p in dataset_config.candidate_emb_dir.iterdir()
                if p.is_dir() and p.suffix == ".fa"
            ]
        )
        queries = sorted(
            [p for p in dataset_config.query_emb_dir.iterdir() if p.is_dir() and p.suffix == ".fa"]
        )

        metrics_path = Path(f"logs/augmentations/{dataset_id}/centralized")
        metrics_path.mkdir(parents=True, exist_ok=True)
        for e, q in zip(embeddings, queries, strict=True):
            logger.info("Evaluating {}", e.stem)
            assert e.stem == q.stem

            evaluator: BaseEvaluator
            if dataset_config.augmentation_mode == "join":
                evaluator = JoinDiscoveryEvaluator(
                    candidate_dataset=NPYDataset([e], verify_column_ids=False),
                    query_dataset=NPYDataset([q], verify_column_ids=False),
                    query_path=dataset_config.queries_path,
                    ground_truth_path=dataset_config.ground_truth_path,
                    checkpoint_path=None,
                    hnsw_config=HNSWConfig(),
                )
            else:
                evaluator = UnionDiscoveryEvaluator(
                    candidate_dataset=NPYDataset([e], verify_column_ids=False),
                    query_dataset=NPYDataset([q], verify_column_ids=False),
                    query_path=dataset_config.queries_path,
                    ground_truth_path=dataset_config.ground_truth_path,
                    checkpoint_path=None,
                    hnsw_config=HNSWConfig(),
                )

            results = evaluator.evaluate(k_vals=DEFAULT_K_VALS, view_idx=0)
            results.write_csv(metrics_path / f"{e.stem}.csv")
            logger.info(
                "Saved {} query results to {}", len(results), metrics_path / f"{e.stem}.csv"
            )

            del evaluator

        logger.success("Evaluation complete for dataset {}", dataset_id)
