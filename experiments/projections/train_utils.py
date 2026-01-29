"""Common utilities for projection model training.

This module provides shared functionality for benchmark scripts:
- RunResult dataclass for tracking training results
- Embedding combination generation
- Metric extraction from PyTorch Lightning trainers
- Result saving (JSON, CSV formats)
"""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..config import EMBEDDING_DIMS, PathConfig

if TYPE_CHECKING:
    from lightning import Trainer
    from lightning.pytorch.loggers import Logger


@dataclass
class RunResult:
    """Training result for a single benchmark run."""

    status: Literal["SUCCESS", "SKIPPED", "FAILED"]
    model_name: str
    pipeline_names: list[str]
    training_time: float
    checkpoint_name: str | None = None
    config: dict[str, Any] | None = None
    error: str | None = None
    test_metrics: dict[str, float] | None = None
    epoch_metrics: list[dict[str, float]] | None = None
    eval_time: float | None = None
    train_max_gpu_memory_gb: float | None = None
    test_max_gpu_memory_gb: float | None = None

    @property
    def num_views(self) -> int:
        """Number of views in this experiment."""
        return len(self.pipeline_names)


def get_available_embeddings(dataset: str, subdir: str) -> list[Path]:
    """Get all .fa embedding directories for a dataset."""
    dataset_dir = PathConfig.from_defaults().embeddings_root / dataset / subdir
    if not dataset_dir.exists():
        return []
    return sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.suffix == ".fa"])


def order_embeddings_by_dim(embeddings: list[Path], reference_view: str) -> list[Path]:
    """Order embeddings by output dimension with reference view first.

    If a dimension is unknown, place it last by treating it as infinity.
    """

    def dim_key(path: Path) -> tuple[float, str]:
        """Retrieve embedding model from path and return its output dimension."""
        dim = float("inf")
        for key, value in EMBEDDING_DIMS.items():
            if path.stem.startswith(key):
                dim = value
                break
        return dim, path.stem

    stem_to_path = {p.stem: p for p in embeddings}
    if reference_view not in stem_to_path:
        raise ValueError(
            f"Reference view '{reference_view}' not found in embeddings: {sorted(stem_to_path)}"
        )

    ordered = sorted(embeddings, key=dim_key)
    return [stem_to_path[reference_view]] + [p for p in ordered if p.stem != reference_view]


def build_view_combinations(ordered_embeddings: list[Path], views: list[int]) -> list[list[Path]]:
    """Create a single deterministic combination per requested view count."""
    combos: list[list[Path]] = []
    for view_count in views:
        if view_count <= 0:
            continue
        if view_count > len(ordered_embeddings):
            raise ValueError(
                f"Requested {view_count} views but only {len(ordered_embeddings)} embeddings are "
                "available"
            )
        combos.append(ordered_embeddings[:view_count])
    return combos


def get_experiment_name(dataset: str, model_name: str, embedding_files: list[Path]) -> str:
    """Generate experiment name based on dataset, model, and embeddings."""

    def get_view_combination_signature(pipeline_names: list[str]) -> str:
        """Generate a signature from pipeline names (first~last for 3+ views)."""
        if not pipeline_names:
            return "none"
        if len(pipeline_names) == 1:
            return pipeline_names[0]
        return f"{pipeline_names[0]}~{pipeline_names[-1]}"

    num_views = len(embedding_files)
    signature = get_view_combination_signature([p.stem for p in embedding_files])
    return f"{dataset}-{model_name}-v={num_views}-{signature}"


def extract_test_metrics(trainer: "Trainer") -> dict[str, float]:
    """Extract test metrics from Lightning trainer.

    Args:
        trainer: PyTorch Lightning Trainer instance

    Returns:
        Dictionary of metric names to float values
    """
    metrics: dict[str, float] = {}
    if hasattr(trainer, "logged_metrics"):
        for key, value in trainer.logged_metrics.items():
            if key.startswith("test_"):
                try:
                    metrics[key] = float(value.item())
                except (TypeError, ValueError):
                    continue
    return metrics


def extract_epoch_metrics(trainer: "Trainer") -> list[dict[str, float]]:
    """Extract per-epoch metrics from trainer logger.

    Args:
        trainer: PyTorch Lightning Trainer instance

    Returns:
        List of dictionaries with per-epoch metrics
    """
    logger: Logger | None = getattr(trainer, "logger", None)
    if logger is not None:
        history: list[dict[str, float]] | None = getattr(logger, "history", None)
        if history is not None:
            return history
    return []


def save_results_json(results: list[RunResult], output_dir: Path) -> Path:
    """Save results to JSON file.

    Args:
        results: List of benchmark results
        output_dir: Output directory

    Returns:
        Path to saved JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([{**asdict(r), "num_views": r.num_views} for r in results], f, indent=2)

    return json_path


def save_results_summary_csv(results: list[RunResult], output_dir: Path) -> Path:
    """Save summary CSV with basic info and all metrics.

    Args:
        results: List of benchmark results
        output_dir: Output directory

    Returns:
        Path to saved CSV file
    """
    if not results:
        return output_dir / "summary.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"

    # Collect all metric keys
    all_metrics = {k for r in results for k in (r.test_metrics or {})}
    base_fields = [
        "status",
        "model_name",
        "num_views",
        "embeddings",
        "training_time",
        "eval_time",
        "train_max_gpu_memory_gb",
        "test_max_gpu_memory_gb",
        "experiment_name",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=base_fields + sorted(all_metrics), extrasaction="ignore"
        )
        writer.writeheader()
        for r in results:
            row = {
                "status": r.status,
                "model_name": r.model_name,
                "num_views": r.num_views,
                "embeddings": "+".join(r.pipeline_names),
                "training_time": f"{r.training_time:.2f}",
                "eval_time": f"{r.eval_time:.2f}" if r.eval_time is not None else "",
                "train_max_gpu_memory_gb": (
                    f"{r.train_max_gpu_memory_gb:.3f}"
                    if r.train_max_gpu_memory_gb is not None
                    else ""
                ),
                "test_max_gpu_memory_gb": (
                    f"{r.test_max_gpu_memory_gb:.3f}"
                    if r.test_max_gpu_memory_gb is not None
                    else ""
                ),
                "experiment_name": r.checkpoint_name or "",
                **{k: f"{v:.6f}" for k, v in (r.test_metrics or {}).items()},
            }
            writer.writerow(row)

    return csv_path


def save_results_epoch_csv(results: list[RunResult], output_dir: Path) -> Path:
    """Save per-epoch metrics to CSV file.

    Args:
        results: List of benchmark results
        output_dir: Output directory

    Returns:
        Path to saved CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_csv_path = output_dir / "per_epoch.csv"

    # Collect all unique field names from epoch metrics
    base_fields = {"model_name", "embeddings"}
    all_fields = base_fields.copy()
    for r in results:
        for epoch_data in r.epoch_metrics or []:
            all_fields.update(epoch_data.keys())

    if all_fields == base_fields:
        return epoch_csv_path

    fieldnames = ["model_name", "embeddings", *sorted(all_fields - base_fields)]
    with epoch_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            for epoch_data in r.epoch_metrics or []:
                row = {
                    "model_name": r.model_name,
                    "embeddings": "+".join(r.pipeline_names),
                    **{k: v for k, v in epoch_data.items() if k not in base_fields},
                }
                writer.writerow(row)

    return epoch_csv_path


def save_all_results(results: list[RunResult], output_dir: Path) -> None:
    """Save all result formats and print summary.

    Saves:
    - results.json: Full results with all metadata
    - summary.csv: Summary with metrics
    - per_epoch.csv: Per-epoch metrics if available

    Args:
        results: List of benchmark results
        output_dir: Output directory
    """
    json_path = save_results_json(results, output_dir)
    csv_path = save_results_summary_csv(results, output_dir)
    epoch_csv_path = save_results_epoch_csv(results, output_dir)

    print(f"\nResults saved to {output_dir}")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {epoch_csv_path}")
