"""Measure inference efficiency of projection models.

This script benchmarks the inference performance (runtime and GPU memory) of trained
ProjectionModel checkpoints. For each checkpoint, it measures the `project()` method
across different batch sizes and views.
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch
from loguru import logger
from torch import Tensor
from tqdm.auto import tqdm

from fedaugment.projections.main import load_projection_model
from fedaugment.projections.models import ProjectionModel
from fedaugment.utils import experiment_setup, get_max_gpu_memory_gb, reset_gpu_memory_tracking

from ..utils import CheckpointMetadata, find_checkpoints, load_checkpoint_metadata


@dataclass
class BenchmarkConfig:
    """Configuration for efficiency benchmarks."""

    experiment_group: str
    output_dir: Path
    device: torch.device
    num_repetitions: int  # First run discarded as warmup
    seed: int

    batch_sizes: list[int] = field(default_factory=lambda: [2**i for i in range(18)])
    rng: torch.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)


def measure_inference_time(
    model: ProjectionModel, batch: Tensor, view_idx: int, device: torch.device
) -> float:
    """Measure inference time for a single projection call.

    Args:
        model: ProjectionModel instance
        batch: Input tensor of shape (batch_size, embedding_dim)
        view_idx: Index of the view/embedding space
        device: Device for computation

    Returns:
        Inference time in milliseconds
    """
    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time the projection
    t0 = time.perf_counter_ns()
    _ = model.project(batch, view_idx)

    # Synchronize after computation
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ns = time.perf_counter_ns() - t0
    return elapsed_ns * 1e-6  # Convert to milliseconds


def benchmark_batch_size(
    model: ProjectionModel,
    ckpt_name: str,
    ckpt_metadata: CheckpointMetadata,
    batch_size: int,
    config: BenchmarkConfig,
) -> list[dict[str, int | float | str | None]]:
    """Benchmark a single batch size for a given view."""
    results: list[dict[str, int | float | str | None]] = []

    for view_idx in tqdm(
        range(ckpt_metadata.num_views),
        desc="Views",
        total=ckpt_metadata.num_views,
        leave=False,
        mininterval=1.0,
        dynamic_ncols=True,
    ):
        input_dim = ckpt_metadata.embedding_dims[view_idx]
        for repetition in range(config.num_repetitions):
            batch = torch.randn(
                (batch_size, input_dim), generator=config.rng, device=config.device
            )
            reset_gpu_memory_tracking()

            # Measure inference time
            inference_time_ms = measure_inference_time(model, batch, view_idx, config.device)

            # Skip first run (warmup)
            if repetition == 0:
                continue

            # Compute metrics and store results
            peak_memory_gb = get_max_gpu_memory_gb()
            results.append(
                {
                    "checkpoint_name": ckpt_name,
                    "view_idx": view_idx,
                    "batch_size": batch_size,
                    "repetition": repetition,
                    "input_dim": input_dim,
                    "output_dim": model.output_dim,
                    "inference_time_ms": inference_time_ms,
                    "peak_gpu_memory_gb": peak_memory_gb,
                    "device": config.device.type,
                }
            )

    return results


@torch.inference_mode()
def benchmark_checkpoint(checkpoint_path: Path, config: BenchmarkConfig) -> pl.DataFrame | None:
    """Benchmark a single checkpoint across batch sizes and views.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Benchmark configuration

    Returns:
        DataFrame with benchmark results, or None if benchmarking failed
    """
    ckpt_name = checkpoint_path.parent.name

    # Load checkpoint and metadata
    ckpt_metadata = load_checkpoint_metadata(checkpoint_path)
    model = load_projection_model(checkpoint_path, config.device)

    results: list[dict[str, int | float | str | None]] = []
    for batch_size in tqdm(
        config.batch_sizes,
        desc="Varying batch size",
        leave=False,
        mininterval=1.0,
        dynamic_ncols=True,
    ):
        try:
            results.extend(
                benchmark_batch_size(model, ckpt_name, ckpt_metadata, batch_size, config)
            )
        except torch.OutOfMemoryError:
            logger.warning(
                "OOM at batch_size={}, skipping larger batches for this checkpoint ({})",
                batch_size,
                ckpt_name,
            )
            # Clear cache and skip remaining batch sizes for this view
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not results:
        logger.warning("No results collected for checkpoint {}", ckpt_name)
        return None

    logger.info("Collected {} measurements for {}", len(results), ckpt_name)
    return pl.DataFrame(results)


def save_results(df: pl.DataFrame, checkpoint_name: str, output_dir: Path) -> None:
    """Save benchmark results to CSV.

    Args:
        df: DataFrame with benchmark results
        checkpoint_name: Name of the checkpoint
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_name}.csv"

    df.write_csv(output_path)
    logger.info("Results saved to: {}", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure projection model inference efficiency")
    parser.add_argument(
        "--experiment-group",
        type=str,
        required=True,
        help="Name of the experiment group to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/efficiency"),
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Ignore checkpoints containing any of these substrings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--num-repetitions",
        type=int,
        default=11,
        help="Number of repetitions per batch size (first run discarded as warmup)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_setup(
        "INFO", log_file_name="efficiency_{time:YYYY-MM-DD_HH-mm-ss}.log", filter_duplicates=False
    )

    # Configure device
    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: {}", device)

    # Create config
    config = BenchmarkConfig(
        experiment_group=args.experiment_group,
        output_dir=args.output_dir / args.experiment_group,
        device=device,
        num_repetitions=args.num_repetitions,
        seed=args.seed,
    )

    # Find checkpoints
    checkpoints = find_checkpoints(args.experiment_group, ignore=args.ignore)
    if not checkpoints:
        logger.error(
            "No checkpoints found in experiment group '{}' after filtering", args.experiment_group
        )
        return
    logger.info("Found {} checkpoint(s)", len(checkpoints))

    # Benchmark each checkpoint
    for idx, checkpoint_path in enumerate(checkpoints, 1):
        checkpoint_name = checkpoint_path.parent.name
        logger.info("Processing checkpoint {}/{}: {}", idx, len(checkpoints), checkpoint_name)

        try:
            df = benchmark_checkpoint(checkpoint_path, config)
            if df is not None:
                save_results(df, checkpoint_name, config.output_dir)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to benchmark {}: {}", checkpoint_name, e)
            logger.exception("Stack trace:")
            continue

    logger.success("Benchmark complete! Results saved to: {}", config.output_dir)


if __name__ == "__main__":
    main()
