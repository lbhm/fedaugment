"""Projection model training on a single dataset.

Models: ContrastiveLearningModel, LocalIsometryModel, ProcrustesModel, Vec2VecModel
Checkpoints saved to: data/checkpoints/{experiment_group}/

Experiment Name Format:
  {dataset}-{model}-v={views}-{signature}

  Components:
    - dataset: Dataset name (e.g., freyja, santos_small)
    - model: Model class acronym (e.g., cl_default, la2m_default, a2m, v2v)
    - views: Number of views (e.g., 8)
    - signature: Embedding combination signature (first~last embedding stems)

Examples:
    freyja-cl_shallow-v=4_arctic~snowflake
    santos_small-la2m-v=2_bert~roberta
    freyja-a2m-v=3-arctic~mpnet

Usage:
  # Basic training
  uv run python -m experiments.projections.train_projection_models \
      --dataset freyja --models LocalIsometryModel ContrastiveLearningModel

  # With experiment group
  uv run python -m experiments.projections.train_projection_models \
      --dataset freyja --models ContrastiveLearningModel \
      --views 2 4 --experiment-group my_experiment
"""

import json
import sys
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError, field_serializer, model_validator

from fedaugment.config import (
    DataModuleConfig,
    ExplicitDataSplit,
    ProjectionModelConfig,
    TrainerConfig,
    TrainingConfig,
)
from fedaugment.projections import train_projection_model
from fedaugment.utils import experiment_setup

from ..config import (
    AVAILABLE_DATASETS,
    EMBEDDING_DIMS,
    PROJECTION_MODEL_REGISTRY,
    TRAINSET_SUBDIR,
    VAL_SPLIT_SUBDIR,
    PathConfig,
)
from .train_utils import (
    RunResult,
    build_view_combinations,
    extract_epoch_metrics,
    extract_test_metrics,
    get_available_embeddings,
    get_experiment_name,
    order_embeddings_by_dim,
    save_all_results,
)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_group: str
    dataset: str
    output_dir: Path
    views: list[int]
    reference_view: str
    models: list[str]
    precision: Literal["16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true"] = (
        "bf16-mixed"
    )
    metric_batch_size: int | None = 65_536
    num_workers: int = 32
    seed: int = 42

    @field_serializer("output_dir")
    def serialize_path(self, value: Path) -> str:
        return str(value)

    @model_validator(mode="after")
    def validate_config(self) -> "ExecutionConfig":
        if self.dataset not in AVAILABLE_DATASETS:
            msg = f"Unknown dataset: {self.dataset}. Available: {AVAILABLE_DATASETS}"
            raise ValueError(msg)
        for model in self.models:
            if model not in PROJECTION_MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model: {model}. Available: {list(PROJECTION_MODEL_REGISTRY.keys())}"
                )
        # Check if reference view name is in EMBEDDING_DIMS (handles suffixes like -dj_adpt)
        if not any(self.reference_view.startswith(key) for key in EMBEDDING_DIMS):
            raise ValueError(
                f"Unknown reference views: {self.reference_view}. "
                f"Available views/models: {sorted(EMBEDDING_DIMS)}"
            )
        return self


def create_training_config(
    model_name: str,
    embedding_files: list[Path],
    precision: Literal["16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true"],
    metric_batch_size: int | None,
    num_workers: int,
    seed: int,
    experiment_name: str,
    experiment_group: str,
    verify_column_ids: bool = True,
) -> TrainingConfig:
    """Create training configuration for a model."""
    spec = PROJECTION_MODEL_REGISTRY[model_name]

    val_files: list[Path] | None = None
    if spec.patience is not None:
        # If patience is set, configure a validation split to use early stopping
        val_files = [
            Path(str(p).replace(TRAINSET_SUBDIR, VAL_SPLIT_SUBDIR)) for p in embedding_files
        ]

    return TrainingConfig(
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        seed=seed,
        data_module=DataModuleConfig(
            class_=spec.data_module_class,
            data=ExplicitDataSplit(train=embedding_files, val=val_files, test=None),
            batch_size=spec.batch_size,
            num_workers=num_workers,
            verify_column_ids=verify_column_ids,
        ),
        projection_model=ProjectionModelConfig(
            class_=spec.class_,
            module_kwargs=spec.module_kwargs,
            criterion=spec.criterion,
            optimizers=spec.optimizers,
            schedulers=spec.schedulers if spec.epochs > 0 else [],
            metric_batch_size=metric_batch_size,
        ),
        trainer=TrainerConfig(
            accelerator="gpu", max_epochs=spec.epochs, patience=spec.patience, precision=precision
        ),
        compile_model=True,
    )


def _print_training_details(
    model_name: str, cfg: ExecutionConfig, embedding_files: list[Path]
) -> None:
    """Print experiment details."""
    emb_names = [p.stem for p in embedding_files]
    num_views = len(embedding_files)

    print("\n" + "=" * 80)
    print(f"Training {model_name}")
    print(f"Dataset: {cfg.dataset} | Views: {num_views}")
    print(f"Embeddings: {', '.join(emb_names[:3])}{'...' if num_views > 3 else ''}")
    print("=" * 80)


def train_one(
    cfg: ExecutionConfig, model_name: str, embedding_files: list[Path]
) -> tuple[RunResult, Path | None]:
    """Train a single model configuration. Returns (result, checkpoint_path)."""
    # Generate experiment name to check for existing checkpoint
    experiment_name = get_experiment_name(cfg.dataset, model_name, embedding_files)
    ckpt_path = (
        PathConfig.from_defaults().checkpoints_root
        / cfg.experiment_group
        / experiment_name
        / "last.ckpt"
    )

    # Skip if checkpoint already exists
    if ckpt_path.exists():
        logger.warning(" Checkpoint for {} already exists - skipping", experiment_name)
        return (
            RunResult(
                status="SKIPPED",
                training_time=0.0,
                model_name=model_name,
                pipeline_names=[p.stem for p in embedding_files],
                config=PROJECTION_MODEL_REGISTRY[model_name].module_kwargs.model_dump(),
                checkpoint_name=experiment_name,
            ),
            ckpt_path,
        )

    _print_training_details(model_name, cfg, embedding_files)
    t0 = time.time()
    try:
        config = create_training_config(
            model_name=model_name,
            embedding_files=embedding_files,
            precision=cfg.precision,
            metric_batch_size=cfg.metric_batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            experiment_name=experiment_name,
            experiment_group=cfg.experiment_group,
        )

        trainer, _, perf_metrics = train_projection_model(config)

        # Collect metrics
        test_metrics = extract_test_metrics(trainer)
        epoch_metrics = extract_epoch_metrics(trainer)

        dt = time.time() - t0
        gpu_parts = []
        if perf_metrics.train_max_gpu_memory_gb:
            gpu_parts.append(f"train={perf_metrics.train_max_gpu_memory_gb:.2f}GB")
        if perf_metrics.test_max_gpu_memory_gb:
            gpu_parts.append(f"test={perf_metrics.test_max_gpu_memory_gb:.2f}GB")
        gpu_str = f", GPU: {', '.join(gpu_parts)}" if gpu_parts else ""
        logger.success(f"{config.experiment_name} completed in {dt:.2f}s{gpu_str}")

        return RunResult(
            status="SUCCESS",
            model_name=model_name,
            pipeline_names=[p.stem for p in embedding_files],
            training_time=perf_metrics.training_time,
            config=config.projection_model.module_kwargs.model_dump(),
            checkpoint_name=config.experiment_name,
            test_metrics=test_metrics,
            epoch_metrics=epoch_metrics,
            eval_time=perf_metrics.eval_time,
            train_max_gpu_memory_gb=perf_metrics.train_max_gpu_memory_gb,
            test_max_gpu_memory_gb=perf_metrics.test_max_gpu_memory_gb,
        ), ckpt_path
    except (RuntimeError, ValueError, OSError) as e:
        dt = time.time() - t0
        logger.error(f"{model_name} on {cfg.dataset} failed after {dt:.2f}s: {e!s}")
        traceback.print_exc()

        return RunResult(
            status="FAILED",
            model_name=model_name,
            pipeline_names=[p.stem for p in embedding_files],
            training_time=dt,
            error=str(e),
            config=PROJECTION_MODEL_REGISTRY[model_name].module_kwargs.model_dump(),
        ), None


def _print_training_header(
    cfg: ExecutionConfig, output_dir: Path, total_experiments: int, num_combos: int
) -> None:
    """Print training setup information."""
    print("=" * 80)
    print("PROJECTION MODEL TRAINING")
    print("=" * 80)
    print(f"Dataset: {cfg.dataset}")
    print(f"Models: {cfg.models}")
    print(f"Views: {cfg.views}")
    print(f"Reference view: {cfg.reference_view}")
    print(f"Output: {output_dir}")
    print(f"Total experiments: {total_experiments}")
    print(f"Testing {num_combos} view configurations")
    print("=" * 80)


def _print_training_summary(
    results: list[RunResult], total_time: float, total_experiments: int
) -> None:
    """Print final training summary."""
    n_successful = sum(r.status == "SUCCESS" for r in results)
    n_failed = sum(r.status == "FAILED" for r in results)
    n_skipped = sum(r.status == "SKIPPED" for r in results)
    pct = n_successful / max(1, total_experiments) * 100

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"Successful: {n_successful}/{total_experiments} ({pct:.0f}%)")
    if n_skipped:
        print(f"Skipped: {n_skipped}/{total_experiments}")
    if n_failed:
        print(f"Failed: {n_failed}/{total_experiments}")


def parse_args() -> ExecutionConfig:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Projection model training")
    parser.add_argument(
        "--dataset", required=True, type=str, choices=AVAILABLE_DATASETS, help="Training dataset"
    )
    parser.add_argument(
        "--output-dir", default="logs/projection-training", type=Path, help="Output directory"
    )
    parser.add_argument(
        "--views", type=int, nargs="+", default=[2, 3], help="View counts (e.g., 2 4 6)"
    )
    parser.add_argument(
        "--reference-view",
        default="distilroberta-dj_adpt",
        help=("Reference embedding view to force as the first view"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*PROJECTION_MODEL_REGISTRY.keys(), "all"],
        default=["cl_optim"],
        help="Models to train",
    )
    parser.add_argument(
        "--precision",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        default="bf16-mixed",
        help="Training precision",
    )
    parser.add_argument("--metric-batch-size", type=int, default=65_536, help="Metric batch size")
    parser.add_argument("--num-workers", type=int, default=32, help="Data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-group", default="all", help="Experiment group name")

    args = parser.parse_args()

    # Build config from args
    data = vars(args).copy()
    if "all" in data["models"]:
        data["models"] = PROJECTION_MODEL_REGISTRY.keys()

    try:
        return ExecutionConfig.model_validate(data)
    except ValidationError as exc:
        parser.error(str(exc))


def main() -> None:
    exec_cfg = parse_args()
    experiment_setup("INFO", log_file_name="train_{time:YYYY-MM-DD_HH-mm-ss}.log")
    torch.set_float32_matmul_precision("high")

    # Create output directory with timestamp
    output_dir = exec_cfg.output_dir / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with (output_dir / "config.json").open("w") as f:
        json.dump(exec_cfg.model_dump(), f, indent=2)

    # Get embeddings
    embeddings = get_available_embeddings(exec_cfg.dataset, TRAINSET_SUBDIR)
    if not embeddings:
        logger.warning("No embeddings found for dataset {}", exec_cfg.dataset)
        sys.exit(1)

    try:
        ordered_embeddings = order_embeddings_by_dim(embeddings, exec_cfg.reference_view)
        view_combos = build_view_combinations(ordered_embeddings, exec_cfg.views)
    except ValueError as exc:
        logger.error("{}", exc)
        sys.exit(1)
    total_experiments = len(exec_cfg.models) * len(view_combos)

    # Print header
    _print_training_header(exec_cfg, output_dir, total_experiments, len(view_combos))

    # Run experiments
    results: list[RunResult] = []
    checkpoints: dict[str, Path] = {}
    t0 = time.time()
    for i, (model_name, combo) in enumerate(
        ((m, c) for m in exec_cfg.models for c in view_combos), start=1
    ):
        logger.info("[{}/{}]", i, total_experiments)
        result, ckpt_path = train_one(exec_cfg, model_name, combo)
        results.append(result)
        if ckpt_path and result.checkpoint_name:
            checkpoints[result.checkpoint_name] = ckpt_path
        save_all_results(results, output_dir)

    # Print summary
    total_time = time.time() - t0
    _print_training_summary(results, total_time, total_experiments)
    logger.info("Results saved to: {}", output_dir)

    if any(r.status == "FAILED" for r in results):
        logger.warning("Some experiments failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
