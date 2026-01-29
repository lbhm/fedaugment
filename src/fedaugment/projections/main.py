"""Main training and inference pipeline for projection models.

This module provides the core functionality for training and using projection models
that align embeddings from multiple views (embedding model + strategy combinations)
into a common vector space. The aligned embeddings can then be used for downstream
tasks like join discovery and union discovery across decentralized data repositories.

Key functions:
    - train_projection_model: Train a projection model with Lightning
    - project_embedding_collection: Apply a trained model to project embeddings
    - load_projection_model: Load a trained model from a checkpoint

Supported projection model types:
    - ContrastiveLearningModel: Neural network with contrastive loss (NT-Xent/InfoNCE)
    - LocalIsometryModel: LA2M algorithm preserving local neighborhoods
    - ProcrustesModel: Orthogonal alignment via Procrustes analysis
    - Vec2VecModel: GAN-based vector translation
    - NaiveModel: Baseline truncation/padding approach
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightning as L
import numpy as np
import orjson
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from loguru import logger
from numpy.typing import NDArray

from fedaugment.config import (
    CLModuleConfig,
    DataModuleConfig,
    LocalIsometryModuleConfig,
    NaiveModuleConfig,
    ProcrustesModuleConfig,
    ProjectionModelConfig,
    TrainingConfig,
    Vec2VecModuleConfig,
)
from fedaugment.projections import data_modules, models
from fedaugment.projections.data_modules import DataModule
from fedaugment.projections.dataset import EmbeddingDataset
from fedaugment.projections.models import ProjectionModel
from fedaugment.utils import (
    LoguruLightningLogger,
    get_max_gpu_memory_gb,
    reset_gpu_memory_tracking,
)


@dataclass
class PerfMetrics:
    """Timing and resource metrics from model training."""

    training_time: float
    eval_time: float
    train_max_gpu_memory_gb: float | None
    test_max_gpu_memory_gb: float | None


def train_projection_model(
    config: TrainingConfig,
) -> tuple[L.Trainer, ProjectionModel, PerfMetrics]:
    """Train a projection model based on the provided configuration.

    Args:
        config: TrainingConfig object specifying data module, projection model,
            and trainer parameters.

    Returns:
        A tuple of (trainer, projection_model, metrics) where metrics contains
        timing and resource usage information.
    """
    logger.debug("Experiment configuration: {}", config.model_dump_json())
    L.seed_everything(config.seed, workers=True)

    data_module = init_data_module(config.data_module, config.seed)
    projection_model = init_projection_model(
        config.projection_model,
        data_module.pipeline_names,
        data_module.embedding_dims,
        config.data_module.batch_size,
    )
    if config.compile_model:
        logger.info("Compiling projection model...")
        projection_model.compile()  # type: ignore[no-untyped-call]
        logger.info("Projection model compiled")

    logger.info("Initializing trainer...")
    ckpt_monitor: str | None
    ckpt_mode: Literal["min", "max"] = "min"
    if config.projection_model.class_ in {"LocalIsometryModel", "NaiveModel", "ProcrustesModel"}:
        config.trainer.max_epochs = 1
        config.trainer.log_every_n_steps = 1
        config.trainer.num_sanity_val_steps = 0
        # torch.linalg does not support half precision for some operations
        config.trainer.precision = "32-true"
        config.trainer.enable_progress_bar = False
        ckpt_monitor = None
    elif config.projection_model.class_ == "ContrastiveLearningModel":
        ckpt_monitor = "val_loss"
    else:
        # vec2vec models
        val_metrics = list(projection_model.val_metrics.keys())
        ckpt_monitor = next((str(m) for m in val_metrics if "entity_stability" in str(m)), None)
        if ckpt_monitor is not None:
            ckpt_mode = "max"

    ckpt_dir = (
        Path.cwd() / "data" / "checkpoints" / config.experiment_group / config.experiment_name
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=(f"version-{config.experiment_version}-{{epoch:02d}}-{{val_loss:.4f}}"),
        monitor=ckpt_monitor,
        save_last="link",
        save_top_k=1,
        mode=ckpt_mode,
    )
    callbacks: list[Callback] = [checkpoint_callback]

    if ckpt_monitor is not None and config.trainer.patience is not None:
        # Use Lightning's EarlyStopping to halt when validation metric stops improving
        early_stop_callback = EarlyStopping(
            monitor=ckpt_monitor,
            mode=ckpt_mode,
            patience=config.trainer.patience,
            min_delta=0.0,
            verbose=True,
        )
        logger.info(
            "EarlyStopping enabled on '{}' with mode={} and patience={}",
            ckpt_monitor,
            ckpt_mode,
            config.trainer.patience,
        )
        callbacks.append(early_stop_callback)

    trainer_kwargs = config.trainer.model_dump()
    trainer_kwargs.pop("patience", None)  # NOTE: patience is only used by EarlyStopping

    lightning_logger = LoguruLightningLogger(config.experiment_name, config.experiment_version)
    trainer = L.Trainer(
        logger=lightning_logger,
        callbacks=callbacks,
        default_root_dir=Path.cwd() / "logs",
        **trainer_kwargs,
    )
    logger.info("Trainer initialized")

    if data_module.val_dataloader() is None:
        # Disable validation if no val_dataloader is provided
        trainer.limit_val_batches = 0

    # Training phase
    training_time = 0.0
    train_max_gpu_memory_gb: float | None = None
    if data_module.train_dataloader() is not None:
        logger.info("Starting training phase...")
        reset_gpu_memory_tracking()
        t0 = time.perf_counter()
        trainer.fit(projection_model, data_module)
        training_time = time.perf_counter() - t0
        train_max_gpu_memory_gb = get_max_gpu_memory_gb()

    # Testing phase
    eval_time = 0.0
    test_max_gpu_memory_gb: float | None = None
    if data_module.test_dataloader() is not None:
        logger.info("Starting testing phase...")
        reset_gpu_memory_tracking()
        t0 = time.perf_counter()
        trainer.test(projection_model, data_module)
        eval_time = time.perf_counter() - t0
        test_max_gpu_memory_gb = get_max_gpu_memory_gb()

    logger.info("Best model checkpoint saved at {}", checkpoint_callback.last_model_path)

    # Save memory and reduce the number of open files
    del data_module

    metrics = PerfMetrics(
        training_time=training_time,
        eval_time=eval_time,
        train_max_gpu_memory_gb=train_max_gpu_memory_gb,
        test_max_gpu_memory_gb=test_max_gpu_memory_gb,
    )
    return trainer, projection_model, metrics


@torch.no_grad()
def project_embedding_collection(
    dataset: EmbeddingDataset[Any],
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 128,
    device: Literal["auto", "cpu", "cuda"] | torch.device = "auto",
    split: Literal["uniform"] | list[float] = "uniform",
    seed: int | None = None,
) -> None:
    """Project an embedding collection into an aligned space using a trained projection model.

    Args:
        dataset: Dataset containing embeddings to project.
        checkpoint_path: Path to the trained projection model checkpoint.
        output_path: Path to save the projected embeddings.
        batch_size: Batch size for processing embeddings.
        device: Device to use for computation ("auto", "cpu", "cuda").
        split: How to choose a view for each embedding. Can be:
            - "uniform": Uniform split size across all views.
            - List of fractions: Custom fractions for each view.
        seed: If None, data splits are sequential. If an integer, data splits are shuffled
            with the given random seed for reproducibility.
    """
    # Create output directory if it doesn't exist
    output_path = output_path.with_suffix(".fa")
    if output_path.exists():
        logger.error("Output path {} already exists. Aborting projection.", output_path)
        return
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up device
    if not isinstance(device, torch.device):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
    logger.debug("Using device: {}", device)

    # Load and initialize the projection model
    logger.debug("Loading projection model from {}", checkpoint_path)
    projection_model = load_projection_model(checkpoint_path, device)

    # Determine split indices
    if split == "uniform":
        split = [1.0 / dataset.num_views] * dataset.num_views
    elif len(split) != dataset.num_views:
        raise ValueError(
            f"Length of split ({len(split)}) does not match number of views ({dataset.num_views})."
        )
    split_indices = random_split(np.arange(len(dataset)), split, seed=seed)

    # Process embeddings
    logger.info("Projecting embeddings {} from {} views", len(dataset), dataset.num_views)
    projected_emb = np.zeros((len(dataset), projection_model.output_dim), dtype=np.float32)
    n_rows_written = 0
    for view_idx, sample_indices in enumerate(split_indices):
        for batch_idx in range(0, len(sample_indices), batch_size):
            batch_sample_indices = sample_indices[batch_idx : batch_idx + batch_size]
            emb_batch = torch.from_numpy(dataset.get_view(view_idx, batch_sample_indices)).to(
                device
            )

            # Project embeddings
            projected_batch = projection_model.project(emb_batch, view_idx)
            projected_batch = F.normalize(projected_batch, dim=1)
            projected_emb[batch_sample_indices] = projected_batch.cpu().numpy()
            n_rows_written += len(batch_sample_indices)

        logger.info(
            "Processed {} embeddings from view  {}/{} ({})",
            len(sample_indices),
            view_idx + 1,
            dataset.num_views,
            dataset.pipeline_names[view_idx],
        )

    if n_rows_written != len(dataset):
        logger.warning(
            "Projected row count ({}) does not match dataset size ({}).",
            n_rows_written,
            len(dataset),
        )
    else:
        logger.success("Successfully projected {} embeddings", n_rows_written)

    # Save data
    np.save(output_path / "embeddings.npy", projected_emb)
    np.save(output_path / "column_ids.npy", dataset.column_ids)
    metadata = {
        "pipeline": "aligned_embeddings",
        "embedding_shape": list(projected_emb.shape),
        "embedding_dtype": str(projected_emb.dtype),
        "num_views": dataset.num_views,
        "split": split,
        "seed": seed,
        "source_pipelines": dataset.pipeline_names,
        "projection_model_checkpoint": str(checkpoint_path),
    }
    Path(output_path / "metadata.json").write_bytes(
        orjson.dumps(metadata, option=orjson.OPT_INDENT_2)
    )
    logger.info("Saved projected embeddings to {}", output_path)


def load_projection_model(checkpoint_path: Path, device: torch.device) -> ProjectionModel:
    """Load a trained projection model from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        ProjectionModel: The trained projection model loaded from the checkpoint,
            moved to the specified device and set to evaluation mode.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    projection_model: ProjectionModel
    if isinstance(ckpt["hyper_parameters"]["module_kwargs"], CLModuleConfig):
        projection_model = models.ContrastiveLearningModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    elif isinstance(ckpt["hyper_parameters"]["module_kwargs"], LocalIsometryModuleConfig):
        projection_model = models.LocalIsometryModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    elif isinstance(ckpt["hyper_parameters"]["module_kwargs"], NaiveModuleConfig):
        projection_model = models.NaiveModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    elif isinstance(ckpt["hyper_parameters"]["module_kwargs"], ProcrustesModuleConfig):
        projection_model = models.ProcrustesModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    elif isinstance(ckpt["hyper_parameters"]["module_kwargs"], Vec2VecModuleConfig):
        projection_model = models.Vec2VecModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    else:
        raise TypeError("Unsupported projection model type found in the checkpoint.")

    projection_model.to(device)
    projection_model.eval()
    return projection_model


####################
# Helper Functions #
####################


def init_data_module(config: DataModuleConfig, seed: int) -> DataModule:
    """Initialize a data module based on the provided configuration.
    
    Args:
        config: DataModuleConfig object specifying the data module class and parameters.
        seed: Random seed for data module initialization.
    Returns:
        DataModule: An initialized data module instance.
    """
    data_module_class: type[DataModule] = getattr(data_modules, config.class_)
    return data_module_class(config, seed)


def init_projection_model(
    config: ProjectionModelConfig,
    pipeline_names: list[str],
    embedding_dims: list[int],
    data_batch_size: int,
) -> ProjectionModel:
    """Initialize a projection model based on the provided configuration.

    Args:
        config: ProjectionModelConfig object specifying the projection model class and parameters.
        pipeline_names: List of pipeline names corresponding to the input embeddings.
        embedding_dims: List of embedding dimensions corresponding to the input embeddings.
        data_batch_size: Batch size for data loading during training.
    Returns:
        ProjectionModel: An initialized projection model instance.
    """
    model_class: type[ProjectionModel] = getattr(models, config.class_)
    return model_class(
        module_kwargs=config.module_kwargs,
        criterion=config.criterion,
        optim_configs=config.optimizers,
        sched_configs=config.schedulers,
        pipeline_names=pipeline_names,
        embedding_dims=embedding_dims,
        data_batch_size=data_batch_size,
        metric_batch_size=config.metric_batch_size,
    )


def random_split[T: np.generic](
    indices: NDArray[T], fractions: list[float], seed: int | None = None
) -> list[NDArray[T]]:
    """Randomly split indices into parts.

    Args:
        indices: Array of indices to split (shuffled in-place if there is a seed).
        fractions: List of fractions for each split. Must sum to 1.
        seed: Random seed for shuffling. If None, no shuffling is done.
    Returns:
        List of arrays of indices for each split.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1.")

    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n = len(indices)
    counts = np.floor(np.array(fractions) * n).astype(int)

    # Distribute remainder equally
    remainder = n - counts.sum()
    for i in range(remainder):
        counts[i % len(counts)] += 1

    # Perform splits
    return np.split(indices, np.cumsum(counts)[:-1])
