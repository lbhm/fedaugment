from __future__ import annotations

import math
import os
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, DirectoryPath, Field, PlainSerializer


def validate_data_splits(values: list[float]) -> list[float]:
    if not math.isclose(sum(values), 1):
        raise ValueError("data_splits must sum to 1.")
    if not len(values) == 3:  # noqa: PLR2004
        raise ValueError("data_splits must contain train, val, and test fraction.")
    return values


def serialize_kwargs_with_callables(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Serialize kwargs dict, converting callables to their __name__ attribute."""
    return {key: value.__name__ if callable(value) else value for key, value in kwargs.items()}


type ClassArgs = Annotated[dict[str, Any], PlainSerializer(serialize_kwargs_with_callables)]
type DataPath = DirectoryPath | list[DirectoryPath]


# NOTE: The keyword arguments in the following configs contain details such as the initial
# learning rate for optimizers or the temperature for the loss functions.
# Please review the respective fedaugment classes or PyTorch documentation for the individual
# parameters and set them as needed.


class TrainingConfig(BaseModel):
    experiment_name: str
    experiment_group: str = "all"
    experiment_version: int = 1
    seed: int = 42

    data_module: DataModuleConfig
    projection_model: ProjectionModelConfig
    trainer: TrainerConfig

    compile_model: bool = False


class ExplicitDataSplit(BaseModel):
    train: DataPath | None
    val: DataPath | None
    test: DataPath | None


class FractionalDataSplit(BaseModel):
    root: DataPath
    fractions: Annotated[list[float], AfterValidator(validate_data_splits)]


class DataModuleConfig(BaseModel):
    class_: Literal["DefaultDataModule", "SingleStepTrainingDataModule"]

    # Dataset params
    data: ExplicitDataSplit | FractionalDataSplit
    mmap_mode: Literal["r+"] | None = "r+"

    # Data loader params
    batch_size: int
    num_workers: int = os.cpu_count() or 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 5
    verify_column_ids: bool = True


class ProjectionModelConfig(BaseModel):
    class_: Literal[
        "ContrastiveLearningModel",
        "LocalIsometryModel",
        "NaiveModel",
        "ProcrustesModel",
        "Vec2VecModel",
    ]
    module_kwargs: Annotated[
        (
            CLModuleConfig
            | LocalIsometryModuleConfig
            | NaiveModuleConfig
            | ProcrustesModuleConfig
            | Vec2VecModuleConfig
        ),
        Field(discriminator="model_type"),
    ]
    criterion: CriterionConfig
    optimizers: list[OptimizerConfig]
    schedulers: list[LRSchedulerConfig]

    # Batching prevents OOM errors during metric computation (None = disable batching)
    metric_batch_size: int | None = None


class CLModuleConfig(BaseModel):
    """Configuration for ContrastiveLearning projection heads."""

    model_type: Literal["projection_head"] = "projection_head"
    out_dim: int
    hidden_dims: list[int] | None = None
    activation: Literal["relu", "gelu", "silu"] = "relu"
    normalization: Literal["batch", "layer"] | None = None
    dropout: float = 0.0

    no_val_orchestrator: bool = False  # Disable all validation metrics but val_loss
    no_test_orchestrator: bool = False  # Disable all test metrics but test_loss


class LocalIsometryModuleConfig(BaseModel):
    """Configuration for Local Isometry projection model.

    Default values are adopted from main.py in the original LA2M repository.
    """

    model_type: Literal["local_isometry"] = "local_isometry"
    reduced_dim: int = 14  # Reduced dimension for PCA projection (0 = no reduction)
    approximate: bool = True  # Use low-rank SVD approximation
    q: int = 1500  # Rank for approximation

    # Clustering parameters
    clustering_method: Literal["kmeans", "avg_linkage"] = "kmeans"
    num_clusters: int = 300


class NaiveModuleConfig(BaseModel):
    """Configuration for Naive projection model."""

    model_type: Literal["naive"] = "naive"
    mode: Literal["truncate", "pad"] = "truncate"


class ProcrustesModuleConfig(BaseModel):
    """Configuration for Procrustes projection model."""

    model_type: Literal["procrustes"] = "procrustes"
    approximate: bool = True  # Use low-rank SVD approximation
    q: int = 1500  # Rank for approximation
    with_rotation: bool = True  # Include rotation in transformation
    use_normalization: bool = True  # Apply normalization


class Vec2VecModuleConfig(BaseModel):
    """Configuration for Vec2Vec projection model."""

    model_type: Literal["vec2vec"] = "vec2vec"

    disc_dim: int = 1024  # hidden dimension for discriminator networks
    translator_dim: int = 1024  # hidden dimension for translator network
    latent_transform_dim: int = 1024  # hidden dimension for latent transform network
    latent_dim: int = 1024  # dimension of shared latent space

    disc_depth: int = 5  # number of hidden layers in discriminator networks
    translator_depth: int = 3  # number of hidden layers in translator network
    latent_transform_depth: int = 4  # number of hidden layers in latent transform network

    weight_init: Literal["kaiming", "orthogonal", "xavier"] = "kaiming"  # weight initialization
    norm_style: Literal["batch", "layer"] = "batch"  # normalization technique

    # GAN training optimizations
    use_spectral_norm: bool = False  # Apply spectral normalization to discriminators
    label_smoothing: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0  # 0.0-0.3 recommended
    disc_updates_per_gen: Annotated[int, Field(ge=1)] = 1  # Number of disc updates per gen update
    use_ema: bool = False  # Use exponential moving average for generator weights
    ema_decay: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.999  # EMA decay rate (if use_ema=True)
    # Regularization
    disc_dropout: Annotated[float, Field(ge=0.0, le=1.0)] = (
        0.0  # Dropout probability in discriminators
    )


class CriterionConfig(BaseModel):
    class_: Literal[
        "MultiViewInfoNCELoss",
        "MultiViewNTXentLoss",
        "Vec2VecLosses",
        "L1Loss",
        "NLLLoss",
        "NLLLoss2d",
        "PoissonNLLLoss",
        "GaussianNLLLoss",
        "KLDivLoss",
        "MSELoss",
        "BCELoss",
        "BCEWithLogitsLoss",
        "HingeEmbeddingLoss",
        "MultiLabelMarginLoss",
        "SmoothL1Loss",
        "HuberLoss",
        "SoftMarginLoss",
        "CrossEntropyLoss",
        "MultiLabelSoftMarginLoss",
        "CosineEmbeddingLoss",
        "MarginRankingLoss",
        "MultiMarginLoss",
        "TripletMarginLoss",
        "TripletMarginWithDistanceLoss",
        "CTCLoss",
    ]
    # Parameters to pass to the criterion's constructor
    kwargs: ClassArgs = {}


class OptimizerConfig(BaseModel):
    class_: Literal[
        "Adafactor",
        "Adadelta",
        "Adagrad",
        "Adam",
        "Adamax",
        "AdamW",
        "ASGD",
        "LBFGS",
        "NAdam",
        "RAdam",
        "RMSprop",
        "Rprop",
        "SGD",
        "SparseAdam",
    ]
    # Parameters to pass to the optimizers's constructor
    kwargs: ClassArgs = {}


class LRSchedulerConfig(BaseModel):
    class_: Literal[
        "ChainedScheduler",
        "ConstantLR",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "CyclicLR",
        "ExponentialLR",
        "LambdaLR",
        "LinearLR",
        "MultiplicativeLR",
        "MultiStepLR",
        "OneCycleLR",
        "PolynomialLR",
        "ReduceLROnPlateau",
        "SequentialLR",
        "StepLR",
    ]
    # When to call `scheduler.step()`
    interval: Literal["epoch", "step"] = "epoch"
    # How many epochs/steps should pass between calls to `scheduler.step()`
    frequency: int = 1
    # Metric to monitor for a `ReduceLROnPlateau` scheduler
    monitor: str | None = None
    # Parameters to pass to the scheduler's constructor
    kwargs: ClassArgs = {}


class TrainerConfig(BaseModel):
    # Required parameters
    max_epochs: int

    # Optional parameters with defaults
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    check_val_every_n_epoch: int = 1
    devices: list[int] | str | int = "auto"
    deterministic: bool = False
    enable_checkpointing: bool = True
    enable_model_summary: bool = True
    enable_progress_bar: bool = True
    log_every_n_steps: int = 50
    patience: Annotated[int, Field(ge=1)] | None = None
    max_time: str | None = "00:24:00:00"
    num_sanity_val_steps: int = 2
    precision: (
        Literal["16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true"] | None
    ) = None
    profiler: Literal["simple", "advanced", "pytorch"] | None = None


class HNSWConfig(BaseModel):
    # Distance metric to use for the HNSW index
    # NOTE: Cosine similarity equates to inner product if the vectors are normalized
    metric: Literal["cosine", "ip", "l2"] = "ip"
    # Number of bi-directional links created for every new element during construction
    M: int = 48
    # Size of the dynamic candidate list during construction
    # Bigger ef_construction leads to longer construction, but better index quality
    ef_construction: int = 200
    # Equivalent time/quality trade-off at query time
    ef: int = 200
    # Random seed for index reproducibility
    seed: int = 42
