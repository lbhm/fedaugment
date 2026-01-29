"""Centralized configuration for dataset benchmarks.

The CL presets encapsulate different architecture and hyperparameter choices for the
ContrastiveLearningModel, making it easy to run benchmarks with different configurations.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from fedaugment.config import (
    CLModuleConfig,
    CriterionConfig,
    LocalIsometryModuleConfig,
    LRSchedulerConfig,
    NaiveModuleConfig,
    OptimizerConfig,
    ProcrustesModuleConfig,
    Vec2VecModuleConfig,
)
from fedaugment.embeddings.models import BatchOpenAIModel, EmbeddingModel, SentenceTransformerModel
from fedaugment.utils import get_repo_root


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Configuration needed to instantiate a SentenceTransformerModel."""

    model_id: str
    alias: str
    default_batch_size: int
    compile_model: bool = True
    trust_remote_code: bool = False
    revision: str | None = None
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None
    max_seq_length: int | None = None
    supports_flash_attn: bool = False

    def build(
        self,
        cache_folder: str | None = None,
        flash_attn: bool = False,
        encoder_batch_size: int | None = None,
    ) -> EmbeddingModel:
        """Instantiate the embedding model."""
        if self.default_batch_size == -1:
            # This is a hack to handle API-based models
            return BatchOpenAIModel(self.model_id, alias=self.alias)  # type: ignore[arg-type]

        batch_size = encoder_batch_size or self.default_batch_size
        model_kwargs = dict(self.model_kwargs or {})
        if self.supports_flash_attn and flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = SentenceTransformerModel(
            model_id=self.model_id,
            alias=self.alias,
            compile_model=self.compile_model,
            encoder_batch_size=batch_size,
            cache_folder=cache_folder,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )

        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length
        return model


EMBEDDING_MODEL_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "kalm_gemma3_12b": EmbeddingModelSpec(
        model_id="tencent/KaLM-Embedding-Gemma3-12B-2511",
        alias="kalm_gemma3_12b",
        default_batch_size=1,
        revision="5153cebe4e45fdb5cba56ac991b6e9431644b1df",
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
        supports_flash_attn=True,
    ),
    "kalm_instruct_v2_5": EmbeddingModelSpec(
        model_id="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        alias="kalm_instruct_v2_5",
        default_batch_size=48,
        revision="753c6fe26abc20a32aeb162003aa03457d15db2f",
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
        supports_flash_attn=True,
    ),
    "llama_nemotron_8b": EmbeddingModelSpec(
        model_id="nvidia/llama-embed-nemotron-8b",
        alias="llama_nemotron_8b",
        default_batch_size=4,
        trust_remote_code=True,
        revision="1acaf42b890bafa464ef9a58d1c0db0dd26120d4",
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
        supports_flash_attn=True,
    ),
    "qwen3_8b": EmbeddingModelSpec(
        model_id="Qwen/Qwen3-Embedding-8B",
        alias="qwen3_8b",
        default_batch_size=2,
        revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
        supports_flash_attn=True,
    ),
    "qwen3_4b": EmbeddingModelSpec(
        model_id="Qwen/Qwen3-Embedding-4B",
        alias="qwen3_4b",
        default_batch_size=4,
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
        supports_flash_attn=True,
    ),
    "qwen3_06b": EmbeddingModelSpec(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        alias="qwen3_06b",
        default_batch_size=8,
        revision="c54f2e6e80b2d7b7de06f51cec4959f6b3e03418",
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
        supports_flash_attn=True,
    ),
    "mini_l6": EmbeddingModelSpec(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        alias="mini_l6",
        default_batch_size=4096,
        revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    ),
    "mini_l12": EmbeddingModelSpec(
        model_id="sentence-transformers/all-MiniLM-L12-v2",
        alias="mini_l12",
        default_batch_size=4096,
        revision="936af83a2ecce5fe87a09109ff5cbcefe073173a",
    ),
    "mpnet": EmbeddingModelSpec(
        model_id="sentence-transformers/all-mpnet-base-v2",
        alias="mpnet",
        default_batch_size=1024,
        revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
    ),
    "distilroberta": EmbeddingModelSpec(
        model_id="sentence-transformers/all-distilroberta-v1",
        alias="distilroberta",
        default_batch_size=1024,
        revision="842eaed40bee4d61673a81c92d5689a8fed7a09f",
    ),
    "gtr_t5": EmbeddingModelSpec(
        model_id="sentence-transformers/gtr-t5-base",
        alias="gtr_t5",
        default_batch_size=1024,
        revision="9801579ce813fb37541e6098dffd17959fffcd6e",
    ),
    "gte_base": EmbeddingModelSpec(
        model_id="thenlper/gte-base",
        alias="gte_base",
        default_batch_size=1024,
        revision="c078288308d8dee004ab72c6191778064285ec0c",
    ),
    # NOTE: We exclude jina_v3 due to occasional crashes
    # "jina_v3": EmbeddingModelSpec(
    #     model_id="jinaai/jina-embeddings-v3",
    #     alias="jina_v3",
    #     default_batch_size=128,
    #     compile_model=False,
    #     trust_remote_code=True,
    #     revision="f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9",
    #     model_kwargs={"torch_dtype": torch.bfloat16, "default_task": "text-matching"},
    #     supports_flash_attn=True,
    # ),
    # NOTE: We exclude OpenAI models from the default registry due to API costs and API instability
    # "openai_v3_small": EmbeddingModelSpec(
    #     model_id="text-embedding-3-small",
    #     alias="openai_v3_small",
    #     default_batch_size=-1,  # Hack to indicate API-based model
    # ),
    # "openai_v3_large": EmbeddingModelSpec(
    #     model_id="text-embedding-3-large",
    #     alias="openai_v3_large",
    #     default_batch_size=-1,  # Hack to indicate API-based model
    # ),
}


@dataclass(frozen=True)
class PathConfig:
    """Configuration for all benchmark paths."""

    repo_root: Path
    datasets_root: Path
    embeddings_root: Path
    checkpoints_root: Path

    @classmethod
    def from_defaults(cls) -> "PathConfig":
        """Create PathConfig from default locations."""
        repo_root = get_repo_root()
        return cls(
            repo_root=repo_root,
            datasets_root=repo_root / "data" / "datasets",
            embeddings_root=repo_root / "data" / "embeddings",
            checkpoints_root=repo_root / "data" / "checkpoints",
        )


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""

    collection_name: str
    augmentation_mode: Literal["join", "union"]
    queries_file: str
    ground_truth_file: str
    paths: PathConfig = field(default_factory=PathConfig.from_defaults)

    emb_dir: Path = field(init=False)
    candidate_emb_dir: Path = field(init=False)
    query_emb_dir: Path = field(init=False)
    queries_path: Path = field(init=False)
    ground_truth_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.emb_dir = self.paths.embeddings_root / self.collection_name
        self.candidate_emb_dir = self.paths.embeddings_root / self.collection_name / "datasets"
        self.query_emb_dir = self.paths.embeddings_root / self.collection_name / "queries"
        self.queries_path = (
            self.paths.datasets_root / self.collection_name / "queries" / self.queries_file
        )
        self.ground_truth_path = (
            self.paths.datasets_root / self.collection_name / "queries" / self.ground_truth_file
        )


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "freyja": DatasetConfig(
        collection_name="freyja",
        augmentation_mode="join",
        queries_file="join_queries.csv",
        ground_truth_file="join_ground_truth.csv",
    ),
    "omnimatch_city_train": DatasetConfig(
        collection_name="omnimatch_city_train",
        augmentation_mode="join",
        queries_file="join_queries.csv",
        ground_truth_file="join_ground_truth.csv",
    ),
    "omnimatch_city_test": DatasetConfig(
        collection_name="omnimatch_city_test",
        augmentation_mode="join",
        queries_file="join_queries.csv",
        ground_truth_file="join_ground_truth.csv",
    ),
    "omnimatch_culture_train": DatasetConfig(
        collection_name="omnimatch_culture_train",
        augmentation_mode="join",
        queries_file="join_queries.csv",
        ground_truth_file="join_ground_truth.csv",
    ),
    "omnimatch_culture_test": DatasetConfig(
        collection_name="omnimatch_culture_test",
        augmentation_mode="join",
        queries_file="join_queries.csv",
        ground_truth_file="join_ground_truth.csv",
    ),
    "santos-join": DatasetConfig(
        collection_name="santos_small",
        augmentation_mode="join",
        queries_file="join_queries.csv",
        ground_truth_file="join_ground_truth.csv",
    ),
    "santos-union": DatasetConfig(
        collection_name="santos_small",
        augmentation_mode="union",
        queries_file="union_queries.csv",
        ground_truth_file="union_ground_truth.csv",
    ),
    "webtable-join": DatasetConfig(
        collection_name="webtable",
        augmentation_mode="join",
        queries_file="join_queries.csv",
        ground_truth_file="join_ground_truth.csv",
    ),
    "webtable-union": DatasetConfig(
        collection_name="webtable",
        augmentation_mode="union",
        queries_file="union_queries.csv",
        ground_truth_file="union_ground_truth.csv",
    ),
}


@dataclass(frozen=True)
class ProjectionModelSpec:
    """Specification for a projection model."""

    # Model class and configurations
    class_: Literal[
        "ContrastiveLearningModel",
        "LocalIsometryModel",
        "NaiveModel",
        "ProcrustesModel",
        "Vec2VecModel",
    ]
    module_kwargs: (
        CLModuleConfig
        | LocalIsometryModuleConfig
        | NaiveModuleConfig
        | ProcrustesModuleConfig
        | Vec2VecModuleConfig
    )
    criterion: CriterionConfig
    optimizers: list[OptimizerConfig]
    schedulers: list[LRSchedulerConfig]

    # Data module
    data_module_class: Literal["DefaultDataModule", "SingleStepTrainingDataModule"] = (
        "DefaultDataModule"
    )
    batch_size: int = 4096

    # Trainer configuration
    epochs: int = 200
    patience: int | None = None


# Model registry with all configuration
#
# Key hyperparameter choices:
# - reduced_dim=128: PCA reduction dimension for LA2M, balances compression vs. information retention
# - q=1500: Low-rank SVD approximation rank, chosen for computational efficiency on large datasets
# - num_clusters=300: K-means cluster count for LA2M local alignment, tuned on validation set
# - out_dim=1024: Common output dimension for aligned embeddings, accommodates diverse input dims
# - hidden_dims: MLP architecture depths, deeper networks for cl_default (exploratory),
#   shallower for cl_optim (tuned for efficiency)
# - temperature=0.1: NT-Xent temperature, standard value from contrastive learning literature
# - T_max=200: Cosine annealing period matching max_epochs
# - patience=20: Early stopping patience (cl_optim only), prevents overfitting
# - Vec2Vec loss weights (weight_cc=10.0): Higher weight on cycle consistency loss for stable GAN training
#
PROJECTION_MODEL_REGISTRY: dict[str, ProjectionModelSpec] = {
    # A2M: Procrustes-based alignment (single-step, no training loop)
    "a2m": ProjectionModelSpec(
        class_="ProcrustesModel",
        module_kwargs=ProcrustesModuleConfig(
            approximate=True, q=1500, with_rotation=True, use_normalization=True
        ),
        criterion=CriterionConfig(class_="MultiViewInfoNCELoss"),  # dummy
        optimizers=[],
        schedulers=[],
        data_module_class="SingleStepTrainingDataModule",
        epochs=0,
    ),
    # LA2M with PCA: Local alignment with dimensionality reduction for efficiency
    "la2m_default": ProjectionModelSpec(
        class_="LocalIsometryModel",
        module_kwargs=LocalIsometryModuleConfig(
            reduced_dim=128, approximate=True, q=1500, clustering_method="kmeans", num_clusters=300
        ),
        criterion=CriterionConfig(class_="MultiViewInfoNCELoss"),  # dummy
        optimizers=[],
        schedulers=[],
        data_module_class="SingleStepTrainingDataModule",
        epochs=0,
    ),
    # LA2M without PCA: Full-dimensional local alignment (reduced_dim=0 disables reduction)
    "la2m_nopca": ProjectionModelSpec(
        class_="LocalIsometryModel",
        module_kwargs=LocalIsometryModuleConfig(
            reduced_dim=0, approximate=True, q=1500, clustering_method="kmeans", num_clusters=300
        ),
        criterion=CriterionConfig(class_="MultiViewInfoNCELoss"),  # dummy
        optimizers=[],
        schedulers=[],
        data_module_class="SingleStepTrainingDataModule",
        epochs=0,
    ),
    # CL default: Deeper MLP architecture (3 hidden layers) for exploratory experiments
    "cl_default": ProjectionModelSpec(
        class_="ContrastiveLearningModel",
        module_kwargs=CLModuleConfig(
            out_dim=1024,
            hidden_dims=[1024, 1536, 1536],
            activation="silu",
            normalization="batch",
            dropout=0.0,
            no_val_orchestrator=True,
        ),
        criterion=CriterionConfig(class_="MultiViewNTXentLoss", kwargs={"temperature": 0.1}),
        optimizers=[OptimizerConfig(class_="AdamW", kwargs={"lr": 1e-3, "weight_decay": 0.01})],
        schedulers=[LRSchedulerConfig(class_="CosineAnnealingLR", kwargs={"T_max": 200})],
    ),
    # CL optimized: Shallower MLP (2 hidden layers) with early stopping, best performer
    "cl_optim": ProjectionModelSpec(
        class_="ContrastiveLearningModel",
        module_kwargs=CLModuleConfig(
            out_dim=1024,
            hidden_dims=[1024, 1024],
            activation="silu",
            normalization="batch",
            dropout=0.0,
            no_val_orchestrator=True,
        ),
        criterion=CriterionConfig(class_="MultiViewNTXentLoss", kwargs={"temperature": 0.1}),
        optimizers=[OptimizerConfig(class_="AdamW", kwargs={"lr": 1e-3, "weight_decay": 0.01})],
        schedulers=[LRSchedulerConfig(class_="CosineAnnealingLR", kwargs={"T_max": 200})],
        patience=20,  # Early stopping after 20 epochs without improvement
    ),
    # NOTE: This is the same as cl_optim but without early stopping (and validation overhead)
    "cl_noval": ProjectionModelSpec(
        class_="ContrastiveLearningModel",
        module_kwargs=CLModuleConfig(
            out_dim=1024,
            hidden_dims=[1024, 1024],
            activation="silu",
            normalization="batch",
            dropout=0.0,
            no_val_orchestrator=True,
        ),
        criterion=CriterionConfig(class_="MultiViewNTXentLoss", kwargs={"temperature": 0.1}),
        optimizers=[OptimizerConfig(class_="AdamW", kwargs={"lr": 1e-3, "weight_decay": 0.01})],
        schedulers=[LRSchedulerConfig(class_="CosineAnnealingLR", kwargs={"T_max": 200})],
        patience=None,
    ),
    # Vec2Vec: GAN-based approach with discriminators and translators between embedding spaces
    "v2v": ProjectionModelSpec(
        class_="Vec2VecModel",
        module_kwargs=Vec2VecModuleConfig(
            disc_dim=1024,
            translator_dim=1024,
            latent_transform_dim=1024,
            latent_dim=1024,
            disc_depth=5,
            translator_depth=3,
            latent_transform_depth=4,
            weight_init="kaiming",
            norm_style="batch",
        ),
        criterion=CriterionConfig(
            class_="Vec2VecLosses",
            kwargs={"weight_rec": 1.0, "weight_vsp": 1.0, "weight_cc": 10.0},
        ),
        optimizers=[
            OptimizerConfig(
                class_="Adam", kwargs={"lr": 2e-5, "fused": False, "betas": (0.5, 0.999)}
            ),
            OptimizerConfig(
                class_="Adam", kwargs={"lr": 1e-5, "eps": 6.25e-10, "betas": (0.5, 0.999)}
            ),
        ],
        schedulers=[LRSchedulerConfig(class_="CosineAnnealingLR", kwargs={"T_max": 200})],
    ),
    # Naive baseline: Zero-padding to match largest embedding dimension (union of dimensions)
    "union_plus": ProjectionModelSpec(
        class_="NaiveModel",
        module_kwargs=NaiveModuleConfig(mode="pad"),
        criterion=CriterionConfig(class_="MultiViewInfoNCELoss"),  # dummy
        optimizers=[],
        schedulers=[],
        data_module_class="SingleStepTrainingDataModule",
        epochs=0,
    ),
    # Naive baseline: Truncation to smallest embedding dimension (intersection of dimensions)
    "union_minus": ProjectionModelSpec(
        class_="NaiveModel",
        module_kwargs=NaiveModuleConfig(mode="truncate"),
        criterion=CriterionConfig(class_="MultiViewInfoNCELoss"),  # dummy
        optimizers=[],
        schedulers=[],
        data_module_class="SingleStepTrainingDataModule",
        epochs=0,
    ),
}

# File and directory constants
CHECKPOINT_FILENAME = "last.ckpt"
EMBEDDING_SUFFIX = ".fa"
EMBEDDINGS_NPY = "embeddings.npy"

# K-values for evaluation
DEFAULT_K_VALS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Directory suffixes
ALIGNED_SUBDIR = "aligned"
ALIGNED_PER_VIEW_SUBDIR = "aligned_per_view"
TRAINSET_SUBDIR = "train-set"
VAL_SPLIT_SUBDIR = "split/val"
TEST_SPLIT_SUBDIR = "split/test"

# Get available datasets from file system
AVAILABLE_DATASETS = os.listdir(PathConfig.from_defaults().embeddings_root)  # noqa: PTH208

# Embedding dimensions (used to sort views by output dimensionality)
EMBEDDING_DIMS: dict[str, int] = {
    "kalm_gemma3_12b": 3840,
    "qwen3_8b": 4096,
    "llama_nemotron_8b": 4096,
    "qwen3_4b": 2560,
    "qwen3_06b": 1024,
    "kalm_instruct_v2_5": 896,
    "gtr_t5": 768,
    "gte_base": 768,
    "mpnet": 768,
    "distilroberta": 768,
    "mini_l12": 384,
    "mini_l6": 384,
}
