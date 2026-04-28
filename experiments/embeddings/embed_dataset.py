"""This script embeds a dataset collection using a variety of embedding models."""

import argparse
from pathlib import Path
from typing import Literal

import torch
from loguru import logger

from fedaugment.embeddings import PipelineComposer
from fedaugment.embeddings.collate_functions import str_cat
from fedaugment.embeddings.models import EmbeddingModel, SentenceTransformerModel
from fedaugment.embeddings.strategies import (
    ColumnEmbeddingStrategy,
    DeepJoinStrategy,
    PromptStrategy,
    RankedColumnEmbeddingStrategy,
)
from fedaugment.utils import experiment_setup

from ..config import EMBEDDING_MODEL_REGISTRY

LOCAL_CHECKPOINT_DEFAULT_ENCODER_BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed a dataset collection with various models.")
    parser.add_argument(
        "--data-path", type=Path, required=True, help="Path to the input dataset directory"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory where generated embeddings will be stored",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(EMBEDDING_MODEL_REGISTRY),
        default=None,
        help="Embedding models to run",
    )
    parser.add_argument(
        "--local-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help="Local model checkpoints to run",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default=None,
        help="HuggingFace cache folder (default: uses HF_HOME env or HF default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for lazy loading (if None, uses automatic batch size selection)",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention (for GPUs that don't support it)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["dj_adapted", "dj_original", "col_str_cat", "bm25"],
        default="dj_adapted",
        help="Serialization strategy for column embedding",
    )
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=None,
        help="Encoder batch size (overrides model-specific defaults)",
    )
    parser.add_argument(
        "--column-types",
        choices=["all", "string"],
        default="all",
        help="Which column types to load from tables",
    )
    return parser.parse_args()


def run_model(
    model_name: str,
    model: EmbeddingModel,
    strategy: PromptStrategy,
    data_path: Path,
    output_path: Path,
    batch_size: int | None,
    column_types: Literal["all", "string"],
) -> None:
    logger.info("Processing model {} for {}", model_name, data_path)
    pipelines = PipelineComposer([model], [strategy])
    pipelines.register_datasets(data_path, batch_size=batch_size, column_types=column_types)

    logger.info("Starting embedding generation for {}...", model_name)
    pipelines.generate_embeddings(output_path)

    # Clean up model from GPU memory before loading next
    del pipelines
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    experiment_setup("INFO")
    args = parse_args()

    # Define prompt strategy based on CLI argument
    logger.info("Creating prompt strategy: {}", args.strategy)
    strategy: PromptStrategy
    match args.strategy:
        case "dj_adapted":
            strategy = DeepJoinStrategy("dj_adpt", max_length=8192 * 4, prompt_mode="adapted")
        case "dj_original":
            strategy = DeepJoinStrategy("dj_orig", max_length=8192 * 4, prompt_mode="original")
        case "col_str_cat":
            strategy = ColumnEmbeddingStrategy("col", str_cat, {"sep": ", ", "max_length": 8192})
        case "bm25":
            strategy = RankedColumnEmbeddingStrategy(
                "bm25", "bm25", {"sep": ", ", "max_length": 8192}
            )
        case _:
            raise ValueError(f"Unknown strategy: {args.strategy}")

    models: dict[str, EmbeddingModel] = {}
    if args.models is not None:
        logger.info("Using registry embedding models: {}", ", ".join(args.models))
        for model_name in args.models:
            model = EMBEDDING_MODEL_REGISTRY[model_name].build(
                cache_folder=args.cache_folder,
                flash_attn=not args.no_flash_attn,
                encoder_batch_size=args.encoder_batch_size,
            )

            # Process one model at a time to avoid OOM
            run_model(
                model_name=model_name,
                model=model,
                strategy=strategy,
                data_path=args.data_path,
                output_path=args.output_path,
                batch_size=args.batch_size,
                column_types=args.column_types,
            )

    if args.local_checkpoints is not None:
        for checkpoint_path in args.local_checkpoints:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Local checkpoint not found at {checkpoint_path}")

            checkpoint_alias = f"local_{checkpoint_path.name}"
            logger.info(
                "Adding local checkpoint model {} from {}", checkpoint_alias, checkpoint_path
            )
            # Registry-backed models already have batch-size defaults.
            # Local checkpoints need an explicit fallback to avoid passing None to encode().
            models[checkpoint_alias] = SentenceTransformerModel(
                model_id=str(checkpoint_path),
                alias=checkpoint_alias,
                compile_model=False,
                encoder_batch_size=(
                    args.encoder_batch_size or LOCAL_CHECKPOINT_DEFAULT_ENCODER_BATCH_SIZE
                ),
                cache_folder=args.cache_folder,
            )

            run_model(
                model_name=checkpoint_alias,
                model=models[checkpoint_alias],
                strategy=strategy,
                data_path=args.data_path,
                output_path=args.output_path,
                batch_size=args.batch_size,
                column_types=args.column_types,
            )
