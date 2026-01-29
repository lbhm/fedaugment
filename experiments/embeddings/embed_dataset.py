"""This script embeds a dataset collection using a variety of embedding models."""

import argparse
from pathlib import Path

import torch
from loguru import logger

from fedaugment.embeddings import PipelineComposer
from fedaugment.embeddings.collate_functions import str_cat
from fedaugment.embeddings.strategies import (
    ColumnEmbeddingStrategy,
    DeepJoinStrategy,
    PromptStrategy,
    RankedColumnEmbeddingStrategy,
)
from fedaugment.utils import experiment_setup

from ..config import EMBEDDING_MODEL_REGISTRY


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
        help="Embedding models to run; default runs all available models",
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
        help="Batch size for lazy loading; if None, uses automatic batch size selection",
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
    return parser.parse_args()


if __name__ == "__main__":
    experiment_setup("INFO")
    args = parse_args()

    selected_models = (
        tuple(args.models) if args.models is not None else tuple(EMBEDDING_MODEL_REGISTRY)
    )
    logger.info("Using embedding models: {}", ", ".join(selected_models))

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

    # Process one model at a time to avoid OOM
    for model_name in selected_models:
        logger.info("Processing model: {}", model_name)
        model = EMBEDDING_MODEL_REGISTRY[model_name].build(
            cache_folder=args.cache_folder,
            flash_attn=not args.no_flash_attn,
            encoder_batch_size=args.encoder_batch_size,
        )

        pipelines = PipelineComposer([model], [strategy])
        pipelines.register_datasets(args.data_path, batch_size=args.batch_size)

        logger.info("Starting embedding generation for {}...", model_name)
        pipelines.generate_embeddings(args.output_path)

        # Clean up model from GPU memory before loading next
        del pipelines
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
