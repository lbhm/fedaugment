import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import orjson
import polars as pl
import psutil
from loguru import logger

from fedaugment.dataset_loader import DatasetLoader
from fedaugment.embeddings.models import EmbeddingModel
from fedaugment.embeddings.strategies import PromptStrategy
from fedaugment.types import FloatArray, StrPath


class PipelineComposer:
    HIGH_MEMORY_USAGE_PERCENT: ClassVar[int] = 85  # Warn when memory usage exceeds this

    def __init__(
        self,
        embedding_models: Sequence[EmbeddingModel],
        prompt_strategies: Sequence[PromptStrategy],
    ) -> None:
        """Create the cross-product of pipelines from embedding models and prompt strategies.

        Args:
            embedding_models: A list of embedding models to be used in the pipelines.
            prompt_strategies: A list of prompt strategies to be used in the pipelines.
        """
        self.models = embedding_models
        self.strategies = prompt_strategies
        self.dataset_loader: DatasetLoader | None = None

    @property
    def max_seq_lengths(self) -> dict[str, int]:
        """Get the maximum sequence lengths for each model in the pipelines."""
        return {model.alias: model.max_seq_length for model in self.models}

    def register_datasets(
        self,
        input_dir: StrPath,
        batch_size: int | None = None,
        column_types: Literal["all", "string"] = "all",
        max_datasets: int | None = None,
    ) -> None:
        """Register datasets from a directory as input for embedding pipelines.

        Args:
            input_dir: Path to a folder containing a table collection.
            batch_size: Number of datasets to process in each batch.
                If None (default), all datasets are loaded at once.
            column_types: Specify which column types to load from the tables. Options are:
                - "all": Load all column types (default).
                - "string": Only load string and categorical columns.
            max_datasets: Maximum number of datasets to load. By default, all datasets are loaded.
        """
        self.dataset_loader = DatasetLoader(
            input_dir=input_dir,
            batch_size=batch_size,
            column_types=column_types,
            max_datasets=max_datasets,
        )

    def generate_embeddings(self, output_dir: StrPath) -> None:
        """Generate embeddings for a table collection using the composed embedding pipelines.

        Args:
            output_dir: Path to a folder where embedding files will be created per pipeline.
        """
        if self.dataset_loader is None:
            raise RuntimeError(
                "No datasets loaded. Please call register_datasets() before generating embeddings."
            )

        start = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = self.dataset_loader.batch_size or len(self.dataset_loader)
        num_batches = (len(self.dataset_loader) + batch_size - 1) // batch_size

        logger.info(
            "Embedding {} datasets in {} batch(es) of up to {} datasets each.",
            len(self.dataset_loader),
            num_batches,
            batch_size,
        )

        total_embedding_count = 0
        for i, batch in enumerate(self.dataset_loader):
            logger.info("Processing batch {}/{}...", i + 1, num_batches)

            # Process this batch with all strategies and models
            batch_embedding_count = self._generate_embedding_batch(
                batch, output_dir, i, num_batches
            )
            total_embedding_count += batch_embedding_count

        logger.success(
            "Embedded {} prompts across {} pipelines based on {} datasets in {:.4f} seconds.",
            total_embedding_count,
            len(self.models) * len(self.strategies),
            len(self.dataset_loader),
            time.time() - start,
        )

    def count_tokens(self) -> dict[str, list[int]]:
        """Count the number of tokens in a table collection for the composed embedding pipelines.

        Returns:
            A dictionary with the token count per model invocation for each pipeline.
        """
        if self.dataset_loader is None:
            raise RuntimeError(
                "No datasets loaded. Please call register_datasets() before counting tokens."
            )

        start = time.time()
        token_counts: dict[str, list[int]] = defaultdict(list)

        batch_size = self.dataset_loader.batch_size or len(self.dataset_loader)
        num_batches = (len(self.dataset_loader) + batch_size - 1) // batch_size

        logger.info(
            "Counting tokens for {} datasets in {} batch(es) of up to {} datasets each.",
            len(self.dataset_loader),
            num_batches,
            batch_size,
        )

        for i, batch in enumerate(self.dataset_loader):
            logger.info("Processing batch {}/{}...", i + 1, num_batches)

            for strategy in self.strategies:
                logger.info("Generating prompts with strategy {}...", strategy.alias)
                prompt_gen_start = time.time()
                _, prompts = strategy.generate_prompts(batch, len(batch))
                self._log_prompts_generated(len(prompts), time.time() - prompt_gen_start)

                for model in self.models:
                    pipeline_name = f"{model.alias}-{strategy.alias}"
                    logger.info("Counting tokens for pipeline {}...", pipeline_name)
                    tokenization_start = time.time()
                    token_count = model.count_tokens(prompts, per_prompt=True)
                    token_counts[pipeline_name].extend(token_count)
                    logger.info(
                        "Counted tokens in {:.4f} seconds.", time.time() - tokenization_start
                    )

        logger.success(
            "Counted tokens for {} pipelines in {:.4f} seconds.",
            len(self.models) * len(self.strategies),
            time.time() - start,
        )
        return token_counts

    def _generate_embedding_batch(
        self,
        batch: list[tuple[str, pl.DataFrame]],
        output_dir: Path,
        batch_idx: int,
        num_batches: int,
    ) -> int:
        """Generate embeddings for a batch of datasets.

        Args:
            batch: List of (table_name, DataFrame) tuples for this batch.
            output_dir: Path to output directory.
            batch_idx: Current batch index (0-based).
            num_batches: Total number of batches.

        Returns:
            Number of embeddings generated in this batch.
        """
        embedding_count = 0

        for strategy in self.strategies:
            logger.info("Generating prompts with strategy {}...", strategy.alias)
            prompt_gen_start = time.time()
            prompt_ids, prompts = strategy.generate_prompts(batch, len(batch))
            self._log_prompts_generated(len(prompts), time.time() - prompt_gen_start)

            for model in self.models:
                pipeline_name = f"{model.alias}-{strategy.alias}"
                output_path = output_dir / f"{pipeline_name}.fa"

                # For first batch, create new file; for subsequent batches, append
                if batch_idx == 0:
                    if output_path.exists():
                        logger.warning(
                            "{}.fa already exists in {}. Skipping.", pipeline_name, output_dir
                        )
                        continue
                    output_path.mkdir(parents=True, exist_ok=True)
                elif not output_path.exists():
                    logger.error(
                        "{}.fa does not exist. Cannot append batch {}. Skipping.",
                        pipeline_name,
                        batch_idx + 1,
                    )
                    continue

                logger.info(
                    "Embedding prompts with pipeline {} (batch {}/{})...",
                    pipeline_name,
                    batch_idx + 1,
                    num_batches,
                )
                embedding_start = time.time()
                embedding_ids, embeddings = model.embed(prompt_ids, prompts)
                embedding_count += len(embedding_ids)
                logger.info(
                    "Embedded {} prompts in {:.4f} seconds.",
                    len(embedding_ids),
                    time.time() - embedding_start,
                )

                logger.info("Post-processing embeddings for pipeline {}...", pipeline_name)
                post_gen_start = time.time()
                col_ids, col_embeddings = strategy.process_embeddings(embedding_ids, embeddings)
                logger.info(
                    "Post-processed embeddings in {:.4f} seconds.", time.time() - post_gen_start
                )

                # Save or append data in FedAugment style NPY format
                emb_path = output_path / "embeddings.npy"
                json_path = output_path / "metadata.json"
                col_id_path = output_path / "column_ids.npy"

                if batch_idx == 0:
                    # Create new NPY files
                    np.save(emb_path, col_embeddings)
                    np.save(col_id_path, col_ids)
                    logger.info("Created new NPY file: {}", emb_path)

                    # Save metadata
                    metadata = {
                        "pipeline": pipeline_name,
                        "embedding_shape": list(col_embeddings.shape),
                        "embedding_dtype": str(col_embeddings.dtype),
                    }
                    Path(json_path).write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
                else:
                    # Append to existing NPY file
                    existing_embeddings: FloatArray = np.load(emb_path)
                    combined_embeddings = np.concatenate(
                        [existing_embeddings, col_embeddings], axis=0
                    )
                    np.save(emb_path, combined_embeddings)

                    # Append column IDs
                    existing_col_ids = np.load(col_id_path, allow_pickle=True)
                    combined_col_ids = np.concatenate([existing_col_ids, col_ids])
                    np.save(col_id_path, combined_col_ids)

                    # Update metadata
                    metadata = {
                        "pipeline": pipeline_name,
                        "embedding_shape": list(combined_embeddings.shape),
                        "embedding_dtype": str(combined_embeddings.dtype),
                    }
                    Path(json_path).write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

                    logger.info(
                        "Appended {} embeddings to {} (total: {})",
                        col_embeddings.shape[0],
                        emb_path,
                        combined_embeddings.shape[0],
                    )

        return embedding_count

    def _log_prompts_generated(self, n_prompts: int, elapsed: float, batch_info: str = "") -> None:
        """Log the number of generated prompts and the time taken.

        Args:
            n_prompts: Number of prompts generated.
            elapsed: Time elapsed in seconds.
            batch_info: Optional batch information to include in the log message.
        """
        mem = psutil.virtual_memory()
        batch_str = f" {batch_info}" if batch_info else ""
        logger.info("Generated {} prompts{} in {:.4f} seconds.", n_prompts, batch_str, elapsed)
        logger.debug(
            "Available memory: {:.2f} GB (memory usage: {}%)",
            mem.available / (1024**3),
            mem.percent,
        )

        # Warn if memory usage is high
        if mem.percent > self.HIGH_MEMORY_USAGE_PERCENT:
            logger.warning(
                "High memory usage detected ({}%). Consider using a smaller batch_size.",
                mem.percent,
            )
