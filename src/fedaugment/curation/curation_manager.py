import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch_pca import PCA

from fedaugment.curation.curators import Curator
from fedaugment.dataset_loader import DatasetLoader, load_dataframe
from fedaugment.embeddings.models import EmbeddingModel
from fedaugment.embeddings.strategies import PromptStrategy
from fedaugment.types import ColIdArray, StrPath


class CurationManager:
    """Manager for curating training data using various curators."""

    def __init__(self) -> None:
        self.embeddings: Tensor | None = None
        self.embedding_ids: ColIdArray | None = None
        self.pipeline_name: str | None = None
        self.dataset_loader: DatasetLoader | None = None

    @torch.no_grad()
    def generate_proxy_embeddings(
        self,
        input_dir: StrPath,
        embedding_model: EmbeddingModel,
        prompt_strategy: PromptStrategy,
        batch_size: int | None = None,
        column_types: Literal["all", "string"] = "all",
        max_datasets: int | None = None,
        truncate_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        """Load datasets and generate proxy embeddings used for data curation.

        Args:
            input_dir: Path to a folder containing a table collection.
            embedding_model: The embedding model to use for generating embeddings.
            prompt_strategy: The prompt strategy to use for generating embeddings.
            batch_size: Number of datasets to process in each batch.
                If None (default), all datasets are loaded at once.
            column_types: Specify which column types to load from the tables. Options are:
                - "all": Load all column types (default).
                - "string": Only load string and categorical columns.
            max_datasets: Maximum number of datasets to load. By default, all datasets are loaded.
            truncate_dim: If specified, truncate embeddings to this dimensionality.
            dtype: Data type for the generated embeddings.
            device: Device to place the generated embeddings on.
        """
        start = time.time()
        device = torch.device(device)
        self.dataset_loader = DatasetLoader(
            input_dir=input_dir,
            batch_size=batch_size,
            column_types=column_types,
            max_datasets=max_datasets,
        )
        self.pipeline_name = f"{embedding_model.alias}-{prompt_strategy.alias}"

        batch_size = self.dataset_loader.batch_size or len(self.dataset_loader)
        num_batches = (len(self.dataset_loader) + batch_size - 1) // batch_size

        logger.info(
            "Embedding {} datasets in {} batch(es) of up to {} datasets each.",
            len(self.dataset_loader),
            num_batches,
            batch_size,
        )
        embedding_count = 0
        embedding_list: list[Tensor] = []
        embedding_id_list: list[ColIdArray] = []
        for i, batch in enumerate(self.dataset_loader):
            if num_batches > 1:
                logger.info("Processing batch {}/{}...", i + 1, num_batches)
            else:
                logger.info("Processing data...")

            prompt_gen_start = time.time()
            prompt_ids, prompts = prompt_strategy.generate_prompts(batch, len(batch))
            logger.info(
                "Generated {} prompts in {:.4f} seconds.",
                len(prompts),
                time.time() - prompt_gen_start,
            )

            embedding_start = time.time()
            prompt_ids, embeddings = embedding_model.embed(prompt_ids, prompts, truncate_dim)
            embedding_count += len(prompt_ids)
            logger.info(
                "Embedded {} prompts in {:.4f} seconds.",
                len(prompt_ids),
                time.time() - embedding_start,
            )

            post_gen_start = time.time()
            embedding_ids, embeddings = prompt_strategy.process_embeddings(prompt_ids, embeddings)
            logger.info(
                "Post-processed embeddings in {:.4f} seconds.", time.time() - post_gen_start
            )

            embedding_list.append(torch.from_numpy(embeddings).to(dtype=dtype, device=device))
            embedding_id_list.append(embedding_ids)

        self.embeddings = torch.cat(embedding_list, dim=0)
        self.embedding_ids = np.concatenate(embedding_id_list, axis=0)
        logger.debug("Generated embeddings tensor shape: {}", self.embeddings.shape)
        logger.debug("Generated embedding IDs array shape: {}", self.embedding_ids.shape)

        logger.success(
            "Embedded {} prompts based on {} datasets in {:.4f} seconds.",
            embedding_count,
            len(self.dataset_loader),
            time.time() - start,
        )

    @torch.no_grad()
    def curate_datasets(
        self,
        curators: Sequence[Curator],
        output_dir: StrPath,
        k: float,
        seed: int,
        pca_dim: float | None = None,
    ) -> None:
        """Curate datasets using the specified curators and save the results.

        Args:
            curators: List of `Curator` instances to use for curation.
            output_dir: Path to a folder where curated datasets will be saved.
            k: Number or proportion of datasets to select during curation.
                * If int, the exact number of datasets to select.
                * If float (should be between 0.0 and 1.0), the proportion of datasets to select.
            seed: Random seed for reproducibility. For some curators, the seed is the first
                selected index.
            pca_dim: If specified, apply PCA to reduce embeddings to this dimensionality before
                curation.
                * If int, number of components to keep.
                * If float (should be between 0.0 and 1.0), the number of components to keep is
                determined by the cumulative percentage of variance explained by the components
                until the proportion is reached.
        """
        if self.embeddings is None or self.embedding_ids is None or self.pipeline_name is None:
            raise RuntimeError(
                "Proxy embeddings have not been generated yet. "
                "Please run generate_proxy_embeddings() first."
            )

        start = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if k <= 0:
            raise ValueError("k must be positive")
        if isinstance(k, float):
            if k > 1.0:
                raise ValueError("If k is a float, it must be between 0.0 and 1.0")
            k = int(k * len(self.embeddings))

        # Optionally apply PCA to reduce dimensionality before data curation
        if pca_dim is not None:
            pca_start = time.time()
            pca_model = PCA(n_components=pca_dim, svd_solver="covariance_eigh")
            embeddings = pca_model.fit_transform(self.embeddings)
            logger.info(
                "Applied PCA to reduce embeddings from {} to {} dimensions in {:.4f} seconds.",
                self.embeddings.shape[1],
                embeddings.shape[1],
                time.time() - pca_start,
            )
            if pca_model.explained_variance_ratio_ is not None:
                logger.info(
                    "Selected components explain {:.4f}% of variance.",
                    pca_model.explained_variance_ratio_.sum() * 100,
                )
                logger.debug(
                    "Explained variance ratio of selected components: {}",
                    pca_model.explained_variance_ratio_,
                )
        else:
            embeddings = self.embeddings

        logger.info(
            "Selecting {}/{} data points using {} curator(s)...", k, len(embeddings), len(curators)
        )
        for curator in curators:
            curation_start = time.time()
            logger.info("Curating data using '{}'...", curator.alias)
            selected_indices = curator.curate(embeddings, k, seed)
            logger.info(
                "Curator '{}' selected {} data points in {:.4f} seconds.",
                curator.alias,
                len(selected_indices),
                time.time() - curation_start,
            )

            curation_config: list[str] = [curator.alias, self.pipeline_name, f"k={k}"]
            if pca_dim is not None:
                curation_config.append(f"pca={pca_dim}")
            curation_dir = output_dir / "-".join(curation_config)

            logger.info("Saving curated columns to '{}'", curation_dir)
            curation_dir.mkdir(parents=True, exist_ok=True)
            self._copy_selected_columns(self.embedding_ids, selected_indices, curation_dir)

        logger.success("Curation completed in {:.4f} seconds.", time.time() - start)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Move the embeddings to the specified device.

        Args:
            device: Device to move the embeddings to.
            dtype: If specified, convert the embeddings to this data type.
        """
        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(device=device, dtype=dtype)

    def _copy_selected_columns(
        self, col_ids: ColIdArray, selected_indices: Tensor, output_dir: Path
    ) -> None:
        """Copy the selected columns to the output directory.

        Args:
            col_ids: Array of column identifiers in the format `<file_name>::<column>`.
            selected_indices: Indices of the selected datasets.
            output_dir: Path to the output directory.
        """
        if self.dataset_loader is None:
            raise RuntimeError(
                "Dataset loader is not initialized. Did you run generate_proxy_embeddings()?"
            )

        if selected_indices.numel() == 0:
            logger.warning("No columns selected; skipping copy step.")
            return

        selected_ids = col_ids[selected_indices.detach().cpu().numpy()]
        if selected_ids.size == 0:
            logger.warning("Selected indices produced an empty column list; nothing to copy.")
            return

        # Group columns by their source table
        grouped_columns: dict[str, list[str]] = defaultdict(list)
        table_ids, _, col_ids = np.strings.partition(
            selected_ids, np.asarray("::", dtype=np.dtypes.StringDType())
        )
        for table_id, col_id in zip(table_ids.tolist(), col_ids.tolist(), strict=True):
            grouped_columns[table_id].append(col_id)

        # Project and copy columns
        dataset_paths = {path.name: path for path in self.dataset_loader.dataset_files}
        copied_columns = 0
        for file_name, columns in grouped_columns.items():
            dataset_path = dataset_paths.get(file_name)
            if dataset_path is None:
                logger.error("File '{}' not found among loaded files; skipping.", file_name)
                continue

            table_name, dataframe = load_dataframe(dataset_path, self.dataset_loader.column_types)
            if dataframe is None:
                logger.error("Failed to load dataset '{}' for projection; skipping.", table_name)
                continue

            projection = dataframe.select(columns)
            projection.write_parquet(output_dir / Path(table_name).with_suffix(".pq"))
            copied_columns += len(columns)

        logger.info("Copied {} column projections.", copied_columns)
