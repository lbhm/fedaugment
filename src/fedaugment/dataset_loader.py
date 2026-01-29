import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import ClassVar, Literal

import polars as pl
from loguru import logger
from tqdm.auto import tqdm

from fedaugment.types import StrPath
from fedaugment.utils import sanitize_col_name

INFER_SCHEMA_LENGTH = 10000


class DatasetLoader:
    FILE_TYPES: ClassVar[set[str]] = {".csv", ".tsv", ".parquet", ".pq"}
    DEFAULT_BATCH_SIZE: ClassVar[int] = 100_000

    def __init__(
        self,
        input_dir: StrPath,
        batch_size: int | None = None,
        column_types: Literal["all", "string"] = "all",
        max_datasets: int | None = None,
        n_workers: int | None = None,
    ) -> None:
        """Create a DatasetLoader to load and iterate over datasets.

        The DatasetLoader is lazy and does not load datasets until it is iterated over.

        Args:
            input_dir: Path to a folder containing a table collection.
            batch_size: Number of datasets to process in each batch.
                If None (default), all datasets are loaded at once.
            column_types: Specify which column types to load from the tables. Options are:
                - "all": Load all column types (default).
                - "string": Only load string and categorical columns.
            max_datasets: Maximum number of datasets to load. By default, all datasets are loaded.
            n_workers: Number of worker threads to use for loading datasets.
                If None (default), uses the number of CPU cores.
        """
        # Configuration used during lazy loading
        self.column_types: Literal["all", "string"] = column_types
        self.batch_size = batch_size
        self.n_workers = n_workers

        # Discovered dataset file paths
        self.input_dir = Path(input_dir)
        logger.info("Initializing dataset loader with datasets from {}...", input_dir)
        start = time.time()

        # Discover files and sort them deterministically by file.name
        self.dataset_files = sorted(
            chain.from_iterable(self.input_dir.glob(f"*{ext}") for ext in self.FILE_TYPES),
            key=lambda f: f.name,
        )
        if max_datasets is not None:
            self.dataset_files = self.dataset_files[:max_datasets]

        self._validate_files_and_batch_size(self.batch_size, len(self.dataset_files))

        logger.success(
            "Discovered {} file paths for lazy loading in {:.4f} seconds.",
            len(self.dataset_files),
            time.time() - start,
        )

    def __iter__(self) -> Iterator[list[tuple[str, pl.DataFrame]]]:
        """Iterate over the datasets, yielding batches of (table_name, DataFrame) tuples."""
        if self.batch_size is None:
            # Load all files at once
            logger.info("Loading datasets...")
            yield list(self._process_files(self.dataset_files, n_threads=self.n_workers))
        else:
            # Load files in batches
            batch_count = 0
            batch_size_sum_mb = 0.0
            for i in range(0, len(self.dataset_files), self.batch_size):
                logger.info("Loading batch {}...", batch_count + 1)
                batch_files = self.dataset_files[i : i + self.batch_size]
                datasets = list(self._process_files(batch_files, n_threads=self.n_workers))

                if datasets:
                    batch_size_sum_mb += sum(df.estimated_size() / (1024**2) for _, df in datasets)
                    batch_count += 1
                    avg_batch_size_mb = batch_size_sum_mb / batch_count

                    logger.debug("Average dataset batch size: {:.2f} MB", avg_batch_size_mb)

                yield datasets

    def __len__(self) -> int:
        """Return the total number of discovered dataset files."""
        return len(self.dataset_files)

    def _validate_files_and_batch_size(self, batch_size: int | None, total_items: int) -> None:
        """Validate that the chosen batch size and the discovered file count.

        Args:
            batch_size: The batch size to validate.
            total_items: Total number of items to process.
        """
        if total_items <= 0:
            raise ValueError("Did not discover any files at the given path")

        if batch_size is None:
            large_file_count = 1_000_000
            if total_items > large_file_count:
                logger.warning(
                    "Discovered {} files, which may be too large to load all at once. "
                    "Consider setting a batch size to not load all files at once.",
                    total_items,
                )
            return

        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        if batch_size > total_items:
            logger.warning(
                "Batch size ({}) is larger than total items ({}). "
                "Consider using a smaller batch size or disabling batching.",
                batch_size,
                total_items,
            )

    def _process_files(
        self, files: list[Path], n_threads: int | None = None
    ) -> Iterator[tuple[str, pl.DataFrame]]:
        """Process a collection of files and yield their names and DataFrames."""
        with (
            tqdm(
                desc="Loading tables",
                total=len(files),
                leave=False,
                unit="file",
                mininterval=1.0,
                dynamic_ncols=True,
            ) as pbar,
            ThreadPoolExecutor(max_workers=n_threads) as executor,
        ):
            futures = {
                executor.submit(load_dataframe, file, self.column_types): idx
                for idx, file in enumerate(files)
            }
            for future in futures:
                future.add_done_callback(lambda _: pbar.update(1))

            results: list[tuple[int, tuple[str, pl.DataFrame]]] = []
            for future in as_completed(futures):
                idx = futures[future]
                table_name, df = future.result()
                if df is not None:
                    results.append((idx, (table_name, df)))

            for _, item in sorted(results, key=itemgetter(0)):
                yield item


def load_dataframe(
    file: Path, column_types: Literal["all", "string"]
) -> tuple[str, pl.DataFrame | None]:
    """Process a single file and return its name and DataFrame.

    Args:
        file: Path to the file to process.
        column_types: Specify which column types to load from the table.

    Returns:
        Tuple of (table_name, DataFrame or None if processing failed).
    """
    try:
        if file.suffix.lower() == ".csv":
            lf = pl.scan_csv(
                file,
                null_values=["--"],
                ignore_errors=True,
                infer_schema_length=INFER_SCHEMA_LENGTH,
            )
        elif file.suffix.lower() == ".tsv":
            lf = pl.scan_csv(
                file,
                separator="\t",
                null_values=["--"],
                ignore_errors=True,
                infer_schema_length=INFER_SCHEMA_LENGTH,
            )
        else:
            lf = pl.scan_parquet(file)

        # Rename columns that cause issues with Polars
        lf = lf.rename(sanitize_col_name)

        if column_types == "string":
            # Collect only string and categorical columns
            lf = lf.select(pl.selectors.string(include_categorical=True))

        table = lf.collect(engine="streaming")
    except Exception as e:  # noqa: BLE001
        logger.warning("Error processing file {}: {}", file, e)
        return file.name, None
    else:
        if table.is_empty():
            logger.debug("The table in {} has no columns (after filtering). Skipping.", file)
            return file.name, None
        return file.name, table
