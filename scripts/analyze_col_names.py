import json
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm import tqdm


def normalize_colname(name: str) -> str:
    """Normalize column names.

    Applies:
    - lowercase
    - trim whitespace
    - collapse internal whitespace into underscores
    """
    name = name.strip().lower()
    return "_".join(name.split())


def chunked[T](items: Sequence[T], chunk_size: int) -> Iterable[list[T]]:
    """Yield consecutive chunks of size `chunk_size`."""
    for i in range(0, len(items), chunk_size):
        yield list(items[i : i + chunk_size])


def process_parquet_batch(file_paths: list[str]) -> Counter[str]:
    """Process a batch of parquet files and return a Counter of normalized column names."""
    counter: Counter[str] = Counter()

    for file_path in file_paths:
        try:
            lf = pl.scan_parquet(file_path)
            schema = lf.collect_schema()
            counter.update(normalize_colname(col) for col in schema.names())
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to process {file_path}: {e}")

    return counter


def merge_counters[T](counters: Iterable[Counter[T]]) -> Counter[T]:
    """Merge multiple Counter objects into one."""
    total: Counter[T] = Counter()
    for c in counters:
        total.update(c)
    return total


def main(
    parquet_dir: str,
    out_path: str,
    batch_size: int = 32,
    max_workers: int | None = None,
    n_common: int = 10,
) -> None:
    """Entry point for parallel parquet schema profiling."""
    root = Path(parquet_dir)
    files: list[str] = [str(p) for p in root.rglob("*.pq")]

    if not files:
        print(f"No parquet files found in {parquet_dir}")
        return

    batches = list(chunked(files, batch_size))
    counters: list[Counter[str]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_parquet_batch, batch) for batch in batches]
        counters.extend(
            fut.result()
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Processing parquet files"
            )
        )

    total_counter = merge_counters(counters)

    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(total_counter, f, indent=2)

    print("\nColumn name distribution:\n")
    for col, freq in total_counter.most_common(n_common):
        print(f"{col:40s} {freq}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Parquet column profiler")
    parser.add_argument("parquet_dir", type=str, help="Directory containing parquet files")
    parser.add_argument("out_path", type=str, help="Output path for column name distribution JSON")
    parser.add_argument("--batch-size", type=int, default=128, help="Files per worker task")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")

    args = parser.parse_args()

    main(args.parquet_dir, args.out_path, args.batch_size, args.workers)
