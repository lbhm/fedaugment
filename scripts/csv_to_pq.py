"""Convert CSV files to Parquet format using multiprocessing and Polars."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

MIN_ROW_COUNT = 1
INFER_SCHEMA_LENGTH = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet using multiprocessing and Polars."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Root directory containing input CSV files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory to store converted Parquet files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Parquet files (default: %(default)s)",
    )

    return parser.parse_args()


def find_csv_files(root_dir: Path) -> list[Path]:
    """Recursively find all .csv files under root_dir."""
    return list(root_dir.rglob("*.csv"))


def process_file(csv_path: Path, input_root: Path, output_root: Path, overwrite: bool) -> str:
    """Convert one CSV file to Parquet if it meets the row-count condition.

    Returns:
        "CONVERT" if converted,
        "SKIP" if ignored due to fewer than 5 rows,
        or an error message string if an exception occurred.
    """
    try:
        # Compute output path with mirrored directory structure
        rel_path = csv_path.relative_to(input_root)
        parquet_path = output_root / rel_path.with_suffix(".pq")
        if parquet_path.exists() and not overwrite:
            return "SKIP"

        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # Load lazily and get row count efficiently
        lf = pl.scan_csv(
            csv_path,
            null_values=["--"],
            ignore_errors=True,
            infer_schema_length=INFER_SCHEMA_LENGTH,
        )
        num_rows = lf.select(pl.len()).collect(engine="streaming").item()

        if num_rows < MIN_ROW_COUNT:
            return "SKIP"

        # Collect and write to parquet
        df = lf.collect(engine="streaming")
        df.write_parquet(parquet_path)
    except Exception as e:  # noqa: BLE001
        return f"{csv_path}: {e}"
    else:
        return "CONVERT"


def main() -> None:
    args = parse_args()

    csv_files = find_csv_files(args.input_dir)
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} CSV files. Starting conversion...")

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_file, csv_path, args.input_dir, args.output_dir, args.overwrite
            )
            for csv_path in csv_files
        ]
        for future in futures:
            future.add_done_callback(lambda _: pbar.update(1))

        results: list[str] = []
        with tqdm(total=len(futures), desc="Converting", unit="file", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    # Aggregate results
    num_converted = sum(r == "CONVERT" for r in results)
    num_skipped = sum(r == "SKIP" for r in results)
    errors = [r for r in results if r not in {"CONVERT", "SKIP"}]

    print("\n=== SUMMARY ===")
    print(f"Converted: {num_converted}")
    print(f"Skipped:   {num_skipped}")
    print(f"Errors:    {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f" - {err}")

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
