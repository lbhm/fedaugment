import logging
import math
import multiprocessing
import os
import subprocess  # noqa: S404
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import loguru
import torch
from loguru import logger
from tqdm.auto import tqdm

from fedaugment.types import HDF5Dict, StrPath
from fedaugment.utils.logger import InterceptHandler


def experiment_setup(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_file_level: str | None = None,
    log_file_name: str | None = None,
    filter_duplicates: bool = True,
    cd_to_top_level: bool = True,
    use_expandable_segments: bool = True,
    use_tqdm: bool = True,
) -> None:
    """Configure logging and further experiment bootstrapping.

    See https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    for details on CUDA memory optimizations in PyTorch.

    Args:
        log_level: The logging level for the console logger.
        log_to_file: Whether to log to a file.
        log_file_level: The logging level for the file logger (defaults to console level).
        log_file_name: The name of the log file (defaults to timestamped name).
        filter_duplicates: Whether to filter duplicate log messages.
        cd_to_top_level: Whether to change the current working directory to the Git root.
        use_expandable_segments: Whether to enable expandable segments for CUDA memory
            allocation to reduce fragmentation.
        use_tqdm: Whether to use a tqdm sink for console logging.
    """
    # We pin the start method to `fork` to avoid pickling issues with h5py until we improve our
    # dataset implementation to lazily open HDF5 files within each worker process (without
    # sacrificing performance).
    try:
        multiprocessing.set_start_method("forkserver")
    except RuntimeError:
        pass  # Start method was already set

    # Remove default logger
    logger.remove()

    # Add a console logger
    dedup_filter = make_dedup_filter() if filter_duplicates else None
    logger.add(
        tqdm_sink if use_tqdm else sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level:.1}</level> | <level>{message}</level>"
        ),
        filter=dedup_filter,
        colorize=True,
        level=log_level,
        diagnose=True,  # Extra exception information
    )

    if cd_to_top_level:
        cd_to_git_root()

    if use_expandable_segments:
        # Reduce VRAM usage by reducing fragmentation
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    else:
        # Ensure that this was not set previously
        os.environ.pop("PYTORCH_ALLOC_CONF", None)

    # Add a file logger (only for main node in distributed training)
    if log_to_file and int(os.getenv("NODE_RANK", "0")) == 0:
        log_folder = Path("logs")
        log_folder.mkdir(parents=True, exist_ok=True)
        file_name = log_file_name or "exp_{time:YYYY-MM-DD_HH-mm-ss}.log"
        logger.add(
            Path(log_folder) / file_name,
            format=("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}"),
            level=log_file_level or "DEBUG",
            encoding="utf-8",
            backtrace=True,  # Enables better stack traces
            enqueue=True,  # Use multiprocessing-safe queuing
        )

    # Intercept all standard logging
    logging.root.handlers = [InterceptHandler()]
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    # Redirect warnings to Loguru
    warnings.showwarning = lambda msg, cat, fn, ln, *args: logger.warning(  # ty: ignore[invalid-assignment]
        f"{cat.__name__}: {msg} ({fn}:{ln})"
    )

    logger.debug(
        "Experiment setup complete (redirected {} loggers)", len(logging.root.manager.loggerDict)
    )
    logger.debug(
        "Set PYTORCH_ALLOC_CONF to '{}'", os.environ.get("PYTORCH_ALLOC_CONF", "<not set>")
    )


def make_dedup_filter(levels: set[str] | None = None) -> Callable[["loguru.Record"], bool]:
    """Create a filter to remove duplicate log messages.

    Args:
        levels: The log levels to filter for duplicates (filters warnings by default).

    Returns:
        A filter function for Loguru.
    """
    seen_messages: set[str] = set()
    if levels is None:
        levels = {"WARNING"}

    def dedup_filter(record: "loguru.Record") -> bool:
        if record["level"].name.upper() in levels:
            if record["message"] in seen_messages:
                return False
            seen_messages.add(record["message"])
        return True

    return dedup_filter


def tqdm_sink(msg: str) -> None:
    """A Loguru sink that writes to tqdm to avoid breaking tqdm's progress bar."""
    tqdm.write(msg, end="")  # end="" to avoid double newlines from Loguru


def cd_to_git_root() -> None:
    """Change the current working directory to the top-level directory of the Git repository.

    This function assumes that `git` is installed and errors if it is called outside a Git
    repository.
    """
    # Get the top-level directory of the current Git repository
    git_root = get_repo_root()

    # Change the working directory to the Git root
    os.chdir(git_root)
    logger.debug("Changed working directory to: {}", git_root)


def get_repo_root() -> Path:
    """Get the root directory of the Git repository.

    Returns:
        Path to the Git repository root.
    """
    try:
        git_root = (
            subprocess.check_output(
                ["/usr/bin/git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )
        return Path(git_root)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Not inside a Git repository or Git is not installed.") from e


def analyze_hdf5_file(file: StrPath) -> tuple[HDF5Dict, dict[str, Any]]:
    """Analyze the hierarchical structure of an HDF5 file.

    Args:
        file: Path to the HDF5 file.

    Returns:
        A nested dictionary representing the group and dataset structure.
        Groups are keys with nested dictionaries as values.
        Datasets are keys with (shape, dtype) tuples as values.
    """

    def analyze_group(group: h5py.Group) -> HDF5Dict:
        structure: HDF5Dict = {}
        for key, item in group.items():
            assert isinstance(key, str), f"Key {key} is not a string"
            if isinstance(item, h5py.Group):
                structure[key] = (analyze_group(item), dict(item.attrs.items()))
            elif isinstance(item, h5py.Dataset):
                structure[key] = (item.shape, item.dtype, dict(item.attrs.items()))
        return structure

    with h5py.File(file, "r") as hdf_file:
        return analyze_group(hdf_file), dict(hdf_file.attrs.items())


def close_hdf5_files(files: list[h5py.File]) -> None:
    """Closes all HDF5 files in the provided list.

    Args:
        files: A list of HDF5 file objects to close.
    """
    for file in files:
        file.close()


def summarize_token_count(token_count: dict[str, list[int]]) -> None:
    """Aggregate the results of a call to count_table_collection_tokens() and print them.

    Args:
        token_count: A dictionary with the token count per model invocation for each pipeline.
    """
    stats = {
        pipeline: {
            "sum": sum(count),
            "min": min(count),
            "max": max(count),
            "mean": sum(count) / len(count),
        }
        for pipeline, count in token_count.items()
    }
    lens = {"name": max((len(p) for p in token_count), default=0)}
    for key in ["sum", "min", "max", "mean"]:
        max_val = max((int(s[key]) for s in stats.values()), default=0)
        lens[key] = math.floor(math.log10(max_val)) + 1 if max_val > 0 else 1

    for name, stat in stats.items():
        logger.info(
            f"{name:>{lens['name']}}: {stat['sum']:>{lens['sum']}} tokens "
            f"(mean: {stat['mean']:>{lens['mean'] + 3}.2f} - min: {stat['min']:>{lens['min']}} - "
            f"max: {stat['max']:>{lens['max']}})"
        )


def sanitize_col_name(col_name: str) -> str:
    """Sanitize a column name to avoid issues with Polars.

    - "*" is interpreted as a wildcard in Polars, so we escape it
    - "::" is not allowed in column names since we use it as a separator in prompt IDs
    - Columns starting with "^" and ending with "$" are interpreted as regex patterns
    """
    if col_name == "*":
        return r"\*"
    col_name = col_name.replace("::", ":\\:")  # Escape double colons
    if col_name.startswith("^") and col_name.endswith("$"):
        return "\\" + col_name + "\\"

    return col_name


def reset_gpu_memory_tracking() -> None:
    """Reset CUDA peak memory tracking statistics.

    Call this before a section of code to measure its peak GPU memory usage.
    Use `get_max_gpu_memory_gb()` afterward to retrieve the peak.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_max_gpu_memory_gb() -> float | None:
    """Get peak GPU memory usage in GB since last reset.

    Returns:
        Peak GPU memory in GB, or None if CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return None
