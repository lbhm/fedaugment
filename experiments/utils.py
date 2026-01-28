"""Shared utilities for evaluation scripts."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import PathConfig


@dataclass
class CheckpointMetadata:
    """Metadata extracted from a checkpoint."""

    pipeline_names: list[str]
    embedding_dims: list[int]
    num_views: int


def load_checkpoint_metadata(checkpoint_path: Path) -> CheckpointMetadata:
    """Load and extract metadata from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        CheckpointMetadata object with extracted information
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper_params = ckpt["hyper_parameters"]

    return CheckpointMetadata(
        pipeline_names=hyper_params.get("pipeline_names", []),
        embedding_dims=hyper_params["embedding_dims"],
        num_views=len(hyper_params["embedding_dims"]),
    )


def find_checkpoints(experiment_group: str, ignore: list[str] | None = None) -> list[Path]:
    """Find all checkpoints for an experiment group.

    Args:
        experiment_group: Name of the experiment group
        ignore: Optional list of substrings; checkpoints with names containing any
            substring will be skipped

    Returns:
        List of checkpoint paths
    """
    ckpt_root = PathConfig.from_defaults().checkpoints_root / experiment_group
    if not ckpt_root.exists():
        return []

    checkpoints = sorted(ckpt_root.glob("*/last.ckpt"))
    if ignore:
        checkpoints = [
            c for c in checkpoints if not any(ignore_str in c.parent.name for ignore_str in ignore)
        ]

    return checkpoints


def find_embedding_files(
    emb_dir: Path, pipeline_names: list[str], num_views: int, normalize_names: bool = True
) -> list[Path] | None:
    """Find embedding files matching pipeline names.

    Args:
        emb_dir: Directory containing embedding files
        pipeline_names: List of pipeline names to match
        num_views: Number of expected views
        normalize_names: Whether to normalize pipeline names for matching

    Returns:
        List of embedding file paths or None if not all views found
    """
    emb_files = sorted([p for p in emb_dir.iterdir() if p.is_dir() and p.suffix == ".fa"])
    stem_to_path = {f.stem.lower(): f for f in emb_files}

    if normalize_names:
        names = [n.lower().replace(".", "_") for n in pipeline_names]
    else:
        names = [n.lower() for n in pipeline_names]

    emb_files = [stem_to_path[n] for n in names if n in stem_to_path]
    if len(emb_files) != num_views:
        return None

    return emb_files


def safe_exp_name(name: str, max_len: int = 180) -> str:
    """Helper function to ensure filenames are not too long for the filesystem."""
    if len(name) <= max_len:
        return name
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    # leave room for hyphen and hash
    return f"{name[: max_len - 9]}-{h}"
