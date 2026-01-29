"""Deterministically split a directory of files into train/val/test copies."""

import argparse
import hashlib
import math
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

SPLIT_NAMES = ("train", "val", "test")
NUM_SPLITS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src", help="directory whose files will be split")
    parser.add_argument(
        "splits",
        type=parse_split_tuple,
        help="three floats that sum to 1.0, e.g. '0.7,0.2,0.1' (train,val,test)",
    )
    return parser.parse_args()


def parse_split_tuple(value: str) -> tuple[float, float, float]:
    cleaned = value.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    tokens = [tok for tok in cleaned.replace(",", " ").split() if tok]
    if len(tokens) != NUM_SPLITS:
        raise argparse.ArgumentTypeError("expected exactly three floats for the splits")

    try:
        splits = tuple(float(tok) for tok in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("splits must be valid floats") from exc

    for split in splits:
        if not 0.0 <= split <= 1.0:
            raise argparse.ArgumentTypeError("split ratios must lie within [0, 1]")

    if not math.isclose(sum(splits), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise argparse.ArgumentTypeError("splits must sum to 1.0")

    train, val, test = splits
    return (train, val, test)


def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            yield Path(dirpath) / name


def select_split(
    rel_path: Path, train_edge: float, val_edge: float
) -> Literal["train", "val", "test"]:
    digest = hashlib.sha1(rel_path.as_posix().encode("utf-8")).digest()  # noqa: S324
    integer = int.from_bytes(digest[:8], "big", signed=False)
    value = integer / float(1 << 64)
    if value < train_edge:
        return "train"
    if value < val_edge:
        return "val"
    return "test"


def main() -> int:
    args = parse_args()

    src = Path(args.src).resolve()
    if not src.is_dir():
        print(f"Source directory does not exist: {src}")
        return 2

    files = list(iter_files(src))
    if not files:
        print(f"No files found inside {src}.")
        return 0

    split_dirs: dict[str, Path] = {name: src.parent / name for name in SPLIT_NAMES}
    for directory in split_dirs.values():
        directory.mkdir(parents=True, exist_ok=False)

    train_edge = args.splits[0]
    val_edge = train_edge + args.splits[1]

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for file_path in files:
        rel_path = file_path.relative_to(src)
        split_name = select_split(rel_path, train_edge, val_edge)
        dst_path = split_dirs[split_name] / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(file_path, dst_path)
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Failed to copy {file_path} -> {dst_path}: {e}")
            continue
        counts[split_name] += 1

    total = len(files)
    print(f"Split {total:,} files into {src.parent}")
    for name in SPLIT_NAMES:
        part = counts[name] / total if total else 0.0
        print(f"{name:>5}: {counts[name]:6} files ({part * 100:6.2f}%) -> {split_dirs[name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
