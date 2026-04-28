import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import hnswlib
import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

EPS = 1e-12
DEFAULT_SAMPLE_CODES = ("001", "005", "050")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute density metrics for curated subsets.")
    parser.add_argument(
        "--dataset", default="webtable", help="Dataset name under data/embeddings/."
    )
    parser.add_argument(
        "--pipeline", default="mpnet-dj_adpt", help="Embedding pipeline directory name."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/curation/webtable/curation_density.pq"),
        help="Output file path (suffix controls format: .pq or .csv).",
    )
    parser.add_argument(
        "--sample-codes",
        nargs="+",
        default=list(DEFAULT_SAMPLE_CODES),
        help="Sampling rate codes, e.g. 001 005 050.",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=100_000,
        help="Number of reference points sampled from the full dataset collection.",
    )
    parser.add_argument(
        "--knn-k", type=int, default=15, help="Number of neighbors used for density estimation."
    )
    parser.add_argument(
        "--metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Distance metric used for density analysis.",
    )
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.1,
        help="Quantile cutoff on reference density used to flag outliers.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reference sampling.")
    return parser.parse_args()


def load_embeddings(path: Path) -> NDArray[np.float32]:
    embeddings_path = path / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embedding files in '{path}'.")

    return np.load(embeddings_path, mmap_mode="r")


def discover_embedding_subsets(
    embeddings_root: Path, dataset: str, pipeline: str, sample_codes: Sequence[str]
) -> list[tuple[str, str, Path]]:
    subsets: list[tuple[str, str, Path]] = []
    for sample_code in sample_codes:
        sample_dir = embeddings_root / dataset / f"sample-{sample_code}"
        if not sample_dir.exists():
            logger.warning("Skipping missing sample directory '{}'.", sample_dir)
            continue

        for method_dir in sorted(
            path for path in sample_dir.iterdir() if path.is_dir() and pipeline in path.name
        ):
            subset_path = method_dir / f"{pipeline}.fa"
            if subset_path.exists():
                subsets.append((sample_code, method_dir.name, subset_path))
            else:
                logger.warning("Skipping missing subset path '{}'.", subset_path)
    return subsets


def sample_embeddings(
    embeddings: NDArray[np.float32], n_samples: int, seed: int
) -> NDArray[np.float32]:
    if embeddings.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix.")
    if n_samples <= 0:
        raise ValueError("Sample size must be positive.")

    n_rows = len(embeddings)
    if n_samples > n_rows:
        return embeddings

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_rows, size=n_samples, replace=False)
    return np.ascontiguousarray(embeddings[indices], dtype=np.float32)


def normalize_embeddings(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return embeddings / norms


def build_hnsw_index(
    embeddings: NDArray[np.float32], metric: str, ef_construction: int = 200
) -> hnswlib.Index:
    if len(embeddings) == 0:
        raise ValueError("Cannot build an ANN index on an empty embedding set.")

    space = "cosine" if metric == "cosine" else "l2"
    index = hnswlib.Index(space=space, dim=embeddings.shape[1])
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=16)
    index.add_items(embeddings, np.arange(len(embeddings)))
    index.set_ef(200)
    return index


def knn_mean_distance(
    query_embeddings: NDArray[np.float32], index: hnswlib.Index, k: int
) -> NDArray[np.float32]:
    k = min(k, index.get_current_count())
    if k <= 0:
        raise ValueError("k must be positive")

    _, distances = index.knn_query(query_embeddings, k=k)
    return distances.astype(np.float32, copy=False).mean(axis=1)


def self_knn_mean_distance(
    embeddings: NDArray[np.float32], index: hnswlib.Index, k: int
) -> NDArray[np.float32]:
    k = min(k + 1, len(embeddings))
    if k <= 1:
        return np.zeros((len(embeddings),), dtype=np.float32)

    _, distances = index.knn_query(embeddings, k=k)
    return distances[:, 1:].astype(np.float32, copy=False).mean(axis=1)


def compute_subset_metrics(
    reference_mean_distance: NDArray[np.float32],
    reference_density: NDArray[np.float32],
    subset_mean_distance: NDArray[np.float32],
    outlier_quantile: float,
) -> dict[str, float]:
    subset_density = 1.0 / (subset_mean_distance + EPS)

    outlier_threshold = float(np.quantile(reference_density, outlier_quantile))

    return {
        "knn_distance_mean": float(np.mean(subset_mean_distance)),
        "knn_distance_median": float(np.median(subset_mean_distance)),
        "knn_distance_q10": float(np.quantile(subset_mean_distance, 0.1)),
        "knn_distance_q90": float(np.quantile(subset_mean_distance, 0.9)),
        "distance_wasserstein": float(
            wasserstein_distance(subset_mean_distance, reference_mean_distance)
        ),
        "knn_density_mean": float(np.mean(subset_density)),
        "knn_density_median": float(np.median(subset_density)),
        "knn_density_q10": float(np.quantile(subset_density, 0.1)),
        "knn_density_q90": float(np.quantile(subset_density, 0.9)),
        "density_wasserstein": float(wasserstein_distance(subset_density, reference_density)),
        "outlier_rate": float(np.mean(subset_density <= outlier_threshold)),
    }


def analyze_subsets(
    full_embeddings: NDArray[np.float32],
    subsets: Iterable[tuple[str, str, Path]],
    metric: str,
    reference_size: int,
    knn_k: int,
    outlier_quantile: float,
    seed: int,
) -> pl.DataFrame:
    logger.info(
        "Sampling {} reference embeddings from the full dataset collection.", reference_size
    )
    reference_embeddings = sample_embeddings(full_embeddings, reference_size, seed)
    if metric == "cosine":
        reference_embeddings = normalize_embeddings(reference_embeddings)

    logger.info("Building HNSW index on reference embeddings with '{}' metric.", metric)
    reference_index = build_hnsw_index(reference_embeddings, metric)

    logger.info("Computing self KNN mean distance for reference embeddings.")
    reference_mean_distance = self_knn_mean_distance(reference_embeddings, reference_index, knn_k)
    reference_density = 1.0 / (reference_mean_distance + EPS)

    rows: list[dict[str, float | int | str]] = []
    for sample_code, method, subset_path in subsets:
        logger.info("Computing density metrics for '{}'.", method)
        subset_embeddings = load_embeddings(subset_path)

        if metric == "cosine":
            subset_embeddings = normalize_embeddings(subset_embeddings)
        subset_mean_distance = knn_mean_distance(subset_embeddings, reference_index, knn_k)
        metrics = compute_subset_metrics(
            reference_mean_distance=reference_mean_distance,
            reference_density=reference_density,
            subset_mean_distance=subset_mean_distance,
            outlier_quantile=outlier_quantile,
        )
        rows.append(
            {
                "method": method,
                "sample_code": sample_code,
                "sample_ratio": int(sample_code) / 1000,
                "metric": metric,
                "n_reference": len(reference_embeddings),
                "n_selected": len(subset_embeddings),
                **metrics,
            }
        )

    return pl.DataFrame(rows).sort(["sample_ratio", "method"])


def write_results(results: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        results.write_csv(output_path)
    else:
        results.write_parquet(output_path)


def main() -> None:
    args = parse_args()
    embeddings_root = Path("data/embeddings")
    full_embeddings_path = (
        embeddings_root / args.dataset / "split" / "train" / f"{args.pipeline}.fa"
    )
    full_embeddings = load_embeddings(full_embeddings_path)
    logger.info("Loaded full embeddings from '{}'.", full_embeddings_path)
    subsets = discover_embedding_subsets(
        embeddings_root=embeddings_root,
        dataset=args.dataset,
        pipeline=args.pipeline,
        sample_codes=args.sample_codes,
    )
    if not subsets:
        raise FileNotFoundError("No curated subsets were found for the requested settings.")
    logger.info("Discovered {} subsets for analysis.", len(subsets))

    results = analyze_subsets(
        full_embeddings=full_embeddings,
        subsets=subsets,
        metric=args.metric,
        reference_size=args.reference_size,
        knn_k=args.knn_k,
        outlier_quantile=args.outlier_quantile,
        seed=args.seed,
    )
    write_results(results, args.output)
    logger.success("Wrote density metrics to '{}'.", args.output)


if __name__ == "__main__":
    main()
