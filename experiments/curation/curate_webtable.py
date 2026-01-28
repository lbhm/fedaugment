from pathlib import Path

import torch
from loguru import logger

from fedaugment.curation import CurationManager
from fedaugment.curation.curators import FarthestFirstTraversal, GridSampling, RandomSampling
from fedaugment.embeddings.strategies import DeepJoinStrategy
from fedaugment.utils import experiment_setup

from ..config import EMBEDDING_MODEL_REGISTRY

if __name__ == "__main__":
    experiment_setup(
        "INFO", log_file_name="curation/webtable/curate_{time:YYYY-MM-DD_HH-mm-ss}.log"
    )

    logger.info("Initializing embedding pipeline...")
    proxy_embedding_model = EMBEDDING_MODEL_REGISTRY["mpnet"].build()
    prompt_strategy = DeepJoinStrategy("dj_adpt", max_length=8192 * 4, prompt_mode="adapted")

    logger.info("Starting embedding generation...")
    manager = CurationManager()
    manager.generate_proxy_embeddings(
        Path("data/datasets/webtable/split/train"),
        proxy_embedding_model,
        prompt_strategy,
        batch_size=500_000,
        device="cuda",
        dtype=torch.bfloat16,
    )

    logger.info("Starting dataset curation...")
    for ratio in [0.001, 0.005, 0.05]:
        manager.curate_datasets(
            [
                FarthestFirstTraversal("fft_euc", metric="euclidean"),
                FarthestFirstTraversal("fft_cos", metric="cosine"),
            ],
            Path(f"data/datasets/webtable/sample-{int(ratio * 1000):0>3}"),
            k=ratio,
            seed=1,
            pca_dim=None,
        )
        manager.to(dtype=torch.float32)  # PCA requires float32
        manager.curate_datasets(
            [
                FarthestFirstTraversal("fft_euc", metric="euclidean"),
                FarthestFirstTraversal("fft_cos", metric="cosine"),
            ],
            Path(f"data/datasets/webtable/sample-{int(ratio * 1000):0>3}"),
            k=ratio,
            seed=1,
            pca_dim=0.9,
        )

    manager.to(device="cpu")
    for ratio in [0.001, 0.005, 0.05]:
        manager.curate_datasets(
            [GridSampling("grid", 4), RandomSampling("random")],
            Path(f"data/datasets/webtable/sample-{int(ratio * 1000):0>3}"),
            k=ratio,
            seed=1,
            pca_dim=None,
        )
        manager.curate_datasets(
            [GridSampling("grid", 4)],
            Path(f"data/datasets/webtable/sample-{int(ratio * 1000):0>3}"),
            k=ratio,
            seed=1,
            pca_dim=0.9,
        )
