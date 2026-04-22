from pathlib import Path

import torch
from loguru import logger

from fedaugment.config import (
    DataModuleConfig,
    ExplicitDataSplit,
    ProjectionModelConfig,
    TrainerConfig,
    TrainingConfig,
)
from fedaugment.curation import CurationManager
from fedaugment.curation.curators import FarthestFirstTraversal, GridSampling
from fedaugment.embeddings import PipelineComposer
from fedaugment.embeddings.strategies import DeepJoinStrategy
from fedaugment.projections import train_projection_model
from fedaugment.utils import experiment_setup

from ..config import EMBEDDING_MODEL_REGISTRY, PROJECTION_MODEL_REGISTRY

if __name__ == "__main__":
    experiment_setup(
        "INFO",
        log_file_name="curation/webtable/proxy_analysis_curate_{time:YYYY-MM-DD_HH:mm:ss}.log",
    )

    MODEL_NAMES = ["distilroberta", "mini_l12", "mini_l6", "gte_base", "gtr_t5", "mpnet"]

    # Curation
    for model_name in MODEL_NAMES:
        logger.info("Initializing embedding pipeline with {}...", model_name)
        proxy_embedding_model = EMBEDDING_MODEL_REGISTRY[model_name].build()
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
        for ratio in [0.005]:
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
        for ratio in [0.005]:
            manager.curate_datasets(
                [GridSampling("grid", 4)],
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

    # Embedding generation
    logger.info("Initializing models...")
    models = [
        EMBEDDING_MODEL_REGISTRY[model_name].build(flash_attn=True) for model_name in MODEL_NAMES
    ]

    logger.info("Creating prompt strategies...")
    # The above embedding models have a maximum input sequence of 8192. We approximate
    # ~4 chars/token and truncate prompts at 32768 characters to not generate unnecessarily long
    # prompts.
    dj_adapted = DeepJoinStrategy("dj_adpt", max_length=8192 * 4, prompt_mode="adapted")

    logger.info("Creating pipeline composer...")
    pipelines = PipelineComposer(models, [dj_adapted])

    for ratio, k in [("005", 63049)]:
        for model_name in MODEL_NAMES:
            for in_path, out_path in [
                (
                    f"data/datasets/webtable/sample-{ratio}/fft_cos-{model_name}-dj_adpt-k={k}",
                    f"data/embeddings/webtable/sample-{ratio}/fft_cos-{model_name}-dj_adpt-k={k}",
                ),
                (
                    f"data/datasets/webtable/sample-{ratio}/fft_cos-{model_name}-dj_adpt-k={k}-pca=0.9",
                    f"data/embeddings/webtable/sample-{ratio}/fft_cos-{model_name}-dj_adpt-k={k}-pca=0.9",
                ),
                (
                    f"data/datasets/webtable/sample-{ratio}/fft_euc-{model_name}-dj_adpt-k={k}",
                    f"data/embeddings/webtable/sample-{ratio}/fft_euc-{model_name}-dj_adpt-k={k}",
                ),
                (
                    f"data/datasets/webtable/sample-{ratio}/fft_euc-{model_name}-dj_adpt-k={k}-pca=0.9",
                    f"data/embeddings/webtable/sample-{ratio}/fft_euc-{model_name}-dj_adpt-k={k}-pca=0.9",
                ),
                (
                    f"data/datasets/webtable/sample-{ratio}/grid-{model_name}-dj_adpt-k={k}",
                    f"data/embeddings/webtable/sample-{ratio}/grid-{model_name}-dj_adpt-k={k}",
                ),
                (
                    f"data/datasets/webtable/sample-{ratio}/grid-{model_name}-dj_adpt-k={k}-pca=0.9",
                    f"data/embeddings/webtable/sample-{ratio}/grid-{model_name}-dj_adpt-k={k}-pca=0.9",
                ),
            ]:
                logger.info(f"Registering datasets from {in_path}...")

                pipelines.register_datasets(in_path)

                logger.info("Starting embedding generation...")
                pipelines.generate_embeddings(out_path)

    # Projection model training
    experiment_setup(
        "INFO",
        log_file_name="curation/webtable/proxy_analysis_train_{time:YYYY-MM-DD_HH:mm:ss}.log",
    )
    torch.set_float32_matmul_precision("high")

    projection_model_spec = PROJECTION_MODEL_REGISTRY["cl_curation"]
    embedding_views = [f"{model_name}-dj_adpt" for model_name in MODEL_NAMES]
    test_paths = [
        Path(f"data/embeddings/webtable/split/test/{view}.fa") for view in embedding_views
    ]

    for ratio, k in [("005", 63049)]:
        for model_name in MODEL_NAMES:
            for method in [
                f"fft_cos-{model_name}-dj_adpt-k={k}",
                f"fft_euc-{model_name}-dj_adpt-k={k}",
                f"grid-{model_name}-dj_adpt-k={k}",
                f"fft_cos-{model_name}-dj_adpt-k={k}-pca=0.9",
                f"fft_euc-{model_name}-dj_adpt-k={k}-pca=0.9",
                f"grid-{model_name}-dj_adpt-k={k}-pca=0.9",
            ]:
                logger.info("Evaluating {} on webtable {}", method, ratio)
                train_paths = [
                    Path(f"data/embeddings/webtable/sample-{ratio}/{method}/{view}.fa")
                    for view in embedding_views
                ]
                _ = train_projection_model(
                    TrainingConfig(
                        experiment_group="proxy_sensitivity",
                        experiment_name=f"proxy_sensitivity_{method}",
                        data_module=DataModuleConfig(
                            class_="DefaultDataModule",
                            data=ExplicitDataSplit(train=train_paths, val=None, test=test_paths),
                            batch_size=projection_model_spec.batch_size,
                            num_workers=32,
                        ),
                        projection_model=ProjectionModelConfig(
                            class_=projection_model_spec.class_,
                            module_kwargs=projection_model_spec.module_kwargs,
                            criterion=projection_model_spec.criterion,
                            optimizers=projection_model_spec.optimizers,
                            schedulers=projection_model_spec.schedulers,
                            metric_batch_size=65_536,
                        ),
                        trainer=TrainerConfig(
                            accelerator="gpu",
                            max_epochs=projection_model_spec.epochs,
                            precision="bf16-mixed",
                            patience=projection_model_spec.patience,
                        ),
                        compile_model=True,
                    )
                )
