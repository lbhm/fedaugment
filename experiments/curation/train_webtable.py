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
from fedaugment.projections import train_projection_model
from fedaugment.utils import experiment_setup

from ..config import PROJECTION_MODEL_REGISTRY

if __name__ == "__main__":
    experiment_setup(
        "INFO", log_file_name="curation/webtable/train_{time:YYYY-MM-DD_HH-mm-ss}.log"
    )
    torch.set_float32_matmul_precision("high")

    projection_model_spec = PROJECTION_MODEL_REGISTRY["cl_curation"]
    embedding_views = [
        "distilroberta-dj_adpt",
        "mini_l12-dj_adpt",
        "mini_l6-dj_adpt",
        "gte_base-dj_adpt",
        "gtr_t5-dj_adpt",
        "mpnet-dj_adpt",
    ]
    val_paths = [Path(f"data/embeddings/webtable/split/val/{view}.fa") for view in embedding_views]
    test_paths = [
        Path(f"data/embeddings/webtable/split/test/{view}.fa") for view in embedding_views
    ]

    for ratio, k in [("001", 12609), ("005", 63049), ("050", 630493)]:
        for method in [
            f"fft_cos-mpnet-dj_adpt-k={k}",
            f"fft_euc-mpnet-dj_adpt-k={k}",
            f"grid-mpnet-dj_adpt-k={k}",
            f"fft_cos-mpnet-dj_adpt-k={k}-pca=0.9",
            f"fft_euc-mpnet-dj_adpt-k={k}-pca=0.9",
            f"grid-mpnet-dj_adpt-k={k}-pca=0.9",
            f"random-mpnet-dj_adpt-k={k}",
        ]:
            logger.info("Evaluating {} on webtable {}", method, ratio)
            train_paths = [
                Path(f"data/embeddings/webtable/sample-{ratio}/{method}/{view}.fa")
                for view in embedding_views
            ]
            _ = train_projection_model(
                TrainingConfig(
                    experiment_group="curation",
                    experiment_name=f"webtable_{method}",
                    data_module=DataModuleConfig(
                        class_="DefaultDataModule",
                        data=ExplicitDataSplit(train=train_paths, val=val_paths, test=test_paths),
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

    train_paths = [
        Path(f"data/embeddings/webtable/split/train/{view}.fa") for view in embedding_views
    ]
    logger.info("Evaluating on full train split of webtable")
    _ = train_projection_model(
        TrainingConfig(
            experiment_group="curation",
            experiment_name="webtable_full",
            data_module=DataModuleConfig(
                class_="DefaultDataModule",
                data=ExplicitDataSplit(train=train_paths, val=val_paths, test=test_paths),
                batch_size=projection_model_spec.batch_size,
                num_workers=2,
                mmap_mode=None,
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
                max_time=None,
            ),
            compile_model=True,
        )
    )
