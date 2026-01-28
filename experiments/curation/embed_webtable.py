from loguru import logger

from fedaugment.embeddings import PipelineComposer
from fedaugment.embeddings.strategies import DeepJoinStrategy
from fedaugment.utils import experiment_setup

from ..config import EMBEDDING_MODEL_REGISTRY

if __name__ == "__main__":
    experiment_setup(
        "INFO", log_file_name="curation/webtable/embed_{time:YYYY-MM-DD_HH-mm-ss}.log"
    )

    logger.info("Initializing models...")
    models = [
        EMBEDDING_MODEL_REGISTRY[model_name].build(flash_attn=True)
        for model_name in ["distilroberta", "mini_l12", "mini_l6", "gte_base", "gtr_t5", "mpnet"]
    ]

    logger.info("Creating prompt strategies...")
    # The above embedding models have a maximum input sequence of 8192. We approximate
    # ~4 chars/token and truncate prompts at 32768 characters to not generate unnecessarily long
    # prompts.
    dj_adapted = DeepJoinStrategy("dj_adpt", max_length=8192 * 4, prompt_mode="adapted")

    logger.info("Creating pipeline composer...")
    pipelines = PipelineComposer(models, [dj_adapted])

    for ratio, k in [("001", 12609), ("005", 63049), ("050", 630493)]:
        for in_path, out_path in [
            (
                f"data/datasets/webtable/sample-{ratio}/fft_cos-mpnet-dj_adpt-k={k}",
                f"data/embeddings/webtable/sample-{ratio}/fft_cos-mpnet-dj_adpt-k={k}",
            ),
            (
                f"data/datasets/webtable/sample-{ratio}/fft_cos-mpnet-dj_adpt-k={k}-pca=0.9",
                f"data/embeddings/webtable/sample-{ratio}/fft_cos-mpnet-dj_adpt-k={k}-pca=0.9",
            ),
            (
                f"data/datasets/webtable/sample-{ratio}/fft_euc-mpnet-dj_adpt-k={k}",
                f"data/embeddings/webtable/sample-{ratio}/fft_euc-mpnet-dj_adpt-k={k}",
            ),
            (
                f"data/datasets/webtable/sample-{ratio}/fft_euc-mpnet-dj_adpt-k={k}-pca=0.9",
                f"data/embeddings/webtable/sample-{ratio}/fft_euc-mpnet-dj_adpt-k={k}-pca=0.9",
            ),
            (
                f"data/datasets/webtable/sample-{ratio}/grid-mpnet-dj_adpt-k={k}",
                f"data/embeddings/webtable/sample-{ratio}/grid-mpnet-dj_adpt-k={k}",
            ),
            (
                f"data/datasets/webtable/sample-{ratio}/grid-mpnet-dj_adpt-k={k}-pca=0.9",
                f"data/embeddings/webtable/sample-{ratio}/grid-mpnet-dj_adpt-k={k}-pca=0.9",
            ),
            (
                f"data/datasets/webtable/sample-{ratio}/random-mpnet-dj_adpt-k={k}",
                f"data/embeddings/webtable/sample-{ratio}/random-mpnet-dj_adpt-k={k}",
            ),
        ]:
            logger.info(f"Registering datasets from {in_path}...")

            pipelines.register_datasets(in_path)

            logger.info("Starting embedding generation...")
            pipelines.generate_embeddings(out_path)

    # We also embed the full train split as a reference
    for in_path, out_path in [
        ("data/datasets/webtable/split/train", "data/embeddings/webtable/split/train")
    ]:
        logger.info(f"Registering datasets from {in_path}...")

        pipelines.register_datasets(in_path)

        logger.info("Starting embedding generation...")
        pipelines.generate_embeddings(out_path)
