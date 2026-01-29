from collections.abc import Iterable
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from fedaugment.types import FloatArray

from .embedding_model import EmbeddingModel

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class SentenceTransformerModel(EmbeddingModel):
    """Sentence Transformer embedding model."""

    def __init__(
        self,
        model_id: str,
        alias: str | None = None,
        device: str | None = None,
        compile_model: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        encoder_batch_size: int = 32,
        tokenizer_batch_size: int = 1024,
        tokenizer_kwargs: dict[str, Any] | None = None,
        cache_folder: str | None = None,
    ) -> None:
        super().__init__(model_id, alias)

        self.model = SentenceTransformer(
            model_name_or_path=model_id,
            device=device,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            cache_folder=cache_folder,
        )
        self.tokenizer: PreTrainedTokenizerBase = self.model.tokenizer
        self.has_fast_tokenizer = isinstance(self.model.tokenizer, PreTrainedTokenizerFast)

        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(
                f"The model {self.model_id} does not have a defined embedding dimension. "
                "Please check the model configuration."
            )
        self.embedding_dim = dim
        self.max_seq_length = self.model.max_seq_length
        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.encoder_batch_size = encoder_batch_size
        self.tokenizer_batch_size = tokenizer_batch_size

        self.model.eval()
        if compile_model:
            logger.debug("Compiling model...")
            self.model.compile()  # type: ignore[no-untyped-call]
            logger.debug("{} model compiled.", self.model_id)

    def embed(
        self, prompt_ids: list[str], prompts: list[str], truncate_dim: int | None = None
    ) -> tuple[list[str], FloatArray]:
        return prompt_ids, self.model.encode(
            sentences=prompts,
            batch_size=self.encoder_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            truncate_dim=truncate_dim,
        )

    def tokenize(self, prompts: Iterable[str], max_length: int | None = None) -> list[list[int]]:
        def process_batch(batch: list[str]) -> list[list[int]]:
            encoding = self.tokenizer(
                batch,
                return_token_type_ids=False,
                return_attention_mask=False,
                truncation=max_length is not None,
                max_length=max_length,
            )
            return cast("list[list[int]]", encoding["input_ids"])

        prompts = list(prompts)
        batches = self._get_prompt_batches(prompts, self.tokenizer_batch_size)

        token_iter = chain.from_iterable(process_batch(batch) for batch in batches)
        wrapped_iter = tqdm(
            token_iter,
            desc="Tokenizing prompts",
            total=len(prompts),
            leave=False,
            unit="prompt",
            mininterval=1.0,
            dynamic_ncols=True,
        )

        return list(wrapped_iter)

    @overload
    def count_tokens(self, prompts: Iterable[str]) -> int: ...

    @overload
    def count_tokens(self, prompts: Iterable[str], per_prompt: Literal[True]) -> list[int]: ...

    @overload
    def count_tokens(self, prompts: Iterable[str], per_prompt: Literal[False]) -> int: ...

    def count_tokens(self, prompts: Iterable[str], per_prompt: bool = False) -> int | list[int]:
        def process_batch(batch: list[str]) -> list[int]:
            if self.has_fast_tokenizer:
                encoding = self.tokenizer(
                    batch,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    return_length=True,
                )
                return cast("list[int]", encoding["length"])

            encoding = self.tokenizer(
                batch, return_token_type_ids=False, return_attention_mask=False
            )
            return [len(tokens) for tokens in cast("list[list[int]]", encoding["input_ids"])]

        prompts = list(prompts)
        batches = self._get_prompt_batches(prompts, self.tokenizer_batch_size)

        count_iter = chain.from_iterable(process_batch(batch) for batch in batches)
        wrapped_iter = tqdm(
            count_iter,
            desc="Counting tokens",
            total=len(prompts),
            leave=False,
            unit="prompt",
            mininterval=1.0,
            dynamic_ncols=True,
        )

        return list(wrapped_iter) if per_prompt else sum(wrapped_iter)
