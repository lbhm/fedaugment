from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Literal, overload

from fedaugment.types import FloatArray


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    model_id: str
    """The vendor-specific identifier of the model."""
    embedding_dim: int
    """The dimension of the model's embeddings."""
    max_seq_length: int
    """The maximum input sequence length of the model."""

    def __init__(self, model_id: str, alias: str | None = None) -> None:
        self.model_id = model_id
        self.alias = alias or model_id

        if "/" in self.alias:
            raise ValueError(
                f"Alias '{self.alias}' contains a slash. Please use a different alias without "
                "slashes as we use the alias for naming files."
            )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_id})"

    @abstractmethod
    def embed(
        self, prompt_ids: list[str], prompts: list[str], truncate_dim: int | None = None
    ) -> tuple[list[str], FloatArray]:
        """Embed a collection of prompts.

        Args:
            prompt_ids: A list of identifiers for each prompt. Since processing can be
                async or out-of-order, each prompt must have a unique ID.
            prompts: A list of embedding prompts.
            truncate_dim: If specified, truncate embeddings to this dimensionality.

        Returns:
            A tuple containing a list of prompt IDs and an array with their embeddings.
            The order is not guaranteed to match the input order.
        """

    @abstractmethod
    def tokenize(self, prompts: Iterable[str], max_length: int | None = None) -> list[list[int]]:
        """Tokenize an iterable of prompts.

        Args:
            prompts: An iterable of prompts to be tokenized.
            max_length: The maximum length of the tokenized sequences. If a prompt exceeds this
                length, it will be truncated. If None, no truncation is applied.

        Returns:
            A list of tokenized prompts.
        """

    @overload
    def count_tokens(self, prompts: Iterable[str]) -> int: ...

    @overload
    def count_tokens(self, prompts: Iterable[str], per_prompt: Literal[True]) -> list[int]: ...

    @overload
    def count_tokens(self, prompts: Iterable[str], per_prompt: Literal[False]) -> int: ...

    def count_tokens(self, prompts: Iterable[str], per_prompt: bool = False) -> int | list[int]:
        """Count the number of tokens in an iterable of prompts.

        Args:
            prompts: An iterable of prompts to count tokens for.
            per_prompt: If True, return a list of token counts for each prompt. Otherwise, return
                the total token count.

        Returns:
            The number of tokens (per prompt or in total).
        """
        tokenized_prompts = self.tokenize(prompts)
        num_tokens = [len(tokens) for tokens in tokenized_prompts]
        return num_tokens if per_prompt else sum(num_tokens)

    def _get_prompt_batches(self, prompts: Iterable[str], batch_size: int) -> Iterator[list[str]]:
        it = iter(prompts)
        while batch := list(islice(it, batch_size)):
            yield batch
