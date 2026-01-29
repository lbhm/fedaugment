from typing import Any

import numpy as np
import polars as pl

from fedaugment.embeddings.collate_functions import CollateFn, get_collate_fn
from fedaugment.types import FloatArray, StrArray
from fedaugment.utils import sanitize_col_name

from .prompt_strategy import PromptStrategy


class ColumnEmbeddingStrategy(PromptStrategy):
    """Embed tables column-wise using a collate function."""

    def __init__(
        self,
        alias: str,
        collate_fn: str | CollateFn,
        collate_fn_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a ColumnEmbeddingStrategy.

        Args:
            alias: A strategy alias.
            collate_fn: The function to collate column values into a single prompt.
            collate_fn_kwargs: Additional arguments for the collate function.
        """
        super().__init__(alias)
        self.collate_fn = get_collate_fn(collate_fn) if isinstance(collate_fn, str) else collate_fn
        self.collate_fn_kwargs = collate_fn_kwargs or {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.alias}, {self.collate_fn.__name__}, "
            f"{self.collate_fn_kwargs})"
        )

    def _generate_prompts_from_table(
        self, table: tuple[str, pl.DataFrame]
    ) -> tuple[list[str], list[str]]:
        table_name, df = table
        prompt_ids, prompts = [], []
        seen_ids = set()

        for col in df:
            col_no_na = col.drop_nulls()
            if col_no_na.is_empty():
                continue

            # There might be duplicate prompt IDs due to sanitization edge cases
            prompt_id = f"{table_name}::{sanitize_col_name(col.name)}"
            if prompt_id not in seen_ids:
                prompt_ids.append(prompt_id)
                prompts.append(self.collate_fn(col_no_na, **self.collate_fn_kwargs))
                seen_ids.add(prompt_id)

        return prompt_ids, prompts

    def process_embeddings(
        self, prompt_ids: list[str], embeddings: FloatArray
    ) -> tuple[StrArray, FloatArray]:
        prompt_ids_arr = np.array(prompt_ids, dtype=np.dtypes.StringDType())
        sorted_indices = np.argsort(prompt_ids_arr)

        return prompt_ids_arr[sorted_indices], embeddings[sorted_indices]
