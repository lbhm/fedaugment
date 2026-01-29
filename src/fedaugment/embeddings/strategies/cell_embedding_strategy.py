from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from fedaugment.types import FloatArray, StrArray
from fedaugment.utils import sanitize_col_name

from .prompt_strategy import PromptStrategy

if TYPE_CHECKING:
    from collections.abc import Callable


class CellEmbeddingStrategy(PromptStrategy):
    """Embed tables cell-wise and aggregate embeddings per column."""

    def __init__(self, alias: str, aggregration: Literal["mean", "median"] = "mean") -> None:
        """Initialize a CellEmbeddingStrategy.

        Args:
            alias: A strategy alias.
            aggregration: The function to aggregate cell embeddings.
        """
        super().__init__(alias)

        self.agg_fn: Callable[..., FloatArray]
        match aggregration:
            case "mean":
                self.agg_fn = np.mean
            case "median":
                self.agg_fn = np.median
            case _:
                raise ValueError(f"Unsupported aggregation function: {aggregration}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.alias}, {self.agg_fn}"

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
            prompt_id_base = f"{table_name}::{sanitize_col_name(col.name)}"
            if prompt_id_base not in seen_ids:
                prompt_ids.extend([f"{prompt_id_base}::{i}" for i in range(len(col_no_na))])
                # Ensure all cell values are converted to strings, regardless of column type
                prompts.extend(col_no_na.cast(pl.Utf8).to_list())
                seen_ids.add(prompt_id_base)

        return prompt_ids, prompts

    def process_embeddings(
        self, prompt_ids: list[str], embeddings: FloatArray
    ) -> tuple[StrArray, FloatArray]:
        # Convert to variable-length string arrays for better efficiency
        col_ids: StrArray = np.strings.rpartition(
            a=np.asarray(prompt_ids, dtype=np.dtypes.StringDType()),
            sep=np.asarray("::", dtype=np.dtypes.StringDType()),
        )[0]

        # Sort by col_id
        sort_indices = np.argsort(col_ids)
        col_ids = col_ids[sort_indices]
        embeddings = embeddings[sort_indices]

        # Unique group info
        unique_ids, group_starts, group_counts = np.unique(
            col_ids, return_index=True, return_counts=True
        )

        # Aggregate embeddings per unique_id group
        aggregated = np.vstack(
            [
                self.agg_fn(embeddings[start : start + count], axis=0)
                for start, count in zip(group_starts, group_counts, strict=True)
            ]
        )

        return unique_ids, aggregated
