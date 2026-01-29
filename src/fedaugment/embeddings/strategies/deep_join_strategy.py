from typing import Literal

import numpy as np
import polars as pl

from fedaugment.embeddings.collate_functions import freq_cat
from fedaugment.types import FloatArray, StrArray
from fedaugment.utils import sanitize_col_name

from .prompt_strategy import PromptStrategy


class DeepJoinStrategy(PromptStrategy):
    """Embed tables column-wise using the serialization strategy from DeepJoin."""

    def __init__(
        self,
        alias: str,
        max_length: int | None = None,
        prompt_mode: Literal["original", "adapted"] = "adapted",
    ) -> None:
        """Initialize a DeepJoinStrategy.

        Args:
            alias: A strategy alias.
            max_length: The maximum character length of the prompt (does not truncate by default).
                Note that this does not equally correspond to the token length.
            prompt_mode: The prompt construction approach. "original" exactly replicates the
                DeepJoin paper's "title-colname-stat-col" method, while "adapted" uses a slightly
                modified prompt format.
        """
        super().__init__(alias)
        self.max_length = max_length
        self.prompt_mode = prompt_mode

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.alias})"

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
                seen_ids.add(prompt_id)

                lens = col_no_na.cast(pl.Utf8).str.len_chars()
                stats = f"{lens.min():d}, {lens.max():d}, {lens.mean():.2f}"  # type: ignore[str-bytes-safe]
                col_values = freq_cat(col_no_na, sep=", ", max_length=self.max_length)
                if self.prompt_mode == "original":
                    # NOTE: The original "title-colname-stat-col" format from DeepJoin has the
                    # following issues/limitations:
                    # - the table name has no semantic meaning in our datasets (random file IDs)
                    # - corpus-wide doc frequency is not available in our distributed setting
                    prompts.append(
                        f"{table_name}. {col.name} contains {len(col_no_na)} values ({stats}): "
                        f"{col_values}"
                    )
                elif self.prompt_mode == "adapted":
                    prompts.append(
                        f"Column '{col.name}' contains {len(col_no_na)} values with lengths "
                        f"(min, max, mean): ({stats}). Values: {col_values}"
                    )
                else:
                    raise ValueError(f"Invalid prompt_mode '{self.prompt_mode}'.")

        return prompt_ids, prompts

    def process_embeddings(
        self, prompt_ids: list[str], embeddings: FloatArray
    ) -> tuple[StrArray, FloatArray]:
        prompt_ids_arr = np.array(prompt_ids, dtype=np.dtypes.StringDType())
        sorted_indices = np.argsort(prompt_ids_arr)

        return prompt_ids_arr[sorted_indices], embeddings[sorted_indices]
