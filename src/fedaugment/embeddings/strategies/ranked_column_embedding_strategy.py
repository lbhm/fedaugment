import operator
from typing import Any, Literal

import numpy as np
import polars as pl

from fedaugment.types import FloatArray, StrArray
from fedaugment.utils import sanitize_col_name

from .prompt_strategy import PromptStrategy


class RankedColumnEmbeddingStrategy(PromptStrategy):
    """Embed tables column-wise after sorting each column by a relevance score."""

    def __init__(
        self,
        alias: str,
        method: Literal["tf_idf", "bm25"] = "bm25",
        method_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a RankedColumnEmbeddingStrategy.

        Args:
            alias: A strategy alias.
            method: The scoring method to use for generating prompts. Can be "tf_idf" or "bm25".
            method_kwargs: Additional arguments for the scoring method.
        """
        super().__init__(alias)
        self.method_fn = self._tf_idf_score if method == "tf_idf" else self._bm25_score
        if method_kwargs is None:
            method_kwargs = {}
        self.sep: str = method_kwargs.pop("sep", ", ")
        self.max_length: int | None = method_kwargs.pop("max_length", None)
        self.method_kwargs = method_kwargs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.alias},"
            f"{self.method_fn.__name__}, sep={self.sep}, max_length={self.max_length}, "
            f"{self.method_kwargs})"
        )

    def _generate_prompts_from_table(
        self, table: tuple[str, pl.DataFrame]
    ) -> tuple[list[str], list[str]]:
        table_name, df = table
        prompt_ids, prompts = [], []
        seen_ids = set()

        # precompute some table statistics
        document_frequency_df = (
            df.select(pl.all().cast(str))
            .unpivot()
            .drop_nulls()
            .with_columns(pl.col("value").str.strip_chars().alias("value"))
            .group_by("value")
            .agg(pl.col("variable").n_unique().alias("doc_frequency"))
            .rename({"value": "term"})
        )
        # Convert to dictionary for O(1) lookups instead of O(n) filtering
        doc_frequencies = dict(document_frequency_df.iter_rows())
        n_docs = len(df.columns)
        avg_doc_length = sum(len(df[col].drop_nulls()) for col in df.columns) / n_docs

        for col in df:
            col_no_na = col.drop_nulls()
            if col_no_na.is_empty():
                continue

            # There might be duplicate prompt IDs due to sanitization edge cases
            prompt_id = f"{table_name}::{sanitize_col_name(col.name)}"
            if prompt_id not in seen_ids:
                prompt_ids.append(prompt_id)
                prompts.append(
                    self._generate_prompt(col_no_na, doc_frequencies, n_docs, avg_doc_length)
                )
                seen_ids.add(prompt_id)

        return prompt_ids, prompts

    def process_embeddings(
        self, prompt_ids: list[str], embeddings: FloatArray
    ) -> tuple[StrArray, FloatArray]:
        prompt_ids_arr = np.array(prompt_ids, dtype=np.dtypes.StringDType())
        sorted_indices = np.argsort(prompt_ids_arr)

        return prompt_ids_arr[sorted_indices], embeddings[sorted_indices]

    def _generate_prompt(
        self,
        column: pl.Series,
        doc_frequencies: dict[str, int],
        n_docs: int,
        avg_doc_length: float,
    ) -> str:
        """Generate a prompt for a single column using the specified sampling method."""
        # 1. calculate term frequency (TF) for the values in the column
        term_frequencies = column.cast(str).value_counts()

        # 2. calculate TF-IDF scores
        scores: dict[str, float] = {}
        value: str
        tf: int
        for value, tf in term_frequencies.iter_rows():
            value_strip = value.strip()
            doc_frequency = doc_frequencies.get(value_strip, 0)
            scores[value_strip] = self.method_fn(
                tf, doc_frequency, n_docs, column, avg_doc_length, **self.method_kwargs
            )

        # 3. sort and concatenate the values by their TF-IDF scores
        return self._concatenate_by_scores(scores)

    def _tf_idf_score(
        self,
        term_frequency: int,
        doc_frequency: int,
        n_docs: int,
        column: pl.Series,
        avg_doc_length: float,
    ) -> float:
        """Calculate the TF-IDF score for a term.

        Args:
            term_frequency: The term frequency of the term.
            doc_frequency: The document frequency of the term.
            n_docs: The total number of documents.
            column: The column in which the term appears.
            avg_doc_length: The average document (i.e., column) length.

        Returns:
            The TF-IDF score.
        """
        idf: float = np.log((n_docs + 1) / (doc_frequency + 1)) + 1  # TF-IDF IDF formula
        return term_frequency * idf

    def _bm25_score(
        self,
        term_frequency: int,
        doc_frequency: int,
        n_docs: int,
        column: pl.Series,
        avg_doc_length: float,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> float:
        """Calculate the BM25 score for a term.

        Args:
            term_frequency: The term frequency of the term.
            doc_frequency: The document frequency of the term.
            n_docs: The total number of documents.
            column: The column in which the term appears.
            avg_doc_length: The average document (i.e., column) length.
            k1: BM25 parameter k1.
            b: BM25 parameter b.
            epsilon: BM25 parameter epsilon.

        Returns:
            The BM25 score.
        """
        idf: float = np.log(
            (n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5) + epsilon
        )  # BM25 IDF formula
        return (
            idf
            * (term_frequency * (k1 + 1))
            / (term_frequency + k1 * (1 - b + b * (len(column) / avg_doc_length)))
        )

    def _concatenate_by_scores(self, scores: dict[str, float]) -> str:
        """Concatenate values sorted by their scores, optionally limited to max_length.

        Args:
            scores: A dictionary mapping values to their scores.

        Returns:
            A single string with the concatenated values sorted by score (descending).
        """
        sorted_values = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

        result: list[str] = []
        current_length = 0
        for val, _ in sorted_values:
            extra = len(val) + (len(self.sep) if result else 0)

            if self.max_length is not None and current_length + extra > self.max_length:
                break

            result.append(val)
            current_length += extra

        return self.sep.join(result)
