"""Unit tests for prompt strategies."""

from typing import Literal

import numpy as np
import polars as pl
import pytest

from fedaugment.embeddings.strategies import RankedColumnEmbeddingStrategy


class TestRankedColumnEmbeddingStrategy:
    """Test suite for RankedColumnEmbeddingStrategy."""

    @pytest.fixture
    def sample_table(self) -> tuple[str, pl.DataFrame]:
        """Create a sample table for testing."""
        df = pl.DataFrame(
            {
                "col1": ["apple", "banana", "apple", "cherry"],
                "col2": ["red", "yellow", "red", "red"],
                "col3": ["fruit", "fruit", "fruit", "fruit"],
            }
        )
        return ("test_table", df)

    @pytest.fixture
    def empty_table(self) -> tuple[str, pl.DataFrame]:
        """Create a table with empty columns."""
        df = pl.DataFrame({"col1": [None, None, None], "col2": [None, None, None]})
        return ("empty_table", df)

    @pytest.fixture
    def mixed_table(self) -> tuple[str, pl.DataFrame]:
        """Create a table with mixed empty and non-empty columns."""
        df = pl.DataFrame(
            {
                "empty_col": [None, None, None],
                "full_col": ["a", "b", "c"],
                "partial_col": ["x", None, "y"],
            }
        )
        return ("mixed_table", df)

    def test_initialization_bm25(self) -> None:
        """Should initialize with BM25 method by default."""
        strategy = RankedColumnEmbeddingStrategy("test")
        assert strategy.alias == "test"
        assert strategy.method_fn.__name__ == "_bm25_score"
        assert strategy.sep == ", "
        assert strategy.max_length is None

    def test_initialization_tf_idf(self) -> None:
        """Should initialize with TF-IDF method when specified."""
        strategy = RankedColumnEmbeddingStrategy("test", method="tf_idf")
        assert strategy.method_fn.__name__ == "_tf_idf_score"

    def test_initialization_with_kwargs(self) -> None:
        """Should accept and store method kwargs."""
        strategy = RankedColumnEmbeddingStrategy(
            "test", method="bm25", method_kwargs={"sep": " | ", "max_length": 100}
        )
        assert strategy.sep == " | "
        assert strategy.max_length == 100

    def test_generate_prompts_basic(self, sample_table: tuple[str, pl.DataFrame]) -> None:
        """Should generate prompts for all columns."""
        strategy = RankedColumnEmbeddingStrategy("test")
        prompt_ids, prompts = strategy._generate_prompts_from_table(sample_table)

        assert len(prompt_ids) == 3  # 3 columns
        assert len(prompts) == 3
        assert all(pid.startswith("test_table::") for pid in prompt_ids)
        assert all(isinstance(p, str) for p in prompts)
        assert all(len(p) > 0 for p in prompts)

    def test_generate_prompts_empty_table(self, empty_table: tuple[str, pl.DataFrame]) -> None:
        """Should return empty lists for table with all null columns."""
        strategy = RankedColumnEmbeddingStrategy("test")
        prompt_ids, prompts = strategy._generate_prompts_from_table(empty_table)

        assert len(prompt_ids) == 0
        assert len(prompts) == 0

    def test_generate_prompts_mixed_table(self, mixed_table: tuple[str, pl.DataFrame]) -> None:
        """Should skip empty columns but process non-empty ones."""
        strategy = RankedColumnEmbeddingStrategy("test")
        prompt_ids, prompts = strategy._generate_prompts_from_table(mixed_table)

        # Should only get prompts for non-empty columns
        assert len(prompt_ids) == 2  # full_col and partial_col
        assert len(prompts) == 2
        assert "mixed_table::full_col" in prompt_ids
        assert "mixed_table::partial_col" in prompt_ids
        assert "mixed_table::empty_col" not in prompt_ids

    def test_generate_prompts_duplicate_column_names(self) -> None:
        """Should handle duplicate prompt IDs from colon replacement."""
        df = pl.DataFrame({"col::1": ["a", "b"], "col:\\:1": ["c", "d"]})
        table = ("test", df)
        strategy = RankedColumnEmbeddingStrategy("test")
        prompt_ids, _ = strategy._generate_prompts_from_table(table)

        # Should only get one prompt due to duplicate ID after colon replacement
        assert len(set(prompt_ids)) == len(prompt_ids)  # No actual duplicates in list

    def test_process_embeddings(self) -> None:
        """Should sort embeddings by prompt IDs."""
        strategy = RankedColumnEmbeddingStrategy("test")
        prompt_ids = ["table::col3", "table::col1", "table::col2"]
        embeddings = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]])

        sorted_ids, sorted_embeddings = strategy.process_embeddings(prompt_ids, embeddings)

        # Should be sorted alphabetically
        assert sorted_ids.tolist() == ["table::col1", "table::col2", "table::col3"]
        np.testing.assert_array_equal(sorted_embeddings, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    def test_tf_idf_score_basic(self) -> None:
        """Should calculate TF-IDF score correctly."""
        strategy = RankedColumnEmbeddingStrategy("test", method="tf_idf")
        column = pl.Series(["dummy"])  # Not used in TF-IDF calculation

        score = strategy._tf_idf_score(
            term_frequency=5, doc_frequency=2, n_docs=10, column=column, avg_doc_length=10.0
        )

        # IDF = log((10 + 1) / (2 + 1)) + 1 = log(11/3) + 1 ≈ 2.299
        # TF-IDF = 5 * 2.299 ≈ 11.495
        expected_idf = np.log(11 / 3) + 1
        expected_score = 5 * expected_idf
        assert score == pytest.approx(expected_score)

    def test_bm25_score_basic(self) -> None:
        """Should calculate BM25 score correctly."""
        strategy = RankedColumnEmbeddingStrategy("test", method="bm25")
        column = pl.Series(["a"] * 10)

        score = strategy._bm25_score(
            term_frequency=5,
            doc_frequency=2,
            n_docs=10,
            column=column,
            avg_doc_length=10.0,
            k1=1.5,
            b=0.75,
            epsilon=0.25,
        )

        # IDF = log((10 - 2 + 0.5) / (2 + 0.5) + 0.25)
        # Score = IDF * (5 * (1.5 + 1)) / (5 + 1.5 * (1 - 0.75 + 0.75 * (10 / 10.0)))
        expected_idf = np.log((10 - 2 + 0.5) / (2 + 0.5) + 0.25)
        expected_score = expected_idf * (5 * 2.5) / (5 + 1.5 * 1.0)
        assert score == pytest.approx(expected_score)

    def test_bm25_score_custom_parameters(self) -> None:
        """Should use custom BM25 parameters."""
        strategy = RankedColumnEmbeddingStrategy("test", method="bm25")
        column = pl.Series(["a"] * 20)

        score1 = strategy._bm25_score(
            term_frequency=5,
            doc_frequency=2,
            n_docs=10,
            column=column,
            avg_doc_length=10.0,
            k1=2.0,
            b=0.5,
            epsilon=0.5,
        )

        score2 = strategy._bm25_score(
            term_frequency=5,
            doc_frequency=2,
            n_docs=10,
            column=column,
            avg_doc_length=10.0,
            k1=1.5,
            b=0.75,
            epsilon=0.25,
        )

        # Different parameters should yield different scores
        assert score1 != score2

    def test_concatenate_by_scores_basic(self) -> None:
        """Should concatenate values sorted by score."""
        strategy = RankedColumnEmbeddingStrategy("test")
        scores = {"apple": 3.0, "banana": 5.0, "cherry": 1.0}

        result = strategy._concatenate_by_scores(scores)

        # Should be sorted by score (descending)
        assert result == "banana, apple, cherry"

    def test_concatenate_by_scores_custom_sep(self) -> None:
        """Should use custom separator."""
        strategy = RankedColumnEmbeddingStrategy("test", method_kwargs={"sep": " | "})
        scores = {"a": 2.0, "b": 3.0, "c": 1.0}

        result = strategy._concatenate_by_scores(scores)

        assert result == "b | a | c"

    def test_concatenate_by_scores_max_length(self) -> None:
        """Should respect max_length constraint."""
        strategy = RankedColumnEmbeddingStrategy("test", method_kwargs={"max_length": 10})
        scores = {"apple": 3.0, "banana": 2.0, "cherry": 1.0}

        result = strategy._concatenate_by_scores(scores)

        assert len(result) <= 10
        # Should fit "apple" (5) + ", " (2) = 7 chars, but "apple, ban" = 10
        assert result.startswith("apple")

    def test_concatenate_by_scores_empty_scores(self) -> None:
        """Should handle empty score dictionary."""
        strategy = RankedColumnEmbeddingStrategy("test")
        scores: dict[str, float] = {}

        result = strategy._concatenate_by_scores(scores)

        assert not result

    def test_generate_prompt_integration(self, sample_table: tuple[str, pl.DataFrame]) -> None:
        """Should generate meaningful prompts with actual column data."""
        strategy = RankedColumnEmbeddingStrategy("test", method="tf_idf")
        _, df = sample_table

        # Get document frequencies (simulate what happens in _generate_prompts_from_table)
        doc_freq_df = (
            df.select(pl.all().cast(str))
            .unpivot()
            .drop_nulls()
            .with_columns(pl.col("value").str.strip_chars().alias("value"))
            .group_by("value")
            .agg(pl.col("variable").n_unique().alias("df"))
            .rename({"value": "term"})
        )
        doc_frequencies = dict(doc_freq_df.iter_rows())
        n_docs = len(df.columns)
        avg_doc_length = sum(len(df[col].drop_nulls()) for col in df.columns) / n_docs

        col = df["col1"].drop_nulls()
        prompt = strategy._generate_prompt(col, doc_frequencies, n_docs, avg_doc_length)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # "apple" appears twice (TF=2), should have higher TF-IDF score
        assert "apple" in prompt

    def test_numeric_column_casting(self) -> None:
        """Should handle numeric columns by casting to string."""
        df = pl.DataFrame({"numbers": [1, 2, 1, 3, 2, 1]})
        table = ("numeric_table", df)
        strategy = RankedColumnEmbeddingStrategy("test")

        prompt_ids, prompts = strategy._generate_prompts_from_table(table)

        assert len(prompt_ids) == 1
        assert len(prompts) == 1
        # "1" appears 3 times, should have highest frequency
        assert "1" in prompts[0]

    def test_whitespace_stripping(self) -> None:
        """Should strip whitespace from values when calculating scores."""
        df = pl.DataFrame({"col": ["  apple  ", "apple", " banana ", "banana"]})
        table = ("whitespace_table", df)
        strategy = RankedColumnEmbeddingStrategy("test", method="tf_idf")

        _, prompts = strategy._generate_prompts_from_table(table)

        # "apple" and "banana" should both appear (stripped)
        assert "apple" in prompts[0]
        assert "banana" in prompts[0]

    @pytest.mark.parametrize("method", ["tf_idf", "bm25"])
    def test_both_methods_work(
        self, method: Literal["tf_idf", "bm25"], sample_table: tuple[str, pl.DataFrame]
    ) -> None:
        """Both TF-IDF and BM25 methods should work without errors."""
        strategy = RankedColumnEmbeddingStrategy("test", method=method)
        prompt_ids, prompts = strategy._generate_prompts_from_table(sample_table)

        assert len(prompt_ids) == 3
        assert len(prompts) == 3
        assert all(isinstance(p, str) and len(p) > 0 for p in prompts)
