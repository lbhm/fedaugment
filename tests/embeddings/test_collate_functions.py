"""Unit tests for collate functions."""

import polars as pl
import pytest

from fedaugment.embeddings.collate_functions import freq_cat, get_collate_fn, str_cat


class TestGetCollateFn:
    """Test suite for get_collate_fn factory function."""

    def test_retrieve_str_cat(self) -> None:
        """Should successfully retrieve str_cat function."""
        fn = get_collate_fn("str_cat")
        assert fn is str_cat
        assert fn.__name__ == "str_cat"

    def test_retrieve_freq_cat(self) -> None:
        """Should successfully retrieve freq_cat function."""
        fn = get_collate_fn("freq_cat")
        assert fn is freq_cat
        assert fn.__name__ == "freq_cat"

    def test_invalid_function_name(self) -> None:
        """Should raise AttributeError for non-existent function."""
        with pytest.raises(AttributeError):
            get_collate_fn("nonexistent_function")


class TestStrCat:
    """Test suite for str_cat function."""

    def test_basic_concatenation(self) -> None:
        """Should concatenate strings with default separator."""
        series = pl.Series(["apple", "banana", "cherry"])
        result = str_cat(series)
        assert result == "apple, banana, cherry"

    def test_custom_separator(self) -> None:
        """Should use custom separator."""
        series = pl.Series(["a", "b", "c"])
        result = str_cat(series, sep=" | ")
        assert result == "a | b | c"

    def test_max_length_truncation(self) -> None:
        """Should truncate at max_length."""
        series = pl.Series(["apple", "banana", "cherry"])
        result = str_cat(series, max_length=10)
        assert len(result) == 10
        assert result == "apple, ban"

    def test_max_length_exact_boundary(self) -> None:
        """Should handle exact max_length boundary."""
        series = pl.Series(["abc", "def"])
        result = str_cat(series, max_length=7)  # "abc, def" is exactly 8 chars
        assert len(result) == 7
        assert result == "abc, de"

    def test_empty_series(self) -> None:
        """Empty series should return empty string."""
        series = pl.Series([], dtype=pl.Utf8)
        result = str_cat(series)
        assert not result

    def test_single_element(self) -> None:
        """Single element should not have separator."""
        series = pl.Series(["apple"])
        result = str_cat(series)
        assert result == "apple"

    def test_unicode_handling(self) -> None:
        """Should handle Unicode characters correctly."""
        series = pl.Series(["🍎", "🍌", "🍒"])
        result = str_cat(series)
        assert result == "🍎, 🍌, 🍒"

    def test_special_characters(self) -> None:
        """Should handle special characters."""
        series = pl.Series(["hello\nworld", "foo\tbar", "baz\\qux"])
        result = str_cat(series)
        assert result == "hello\nworld, foo\tbar, baz\\qux"

    def test_empty_strings_in_series(self) -> None:
        """Should handle empty strings within series."""
        series = pl.Series(["a", "", "b"])
        result = str_cat(series)
        assert result == "a, , b"

    def test_max_length_with_no_truncation(self) -> None:
        """Max length larger than content should not truncate."""
        series = pl.Series(["a", "b"])
        result = str_cat(series, max_length=100)
        assert result == "a, b"


class TestFreqCat:
    """Test suite for freq_cat function."""

    def test_frequency_sorting(self) -> None:
        """Should sort by frequency (most frequent first)."""
        series = pl.Series(["a", "b", "a", "c", "a", "b"])
        result = freq_cat(series, sep=", ")
        # "a" appears 3x, "b" appears 2x, "c" appears 1x
        assert result == "a, b, c"

    def test_descending_frequency_order(self) -> None:
        """Should maintain descending frequency order."""
        series = pl.Series(["x"] * 3 + ["y"] * 5 + ["z"] * 1)
        result = freq_cat(series, sep=", ")
        assert result == "y, x, z"

    def test_equal_frequencies_deterministic(self) -> None:
        """Should handle ties consistently."""
        series = pl.Series(["x", "y", "z"])  # All appear once
        result = freq_cat(series, sep=", ")
        # Order should be deterministic (though may vary by implementation)
        parts = result.split(", ")
        assert len(parts) == 3
        assert set(parts) == {"x", "y", "z"}

    def test_custom_separator(self) -> None:
        """Should use custom separator."""
        series = pl.Series(["a", "a", "b"])
        result = freq_cat(series, sep=" | ")
        assert result == "a | b"

    def test_max_length_truncation(self) -> None:
        """Should truncate at max_length."""
        series = pl.Series(["apple"] * 3 + ["banana"] * 1 + ["cherry"] * 2)
        result = freq_cat(series, max_length=10)
        assert result == "apple, che"

    def test_empty_series(self) -> None:
        """Empty series should return empty string."""
        series = pl.Series([], dtype=pl.Utf8)
        result = freq_cat(series)
        assert not result

    def test_single_element_repeated(self) -> None:
        """Single unique value repeated should return that value."""
        series = pl.Series(["test"] * 10)
        result = freq_cat(series)
        assert result == "test"

    def test_all_unique_values(self) -> None:
        """All unique values should all appear."""
        series = pl.Series(["a", "b", "c", "d"])
        result = freq_cat(series, sep=", ")
        parts = result.split(", ")
        assert len(parts) == 4
        assert set(parts) == {"a", "b", "c", "d"}

    def test_unicode_with_frequency(self) -> None:
        """Should handle Unicode with frequency sorting."""
        series = pl.Series(["🍎"] * 3 + ["🍌"] * 1 + ["🍒"] * 2)
        result = freq_cat(series, sep=", ")
        assert result == "🍎, 🍒, 🍌"
