import sys
from typing import Any, Protocol

import polars as pl


class CollateFn(Protocol):
    """Protocol for a function that converts a Polars Series into a single string."""

    __name__: str

    def __call__(self, column: pl.Series, *args: Any, **kwargs: Any) -> str: ...


def get_collate_fn(fn_id: str) -> CollateFn:
    fn: CollateFn = getattr(sys.modules[__name__], fn_id)
    return fn


def str_cat(column: pl.Series, *, sep: str = ", ", max_length: int | None = None) -> str:
    """Concatenate the values of a column into a single string, separated by a delimiter.

    Args:
        column: The input Series.
        sep: The delimiter to use for concatenation.
        max_length: The maximum length of the concatenated string (does not truncate by default).

    Returns:
        A single string with the concatenated values.
    """
    concat: str = column.unique(maintain_order=True).str.join(delimiter=sep).item()
    return concat[:max_length]


def freq_cat(column: pl.Series, *, sep: str = ", ", max_length: int | None = None) -> str:
    """Concatenate the values of a column into a single string using frequency-based sorting.

    Args:
        column: The input Series.
        sep: The delimiter to use for concatenation.
        max_length: The maximum length of the concatenated string (does not truncate by default).

    Returns:
        A single string with the concatenated values.
    """
    frequency_df = column.value_counts(sort=True, parallel=True, name="a random unique name")
    sorted_values = frequency_df[column.name].cast(pl.Utf8)
    concat: str = sorted_values.str.join(delimiter=sep).item()
    return concat[:max_length]
