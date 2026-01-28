"""Shared test fixtures for fedaugment tests."""

from collections.abc import Generator

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_random_seed() -> Generator[None, None, None]:
    """Set random seed for reproducibility across all tests."""
    torch.manual_seed(42)
    yield
    torch.manual_seed(torch.initial_seed())
