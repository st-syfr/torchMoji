"""Shared pytest fixtures for the test suite."""
import json
from pathlib import Path

import pytest

import test_helper  # noqa: F401  # Ensures repository root is on sys.path
from torchmoji.global_variables import VOCAB_PATH


@pytest.fixture(scope="session")
def vocab():
    """Load the vocabulary once per test session."""
    vocab_file = Path(VOCAB_PATH)
    with vocab_file.open("r", encoding="utf-8") as f:
        return json.load(f)
