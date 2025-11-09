"""Tests for global_variables module."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_get_base_path_normal_environment(monkeypatch):
    """Test that _get_base_path returns the correct path in normal environment."""
    from torchmoji.global_variables import _get_base_path
    
    # Ensure we're not in a frozen environment
    monkeypatch.setattr(sys, 'frozen', False, raising=False)
    if hasattr(sys, '_MEIPASS'):
        monkeypatch.delattr(sys, '_MEIPASS', raising=False)
    
    base_path = _get_base_path()
    
    # The base path should end with the repository directory
    assert Path(base_path).exists()
    assert (Path(base_path) / 'torchmoji').exists()


def test_get_base_path_pyinstaller_environment(monkeypatch):
    """Test that _get_base_path returns _MEIPASS when running in PyInstaller bundle."""
    from torchmoji.global_variables import _get_base_path
    
    # Simulate PyInstaller environment
    test_path = '/tmp/test_meipass'
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', test_path, raising=False)
    
    base_path = _get_base_path()
    
    # Should return the _MEIPASS path
    assert base_path == test_path


def test_vocab_path_construction():
    """Test that VOCAB_PATH is correctly constructed."""
    from torchmoji.global_variables import VOCAB_PATH, ROOT_PATH
    
    # VOCAB_PATH should be a path to vocabulary.json in the model directory
    assert 'vocabulary.json' in VOCAB_PATH
    assert 'model' in VOCAB_PATH
    # Should use ROOT_PATH as base
    assert VOCAB_PATH.startswith(ROOT_PATH)


def test_pretrained_path_construction():
    """Test that PRETRAINED_PATH is correctly constructed."""
    from torchmoji.global_variables import PRETRAINED_PATH, ROOT_PATH
    
    # PRETRAINED_PATH should be a path to pytorch_model.bin in the model directory
    assert 'pytorch_model.bin' in PRETRAINED_PATH
    assert 'model' in PRETRAINED_PATH
    # Should use ROOT_PATH as base
    assert PRETRAINED_PATH.startswith(ROOT_PATH)


def test_paths_exist_in_normal_environment():
    """Test that the vocabulary file exists in a normal development environment."""
    from torchmoji.global_variables import VOCAB_PATH
    
    # In normal environment, vocabulary.json should exist
    assert Path(VOCAB_PATH).exists(), f"Vocabulary file not found at {VOCAB_PATH}"
