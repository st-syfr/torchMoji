"""Tests for API settings in TorchMojiSettings."""
from __future__ import annotations

from torchmoji.settings import TorchMojiSettings


def test_default_api_settings() -> None:
    """Test that default API settings are correct."""
    settings = TorchMojiSettings()
    assert settings.api_enabled is True
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 5000


def test_api_settings_to_dict() -> None:
    """Test that API settings are included in to_dict()."""
    settings = TorchMojiSettings(
        api_enabled=True,
        api_host="0.0.0.0",
        api_port=8080
    )
    data = settings.to_dict()
    assert data["api_enabled"] is True
    assert data["api_host"] == "0.0.0.0"
    assert data["api_port"] == 8080


def test_api_settings_from_dict() -> None:
    """Test that API settings can be loaded from dict."""
    data = {
        "api_enabled": True,
        "api_host": "localhost",
        "api_port": 3000
    }
    settings = TorchMojiSettings.from_dict(data)
    assert settings.api_enabled is True
    assert settings.api_host == "localhost"
    assert settings.api_port == 3000


def test_api_settings_partial_dict() -> None:
    """Test that missing API settings use defaults."""
    data = {"api_enabled": True}
    settings = TorchMojiSettings.from_dict(data)
    assert settings.api_enabled is True
    assert settings.api_host == "127.0.0.1"  # Default
    assert settings.api_port == 5000  # Default


def test_api_settings_persistence() -> None:
    """Test that API settings can be round-tripped through dict."""
    original = TorchMojiSettings(
        api_enabled=True,
        api_host="192.168.1.1",
        api_port=9999
    )
    data = original.to_dict()
    restored = TorchMojiSettings.from_dict(data)
    
    assert restored.api_enabled == original.api_enabled
    assert restored.api_host == original.api_host
    assert restored.api_port == original.api_port
