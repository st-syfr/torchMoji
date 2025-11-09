"""Persistent settings helpers for TorchMoji CLIs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .global_variables import PRETRAINED_PATH, VOCAB_PATH

_LIST_FIELDS = ("emotions", "weak_emotions", "strong_emotions")


@dataclass
class TorchMojiSettings:
    """Container for configurable TorchMoji CLI defaults."""

    top_k: int = 5
    maxlen: int = 30
    scores: bool = False
    mode: str = "standard"
    emotions: list[str] | None = None
    weak_emotions: list[str] | None = None
    strong_emotions: list[str] | None = None
    weights: str = str(PRETRAINED_PATH)
    vocab: str = str(VOCAB_PATH)
    api_enabled: bool = True
    api_host: str = "127.0.0.1"
    api_port: int = 5000

    def __post_init__(self) -> None:  # pragma: no cover - exercised indirectly
        for name in _LIST_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, list):
                setattr(self, name, list(value))
        self.weights = str(self.weights)
        self.vocab = str(self.vocab)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the settings to JSON-friendly primitives."""

        payload: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in _LIST_FIELDS and value is not None:
                payload[field.name] = list(value)
            else:
                payload[field.name] = value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TorchMojiSettings":
        """Create settings from a mapping, applying defaults for missing keys."""

        base = cls()
        kwargs: MutableMapping[str, Any] = {}
        for field in fields(base):
            if field.name in data:
                value = data[field.name]
            else:
                value = getattr(base, field.name)
            if field.name in _LIST_FIELDS:
                kwargs[field.name] = list(value) if value is not None else None
            else:
                kwargs[field.name] = value
        return cls(**kwargs)  # type: ignore[arg-type]

    def merge_with_namespace(
        self, namespace: argparse.Namespace
    ) -> tuple["TorchMojiSettings", set[str]]:
        """Return new settings updated with values from an ``argparse`` namespace."""

        updates: dict[str, Any] = {}
        touched: set[str] = set()
        for field in fields(self):
            if not hasattr(namespace, field.name):
                continue
            value = getattr(namespace, field.name)
            if isinstance(value, Path):
                value = str(value)
            if field.name in _LIST_FIELDS:
                value = list(value) if value is not None else None
            updates[field.name] = value
            touched.add(field.name)
        if not updates:
            return self, touched
        current = self.to_dict()
        current.update(updates)
        return type(self)(**current), touched

    def apply_to_namespace(self, namespace: argparse.Namespace) -> argparse.Namespace:
        """Populate a namespace with this settings object's values."""

        result = argparse.Namespace(**vars(namespace))
        for field in fields(self):
            setattr(result, field.name, getattr(self, field.name))
        return result


def default_settings_path() -> Path:
    """Return the path where persisted settings should be stored."""

    if sys.platform.startswith("win"):
        root = os.environ.get("APPDATA")
        if root is None:
            root = Path.home() / "AppData" / "Roaming"
        else:
            root = Path(root)
        return Path(root) / "TorchMoji" / "settings.json"

    config_root = os.environ.get("XDG_CONFIG_HOME")
    if config_root is None:
        config_root = Path.home() / ".config"
    else:
        config_root = Path(config_root)
    return Path(config_root) / "torchmoji" / "settings.json"


def load_settings(path: Path | None = None) -> TorchMojiSettings:
    """Load persisted settings, falling back to defaults if unavailable."""

    target = path or default_settings_path()
    if not Path(target).exists():
        return TorchMojiSettings()
    try:
        with Path(target).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return TorchMojiSettings()
    if not isinstance(payload, Mapping):
        return TorchMojiSettings()
    return TorchMojiSettings.from_dict(payload)


def save_settings(settings: TorchMojiSettings, path: Path | None = None) -> None:
    """Persist settings to disk as JSON."""

    target = Path(path or default_settings_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(settings.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
