"""Graphical user interface entry points for TorchMoji."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["TorchMojiApplication"]


if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .app import TorchMojiApplication  # noqa: F401


def __getattr__(name: str):  # pragma: no cover - small wrapper
    if name != "TorchMojiApplication":
        raise AttributeError(name)
    try:
        module = import_module("torchmoji.gui.app")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        message = "TorchMoji GUI requires PySide6 and pystray. Reinstall torchmoji to restore missing dependencies."
        if exc.name and exc.name not in {"torchmoji.gui.app", "torchmoji.gui"}:
            message = f"{message} Missing module: {exc.name}."
        raise ImportError(message) from exc
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "TorchMoji GUI requires PySide6 and pystray. Reinstall torchmoji to restore missing dependencies."
        ) from exc
    return getattr(module, name)
