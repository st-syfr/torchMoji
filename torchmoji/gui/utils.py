"""Utility helpers shared by the TorchMoji GUI modules."""
from __future__ import annotations

import shlex
from typing import Iterable, Sequence

from ..cli import EMOTION_BUNDLES
from ..runtime import PredictionSettings
from ..settings import TorchMojiSettings

__all__ = [
    "build_cli_command",
    "format_cli_command",
    "resolve_prediction_settings",
]


def resolve_prediction_settings(settings: TorchMojiSettings) -> PredictionSettings:
    """Convert ``TorchMojiSettings`` into :class:`PredictionSettings`.

    The GUI mirrors the CLI semantics where ``None`` values fall back to the
    defaults provided by the selected emotion bundle.
    """

    bundle = EMOTION_BUNDLES.get(settings.mode, EMOTION_BUNDLES["standard"])

    if settings.emotions is not None:
        allowed_emotions = list(settings.emotions)
    else:
        allowed_emotions = list(bundle["emotions"])

    if settings.weak_emotions is not None:
        weak_emotions: Iterable[str] | None = list(settings.weak_emotions)
    else:
        weak_bundle = bundle.get("weak")
        weak_emotions = list(weak_bundle) if weak_bundle is not None else None

    if settings.strong_emotions is not None:
        strong_emotions: Iterable[str] | None = list(settings.strong_emotions)
    else:
        strong_bundle = bundle.get("strong")
        strong_emotions = list(strong_bundle) if strong_bundle is not None else None

    return PredictionSettings(
        top_k=int(settings.top_k),
        allowed_emotions=allowed_emotions,
        weak_emotions=weak_emotions,
        strong_emotions=strong_emotions,
    )


def build_cli_command(
    settings: TorchMojiSettings, text: str | None = None
) -> list[str]:
    """Return the CLI invocation representing ``settings`` and ``text``.

    The resulting list is suitable for :func:`subprocess.run`. A formatted
    string representation can be produced with :func:`format_cli_command`.
    """

    tokens: list[str] = [
        "torchmoji",
        "emojize",
        "--top-k",
        str(int(settings.top_k)),
        "--maxlen",
        str(int(settings.maxlen)),
        "--weights",
        str(settings.weights),
        "--vocab",
        str(settings.vocab),
        "--mode",
        settings.mode,
    ]

    default_settings = TorchMojiSettings()

    if settings.scores:
        tokens.append("--scores")

    def _append_list(flag: str, values: Sequence[str] | None, default: Sequence[str] | None) -> None:
        if values is None:
            return
        if default is not None and list(values) == list(default):
            # Avoid redundant flags if the explicit values match the defaults.
            return
        tokens.append(flag)
        tokens.extend(list(values))

    _append_list("--emotions", settings.emotions, default_settings.emotions)
    _append_list("--weak-emotions", settings.weak_emotions, default_settings.weak_emotions)
    _append_list("--strong-emotions", settings.strong_emotions, default_settings.strong_emotions)

    if text:
        tokens.append(text)

    return tokens


def format_cli_command(tokens: Sequence[str]) -> str:
    """Return a shell-friendly string representation of ``tokens``."""

    try:
        return shlex.join(tokens)
    except AttributeError:  # pragma: no cover - Python < 3.8 fallback
        return " ".join(shlex.quote(token) for token in tokens)
