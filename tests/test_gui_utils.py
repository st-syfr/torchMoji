from __future__ import annotations

import shlex

from torchmoji.cli import EMOTION_BUNDLES
from torchmoji.gui.utils import build_cli_command, format_cli_command, resolve_prediction_settings
from torchmoji.settings import TorchMojiSettings


def test_build_cli_command_includes_all_basic_flags() -> None:
    settings = TorchMojiSettings(
        top_k=7,
        maxlen=64,
        scores=True,
        mode="simple",
        emotions=["joy", "sadness"],
        weak_emotions=["joy"],
        strong_emotions=["sadness"],
        weights="/tmp/weights.h5",
        vocab="/tmp/vocab.json",
    )
    tokens = build_cli_command(settings, "hello world")

    assert tokens[:2] == ["torchmoji", "emojize"]
    assert "--scores" in tokens
    assert tokens[-1] == "hello world"
    assert "--mode" in tokens
    assert "--weights" in tokens and "/tmp/weights.h5" in tokens
    assert "--vocab" in tokens and "/tmp/vocab.json" in tokens


def test_format_cli_command_quotes_text() -> None:
    tokens = ["torchmoji", "emojize", "--top-k", "5", "hello world"]
    formatted = format_cli_command(tokens)

    assert "hello world" in formatted
    assert formatted.endswith(shlex.quote("hello world"))


def test_resolve_prediction_settings_uses_bundle_defaults() -> None:
    settings = TorchMojiSettings(mode="simple", emotions=None, weak_emotions=None, strong_emotions=None)
    prediction = resolve_prediction_settings(settings)

    bundle = EMOTION_BUNDLES["simple"]
    assert list(prediction.allowed_emotions) == list(bundle["emotions"])
    weak = bundle["weak"]
    strong = bundle["strong"]
    expected_weak = list(weak) if weak is not None else None
    expected_strong = list(strong) if strong is not None else None
    assert prediction.weak_emotions == expected_weak
    assert prediction.strong_emotions == expected_strong
