from __future__ import annotations

import json

import numpy as np

from torchmoji import app_cli, cli
from torchmoji import settings as settings_module
from torchmoji.emojis import EMOJI_ALIASES, EmotionRanking
from torchmoji import runtime


def test_cli_missing_weights(tmp_path, capsys):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"hello": 1}), encoding="utf-8")

    exit_code = cli.main(
        [
            "emojize",
            "hello",
            "--vocab",
            str(vocab_path),
            "--weights",
            str(tmp_path / "missing.bin"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "could not find pretrained weights" in captured.err
    assert "Download them" in captured.err


def test_cli_emojize_outputs_predictions(monkeypatch, tmp_path, capsys):
    vocab_path = tmp_path / "vocab.json"
    vocab = {"hello": 1}
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"stub")

    probabilities = np.zeros(len(EMOJI_ALIASES), dtype=np.float32)
    probabilities[3] = 0.9
    probabilities[5] = 0.5
    probabilities[0] = 0.2

    selections = [
        runtime.EmojiSelection(3, EmotionRanking("sadness", "strong"), 0.9),
        runtime.EmojiSelection(5, EmotionRanking("happiness", "strong"), 0.5),
        runtime.EmojiSelection(0, EmotionRanking("happiness", "strong"), 0.2),
    ]

    dummy_runtime = _DummyRuntime(runtime.EmojiPredictionResult(probabilities, selections))

    monkeypatch.setattr(cli, "get_runtime", lambda *args, **kwargs: dummy_runtime)

    exit_code = cli.main(
        [
            "emojize",
            "hello world",
            "--vocab",
            str(vocab_path),
            "--weights",
            str(weights_path),
            "--top-k",
            "3",
            "--maxlen",
            "40",
            "--scores",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Input: hello world" in captured.out
    assert ":sob:" in captured.out  # index 3
    assert ":pensive:" in captured.out  # index 5
    assert ":joy:" in captured.out  # index 0
    assert captured.out.count("-> sadness (strong)") >= 1
    assert "-> happiness (strong)" in captured.out
    assert "score=0.9000" in captured.out
    assert dummy_runtime.calls


def test_cli_emojize_emotion_filters(monkeypatch, tmp_path, capsys):
    vocab_path = tmp_path / "vocab.json"
    vocab = {"filter": 1}
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"stub")

    probabilities = np.zeros(len(EMOJI_ALIASES), dtype=np.float32)
    probabilities[EMOJI_ALIASES.index(":joy:")] = 0.8
    probabilities[EMOJI_ALIASES.index(":smile:")] = 0.7
    probabilities[EMOJI_ALIASES.index(":neutral_face:")] = 0.6

    selections = [
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":joy:"),
            EmotionRanking("happiness", "strong"),
            0.8,
        ),
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":smile:"),
            EmotionRanking("happiness", "weak"),
            0.7,
        ),
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":neutral_face:"),
            EmotionRanking("neutral"),
            0.6,
        ),
    ]

    dummy_runtime = _DummyRuntime(runtime.EmojiPredictionResult(probabilities, selections))

    monkeypatch.setattr(cli, "get_runtime", lambda *args, **kwargs: dummy_runtime)

    exit_code = cli.main(
        [
            "emojize",
            "filter",
            "--vocab",
            str(vocab_path),
            "--weights",
            str(weights_path),
            "--top-k",
            "3",
            "--emotions",
            "happiness",
            "neutral",
            "--weak-emotions",
            "happiness",
            "--strong-emotions",
            "happiness",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert ":joy:" in captured.out
    assert ":smile:" in captured.out
    assert ":neutral_face:" in captured.out
    assert "-> happiness (strong)" in captured.out
    assert "-> neutral" in captured.out
    assert dummy_runtime.calls[0][1].allowed_emotions == ["happiness", "neutral"]


def _prepare_vocab_and_weights(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocab = {"bundle": 1}
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"stub")
    return vocab_path, weights_path


def test_cli_emojize_standard_mode_allows_all_emotions(monkeypatch, tmp_path, capsys):
    vocab_path, weights_path = _prepare_vocab_and_weights(tmp_path)
    probabilities = np.zeros(len(EMOJI_ALIASES), dtype=np.float32)
    probabilities[EMOJI_ALIASES.index(":information_desk_person:")] = 0.9
    probabilities[EMOJI_ALIASES.index(":see_no_evil:")] = 0.8

    selections = [
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":information_desk_person:"),
            EmotionRanking("contempt", "strong"),
            0.9,
        ),
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":see_no_evil:"),
            EmotionRanking("surprise", "strong"),
            0.8,
        ),
    ]

    dummy_runtime = _DummyRuntime(runtime.EmojiPredictionResult(probabilities, selections))

    monkeypatch.setattr(cli, "get_runtime", lambda *args, **kwargs: dummy_runtime)

    exit_code = cli.main(
        [
            "emojize",
            "bundle",
            "--vocab",
            str(vocab_path),
            "--weights",
            str(weights_path),
            "--mode",
            "standard",
            "--top-k",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "-> contempt (strong)" in captured.out
    assert "-> surprise (strong)" in captured.out


def test_cli_emojize_simple_mode_limits_emotions(monkeypatch, tmp_path, capsys):
    vocab_path, weights_path = _prepare_vocab_and_weights(tmp_path)
    probabilities = np.zeros(len(EMOJI_ALIASES), dtype=np.float32)
    probabilities[EMOJI_ALIASES.index(":joy:")] = 0.9
    probabilities[EMOJI_ALIASES.index(":rage:")] = 0.8
    probabilities[EMOJI_ALIASES.index(":mask:")] = 0.7
    probabilities[EMOJI_ALIASES.index(":cry:")] = 0.6

    selections = [
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":joy:"),
            EmotionRanking("happiness", "strong"),
            0.9,
        ),
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":rage:"),
            EmotionRanking("anger", "strong"),
            0.8,
        ),
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":mask:"),
            EmotionRanking("fear", "strong"),
            0.7,
        ),
        runtime.EmojiSelection(
            EMOJI_ALIASES.index(":cry:"),
            EmotionRanking("sadness", "strong"),
            0.6,
        ),
    ]

    dummy_runtime = _DummyRuntime(runtime.EmojiPredictionResult(probabilities, selections))

    monkeypatch.setattr(cli, "get_runtime", lambda *args, **kwargs: dummy_runtime)

    exit_code = cli.main(
        [
            "emojize",
            "bundle",
            "--vocab",
            str(vocab_path),
            "--weights",
            str(weights_path),
            "--mode",
            "simple",
            "--top-k",
            "4",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "-> anger (" in captured.out
    assert "-> fear (" in captured.out
    assert "contempt" not in captured.out
    assert "-> anger (strong)" in captured.out
    assert "-> contempt" not in captured.out
    assert "-> surprise" not in captured.out
    assert "contempt" not in dummy_runtime.calls[0][1].allowed_emotions


def test_app_cli_uses_persisted_defaults(monkeypatch, tmp_path, capsys):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_module, "default_settings_path", lambda: settings_path)

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"stub")

    persisted = settings_module.TorchMojiSettings(
        top_k=3,
        maxlen=42,
        scores=True,
        mode="simple",
        emotions=["happiness", "sadness"],
        weak_emotions=["happiness"],
        strong_emotions=["happiness"],
        weights=str(weights_path),
        vocab=str(vocab_path),
    )
    settings_module.save_settings(persisted)

    probabilities = np.zeros(len(EMOJI_ALIASES), dtype=np.float32)
    probabilities[3] = 0.9
    probabilities[5] = 0.5
    probabilities[0] = 0.2

    selections = [
        runtime.EmojiSelection(3, EmotionRanking("sadness", "strong"), 0.9),
        runtime.EmojiSelection(5, EmotionRanking("happiness", "strong"), 0.5),
        runtime.EmojiSelection(0, EmotionRanking("happiness", "strong"), 0.2),
    ]

    dummy_runtime = _DummyRuntime(runtime.EmojiPredictionResult(probabilities, selections))
    monkeypatch.setattr(app_cli, "get_runtime", lambda *args, **kwargs: dummy_runtime)

    exit_code = app_cli.main(["hello world"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "score=0.9000" in captured.out
    assert dummy_runtime.calls[0][1].top_k == 3
    assert dummy_runtime.calls[0][1].allowed_emotions == ["happiness", "sadness"]


def test_app_cli_overrides_are_persisted(monkeypatch, tmp_path, capsys):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_module, "default_settings_path", lambda: settings_path)

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"stub")

    initial = settings_module.TorchMojiSettings(
        top_k=2,
        maxlen=20,
        scores=False,
        weights=str(weights_path),
        vocab=str(vocab_path),
    )
    settings_module.save_settings(initial)

    probabilities = np.zeros(len(EMOJI_ALIASES), dtype=np.float32)
    probabilities[3] = 0.6
    probabilities[5] = 0.4

    selections = [
        runtime.EmojiSelection(3, EmotionRanking("sadness", "strong"), 0.6),
        runtime.EmojiSelection(5, EmotionRanking("happiness", "strong"), 0.4),
    ]

    dummy_runtime = _DummyRuntime(runtime.EmojiPredictionResult(probabilities, selections))
    monkeypatch.setattr(app_cli, "get_runtime", lambda *args, **kwargs: dummy_runtime)

    exit_code = app_cli.main(
        [
            "hello world",
            "--top-k",
            "4",
            "--scores",
            "--maxlen",
            "55",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "score=0.6000" in captured.out
    assert dummy_runtime.calls[0][1].top_k == 4

    persisted = settings_module.load_settings()
    assert persisted.top_k == 4
    assert persisted.maxlen == 55
    assert persisted.scores is True


class _DummyRuntime:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def predict(self, text, settings):
        self.calls.append((text, settings))
        return self._result
