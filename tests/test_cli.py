from __future__ import annotations

import json

import numpy as np
import torch

from torchmoji import cli
from torchmoji.emojis import EMOJI_ALIASES


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

    class DummyTokenizer:
        def __init__(self, vocabulary, maxlen):
            self.vocabulary = vocabulary
            self.maxlen = maxlen

        def tokenize_sentences(self, sentences):
            return np.zeros((len(sentences), 2), dtype=np.int64), None, None

    def fake_build_tokenizer(vocabulary, maxlen):
        assert vocabulary == vocab
        assert maxlen == 40
        return DummyTokenizer(vocabulary, maxlen)

    class DummyModel:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True

        def __call__(self, tokenized):
            tensor = torch.zeros((1, len(EMOJI_ALIASES)), dtype=torch.float32)
            tensor[0, 3] = 0.9
            tensor[0, 5] = 0.5
            tensor[0, 0] = 0.2
            return tensor

    dummy_model = DummyModel()

    monkeypatch.setattr(cli, "_build_tokenizer", fake_build_tokenizer)
    monkeypatch.setattr(cli, "_load_model", lambda path: dummy_model)

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
    assert "-> sadness (strong)" in captured.out
    assert "-> sadness (weak)" in captured.out
    assert "-> happiness (strong)" in captured.out
    assert "score=0.9000" in captured.out
    assert dummy_model.eval_called


def test_cli_emojize_emotion_filters(monkeypatch, tmp_path, capsys):
    vocab_path = tmp_path / "vocab.json"
    vocab = {"filter": 1}
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"stub")

    class DummyTokenizer:
        def __init__(self, vocabulary, maxlen):
            self.vocabulary = vocabulary
            self.maxlen = maxlen

        def tokenize_sentences(self, sentences):
            return np.zeros((len(sentences), 2), dtype=np.int64), None, None

    def fake_build_tokenizer(vocabulary, maxlen):
        return DummyTokenizer(vocabulary, maxlen)

    class DummyModel:
        def eval(self):
            pass

        def __call__(self, tokenized):
            tensor = torch.zeros((1, len(EMOJI_ALIASES)), dtype=torch.float32)
            tensor[0, EMOJI_ALIASES.index(":joy:")] = 0.8
            tensor[0, EMOJI_ALIASES.index(":smile:")] = 0.7
            tensor[0, EMOJI_ALIASES.index(":neutral_face:")] = 0.6
            return tensor

    monkeypatch.setattr(cli, "_build_tokenizer", fake_build_tokenizer)
    monkeypatch.setattr(cli, "_load_model", lambda path: DummyModel())

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
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert ":joy:" in captured.out
    assert ":smile:" in captured.out
    assert ":neutral_face:" in captured.out
    assert "-> happiness (weak)" in captured.out
    assert "-> neutral" in captured.out
