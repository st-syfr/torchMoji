from __future__ import annotations

import json

import numpy as np
import torch

from torchmoji import runtime
from torchmoji.emojis import EMOJI_ALIASES


def test_get_runtime_reuses_cache(monkeypatch, tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"hello": 1}), encoding="utf-8")

    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"stub")

    call_counts = {"tokenizer": 0, "model": 0}

    class DummyTokenizer:
        def __init__(self, vocabulary, maxlen):
            call_counts["tokenizer"] += 1
            self.vocabulary = vocabulary
            self.maxlen = maxlen

        def tokenize_sentences(self, sentences):
            return np.zeros((len(sentences), 2), dtype=np.int64), None, None

    class DummyModel:
        def __init__(self):
            call_counts["model"] += 1

        def eval(self):
            pass

        def __call__(self, tokenized):
            tensor = torch.zeros((1, len(EMOJI_ALIASES)), dtype=torch.float32)
            tensor[0, 0] = 1.0
            return tensor

    monkeypatch.setattr(runtime, "SentenceTokenizer", DummyTokenizer)
    monkeypatch.setattr(runtime, "torchmoji_emojis", lambda path: DummyModel())
    monkeypatch.setattr(runtime, "_runtime_cache", {})

    first = runtime.get_runtime(weights_path, vocab_path, 10)
    second = runtime.get_runtime(weights_path, vocab_path, 10)

    assert first is second
    assert call_counts["tokenizer"] == 1
    assert call_counts["model"] == 1


def test_runtime_predict_returns_expected_metadata(monkeypatch, tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"world": 1}), encoding="utf-8")

    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"stub")

    tokenizer_calls = {"instances": 0, "tokenize": 0}

    class DummyTokenizer:
        def __init__(self, vocabulary, maxlen):
            tokenizer_calls["instances"] += 1
            self.vocabulary = vocabulary
            self.maxlen = maxlen

        def tokenize_sentences(self, sentences):
            tokenizer_calls["tokenize"] += 1
            return np.zeros((len(sentences), 2), dtype=np.int64), None, None

    class DummyModel:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True

        def __call__(self, tokenized):
            tensor = torch.zeros((1, len(EMOJI_ALIASES)), dtype=torch.float32)
            tensor[0, EMOJI_ALIASES.index(":joy:")] = 0.9
            tensor[0, EMOJI_ALIASES.index(":sob:")] = 0.8
            tensor[0, EMOJI_ALIASES.index(":mask:")] = 0.7
            return tensor

    dummy_model = DummyModel()

    monkeypatch.setattr(runtime, "SentenceTokenizer", DummyTokenizer)
    monkeypatch.setattr(runtime, "torchmoji_emojis", lambda path: dummy_model)
    monkeypatch.setattr(runtime, "_runtime_cache", {})

    cached_runtime = runtime.get_runtime(weights_path, vocab_path, 20)
    settings = runtime.PredictionSettings(
        top_k=2,
        allowed_emotions=["happiness", "sadness", "fear", "anger", "surprise", "disgust", "contempt", "neutral"],
        weak_emotions=None,
        strong_emotions=None,
    )

    first_result = cached_runtime.predict("hello world", settings)
    second_result = cached_runtime.predict("another", settings)

    assert dummy_model.eval_called
    assert tokenizer_calls["instances"] == 1
    assert tokenizer_calls["tokenize"] == 2
    assert first_result.probabilities.shape == (len(EMOJI_ALIASES),)
    assert second_result.probabilities.shape == (len(EMOJI_ALIASES),)
    assert first_result.selections
    assert first_result.selections[0].ranking is not None
    assert first_result.selections[0].score is not None
