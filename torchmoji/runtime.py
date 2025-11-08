"""Runtime utilities for loading and reusing the TorchMoji model."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from typing import Iterable, Sequence

import numpy as np
import torch

from .emojis import EMOJI_ALIASES, EmotionRanking, filter_emojis_by_emotion
from .model_def import torchmoji_emojis
from .sentence_tokenizer import SentenceTokenizer


@dataclass(frozen=True)
class EmojiSelection:
    """Container for an emoji prediction and its metadata."""

    index: int
    ranking: EmotionRanking | None
    score: float | None


@dataclass(frozen=True)
class EmojiPredictionResult:
    """Prediction output returned by :class:`TorchMojiRuntime`."""

    probabilities: np.ndarray
    selections: list[EmojiSelection]


@dataclass(frozen=True)
class PredictionSettings:
    """Configuration controlling how predictions are generated."""

    top_k: int
    allowed_emotions: Sequence[str]
    weak_emotions: Iterable[str] | None
    strong_emotions: Iterable[str] | None


class TorchMojiRuntime:
    """Helper encapsulating the TorchMoji tokenizer and model."""

    def __init__(self, weights_path: Path, vocab_path: Path, maxlen: int) -> None:
        self._weights_path = Path(weights_path)
        self._vocab_path = Path(vocab_path)
        self._maxlen = maxlen

        self._vocabulary = self._load_vocabulary(self._vocab_path)
        self._tokenizer = SentenceTokenizer(self._vocabulary, maxlen)
        self._model = torchmoji_emojis(str(self._weights_path))
        self._model.eval()

    @staticmethod
    def _load_vocabulary(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def predict(self, text: str, settings: PredictionSettings) -> EmojiPredictionResult:
        try:
            tokenized, _, _ = self._tokenizer.tokenize_sentences([text])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Error tokenizing text: {exc}") from exc

        with torch.no_grad():
            probabilities = self._model(tokenized)[0]

        if hasattr(probabilities, "cpu"):
            probabilities = probabilities.cpu()
        if hasattr(probabilities, "numpy"):
            probabilities = probabilities.numpy()

        probabilities_array = np.asarray(probabilities, dtype=np.float32)

        selections: list[EmojiSelection]
        if probabilities_array.size == len(EMOJI_ALIASES):
            filtered = filter_emojis_by_emotion(
                probabilities_array,
                settings.top_k,
                settings.allowed_emotions,
                settings.weak_emotions,
                settings.strong_emotions,
            )
            selections = [
                EmojiSelection(
                    index=index,
                    ranking=ranking,
                    score=float(probabilities_array[index]) if index < probabilities_array.size else None,
                )
                for index, ranking in filtered
            ]
        else:
            indices = _top_indices(probabilities_array, settings.top_k)
            selections = [
                EmojiSelection(
                    index=index,
                    ranking=None,
                    score=float(probabilities_array[index]) if index < probabilities_array.size else None,
                )
                for index in indices
            ]

        return EmojiPredictionResult(probabilities_array, selections)


def _top_indices(array: np.ndarray, limit: int) -> Sequence[int]:
    limit = max(1, min(limit, array.size))
    indices = np.argpartition(array, -limit)[-limit:]
    return indices[np.argsort(array[indices])][::-1]


# Cache runtime instances so repeated predictions can reuse shared state.
_runtime_cache: dict[tuple[str, str, int], TorchMojiRuntime] = {}
_runtime_lock = Lock()


def get_runtime(weights_path: Path, vocab_path: Path, maxlen: int) -> TorchMojiRuntime:
    """Return a cached :class:`TorchMojiRuntime` for the provided resources.

    Access is thread-safe via the module-level lock guarding the cache.
    """

    weights_key = str(Path(weights_path).expanduser().resolve())
    vocab_key = str(Path(vocab_path).expanduser().resolve())
    cache_key = (weights_key, vocab_key, int(maxlen))

    with _runtime_lock:
        runtime = _runtime_cache.get(cache_key)
        if runtime is None:
            runtime = TorchMojiRuntime(Path(weights_path), Path(vocab_path), maxlen)
            _runtime_cache[cache_key] = runtime
        return runtime


__all__ = [
    "EmojiPredictionResult",
    "EmojiSelection",
    "PredictionSettings",
    "TorchMojiRuntime",
    "get_runtime",
]

