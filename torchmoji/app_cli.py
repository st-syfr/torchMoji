"""Alternative CLI that integrates with persisted TorchMoji settings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import settings as settings_module
from .cli import (
    EMOTION_BUNDLES,
    EMOJI_ALIASES,
    _emoji_from_alias,
    _ensure_file,
    _resolve_emotion_filters,
)
from .emojis import EmotionName, NonNeutralEmotionName
from .runtime import PredictionSettings, get_runtime

_HELP_URL = "https://github.com/huggingface/torchMoji#download-the-pretrained-weights"


def build_parser(current: settings_module.TorchMojiSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchmoji.app",
        description=(
            "Predict emojis for text while using defaults managed by the graphical "
            "application."
        ),
    )
    parser.add_argument(
        "text",
        metavar="TEXT",
        help="Text to emojize. Surround multi-word inputs with quotes.",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help=f"Number of emoji predictions to return (default: {current.top_k}).",
    )
    parser.add_argument(
        "--maxlen",
        dest="maxlen",
        type=int,
        default=argparse.SUPPRESS,
        metavar="N",
        help=f"Maximum tokenized length for the input text (default: {current.maxlen}).",
    )
    parser.add_argument(
        "--weights",
        dest="weights",
        type=Path,
        default=argparse.SUPPRESS,
        help=f"Path to the pretrained TorchMoji weights (default: {current.weights}).",
    )
    parser.add_argument(
        "--vocab",
        dest="vocab",
        type=Path,
        default=argparse.SUPPRESS,
        help=f"Path to the TorchMoji vocabulary JSON (default: {current.vocab}).",
    )
    parser.add_argument(
        "--scores",
        dest="scores",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Include the prediction score for each emoji.",
    )
    parser.add_argument(
        "--no-scores",
        dest="scores",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Do not display prediction scores.",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=tuple(EMOTION_BUNDLES.keys()),
        default=argparse.SUPPRESS,
        help="Emotion bundle to apply.",
    )
    parser.add_argument(
        "--emotions",
        dest="emotions",
        nargs="+",
        choices=EmotionName,
        default=argparse.SUPPRESS,
        metavar="EMOTION",
        help="Ekman core emotions to allow in the output.",
    )
    parser.add_argument(
        "--weak-emotions",
        dest="weak_emotions",
        nargs="*",
        choices=NonNeutralEmotionName,
        default=argparse.SUPPRESS,
        metavar="EMOTION",
        help="Emotions that may appear in their weak form.",
    )
    parser.add_argument(
        "--strong-emotions",
        dest="strong_emotions",
        nargs="*",
        choices=NonNeutralEmotionName,
        default=argparse.SUPPRESS,
        metavar="EMOTION",
        help="Emotions that may appear in their strong form.",
    )
    return parser


def _predict(
    text: str, settings: settings_module.TorchMojiSettings
) -> tuple[int, str]:
    weights_path = Path(settings.weights)
    vocab_path = Path(settings.vocab)

    if not _ensure_file(weights_path, "pretrained weights"):
        return 2, ""
    if not _ensure_file(vocab_path, "vocabulary file"):
        return 2, ""

    allowed_emotions, weak_emotions, strong_emotions = _resolve_emotion_filters(
        settings.mode,
        settings.emotions,
        settings.weak_emotions,
        settings.strong_emotions,
    )

    try:
        runtime = get_runtime(weights_path, vocab_path, settings.maxlen)
    except FileNotFoundError:
        print(f"Error: pretrained weights not found at '{weights_path}'.", file=sys.stderr)
        print(f"Download them following the instructions at {_HELP_URL}", file=sys.stderr)
        return 2, ""
    except json.JSONDecodeError as exc:
        print(f"Error: failed to parse vocabulary JSON: {exc}", file=sys.stderr)
        return 2, ""
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Error: could not initialise TorchMoji runtime: {exc}", file=sys.stderr)
        return 2, ""

    try:
        result = runtime.predict(
            text,
            PredictionSettings(
                top_k=settings.top_k,
                allowed_emotions=allowed_emotions,
                weak_emotions=weak_emotions,
                strong_emotions=strong_emotions,
            ),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2, ""

    probabilities = result.probabilities
    output_lines = [f"Input: {text}", "Top predictions:"]

    if probabilities.size != len(EMOJI_ALIASES):
        print(
            f"Warning: expected {len(EMOJI_ALIASES)} emoji classes but received {probabilities.size}.",
            file=sys.stderr,
        )

    for rank, selection in enumerate(result.selections, start=1):
        index = selection.index
        alias = EMOJI_ALIASES[index]
        char = _emoji_from_alias(alias)
        emotion_label = None
        ranking = selection.ranking
        if ranking is not None:
            emotion_label = ranking.emotion
            if ranking.intensity is not None:
                emotion_label = f"{emotion_label} ({ranking.intensity})"
        label_suffix = f" -> {emotion_label}" if emotion_label else ""
        if settings.scores and selection.score is not None:
            score = selection.score
            output_lines.append(
                f"{rank}. {char} {alias}{label_suffix} (score={score:.4f})"
            )
        else:
            output_lines.append(f"{rank}. {char} {alias}{label_suffix}")

    return 0, "\n".join(output_lines)


def main(argv: Sequence[str] | None = None) -> int:
    persisted = settings_module.load_settings()
    parser = build_parser(persisted)
    args = parser.parse_args(argv)

    updated_settings, touched = persisted.merge_with_namespace(args)
    exit_code, output = _predict(args.text, updated_settings)

    if output:
        print(output)

    if touched:
        settings_module.save_settings(updated_settings)

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
