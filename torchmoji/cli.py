"""Command line interface for TorchMoji."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import emoji

from .emojis import EMOJI_ALIASES, EmotionName, NonNeutralEmotionName
from .global_variables import PRETRAINED_PATH, VOCAB_PATH
from .runtime import PredictionSettings, get_runtime

_HELP_URL = "https://github.com/huggingface/torchMoji#download-the-pretrained-weights"


EMOTION_BUNDLES: dict[str, dict[str, Iterable[str] | None]] = {
    "simple": {
        "emotions": ("happiness", "sadness", "fear", "anger", "neutral"),
        "weak": ("happiness", "sadness", "fear", "anger"),
        "strong": ("happiness", "sadness", "fear", "anger"),
    },
    "standard": {
        "emotions": EmotionName,
        "weak": NonNeutralEmotionName,
        "strong": NonNeutralEmotionName,
    },
}


def _resolve_emotion_filters(
    mode: str,
    emotions: Iterable[str] | None,
    weak_emotions: Iterable[str] | None,
    strong_emotions: Iterable[str] | None,
) -> tuple[list[str], list[str] | None, list[str] | None]:
    """Resolve the effective emotion filters based on CLI arguments."""

    bundle = EMOTION_BUNDLES.get(mode, EMOTION_BUNDLES["standard"])

    allowed_emotions = list(emotions) if emotions is not None else list(bundle["emotions"])

    if weak_emotions is not None:
        resolved_weak = list(weak_emotions)
    else:
        weak_bundle = bundle["weak"]
        resolved_weak = list(weak_bundle) if weak_bundle is not None else None

    if strong_emotions is not None:
        resolved_strong = list(strong_emotions)
    else:
        strong_bundle = bundle["strong"]
        resolved_strong = list(strong_bundle) if strong_bundle is not None else None

    return allowed_emotions, resolved_weak, resolved_strong


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchmoji",
        description="Utilities for working with the TorchMoji model.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    emojize_parser = subparsers.add_parser(
        "emojize",
        help="Predict emojis for the provided text",
        description="Predict the most likely emojis for a piece of text using the pretrained TorchMoji model.",
    )
    emojize_parser.add_argument(
        "text",
        metavar="TEXT",
        help="Text to emojize. Surround multi-word inputs with quotes.",
    )
    emojize_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of emoji predictions to return (default: 5).",
    )
    emojize_parser.add_argument(
        "--maxlen",
        type=int,
        default=30,
        metavar="N",
        help="Maximum tokenized length for the input text (default: 30).",
    )
    emojize_parser.add_argument(
        "--weights",
        type=Path,
        default=Path(PRETRAINED_PATH),
        help="Path to the pretrained TorchMoji weights (default: %(default)s).",
    )
    emojize_parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(VOCAB_PATH),
        help="Path to the TorchMoji vocabulary JSON (default: %(default)s).",
    )
    emojize_parser.add_argument(
        "--scores",
        action="store_true",
        help="Include the prediction score for each emoji.",
    )
    emojize_parser.add_argument(
        "--mode",
        choices=tuple(EMOTION_BUNDLES.keys()),
        default="standard",
        help=(
            "Emotion bundle to apply. 'simple' exposes neutral plus happiness, fear, sadness,"
            " and anger in weak and strong forms while 'standard' includes all emotions."
        ),
    )
    emojize_parser.add_argument(
        "--emotions",
        nargs="+",
        choices=EmotionName,
        default=None,
        metavar="EMOTION",
        help=(
            "Ekman core emotions to allow in the output. Defaults to the selected mode."
        ),
    )
    emojize_parser.add_argument(
        "--weak-emotions",
        nargs="*",
        choices=NonNeutralEmotionName,
        metavar="EMOTION",
        help=(
            "Emotions that may appear in their weak form. Defaults follow the selected mode."
        ),
    )
    emojize_parser.add_argument(
        "--strong-emotions",
        nargs="*",
        choices=NonNeutralEmotionName,
        metavar="EMOTION",
        help=(
            "Emotions that may appear in their strong form. Defaults follow the selected mode."
        ),
    )
    emojize_parser.set_defaults(func=_run_emojize)

    return parser
def _emoji_from_alias(alias: str) -> str:
    try:
        return emoji.emojize(alias, language="alias")
    except TypeError:
        # emoji<2 uses the use_aliases flag instead of the language argument
        return emoji.emojize(alias, use_aliases=True)


def _ensure_file(path: Path, description: str) -> bool:
    if path.exists():
        return True
    print(
        f"Error: could not find {description} at '{path}'.",
        file=sys.stderr,
    )
    if description == "pretrained weights":
        print(
            f"Download them following the instructions at {_HELP_URL}",
            file=sys.stderr,
        )
    return False


def _run_emojize(args: argparse.Namespace) -> int:
    weights_path = args.weights
    vocab_path = args.vocab

    if not _ensure_file(weights_path, "pretrained weights"):
        return 2
    if not _ensure_file(vocab_path, "vocabulary file"):
        return 2

    allowed_emotions, weak_emotions, strong_emotions = _resolve_emotion_filters(
        args.mode,
        args.emotions,
        args.weak_emotions,
        args.strong_emotions,
    )

    try:
        runtime = get_runtime(weights_path, vocab_path, args.maxlen)
    except FileNotFoundError:
        print(f"Error: pretrained weights not found at '{weights_path}'.", file=sys.stderr)
        print(f"Download them following the instructions at {_HELP_URL}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Error: failed to parse vocabulary JSON: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: could not initialise TorchMoji runtime: {exc}", file=sys.stderr)
        return 2

    try:
        result = runtime.predict(
            args.text,
            PredictionSettings(
                top_k=args.top_k,
                allowed_emotions=allowed_emotions,
                weak_emotions=weak_emotions,
                strong_emotions=strong_emotions,
            ),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    probabilities = result.probabilities

    if probabilities.size != len(EMOJI_ALIASES):
        print(
            f"Warning: expected {len(EMOJI_ALIASES)} emoji classes but received {probabilities.size}.",
            file=sys.stderr,
        )

    print(f"Input: {args.text}")
    print("Top predictions:")
    for rank, selection in enumerate(result.selections, start=1):
        index = selection.index
        ranking = selection.ranking
        alias = EMOJI_ALIASES[index]
        char = _emoji_from_alias(alias)
        emotion_label = None
        if ranking is not None:
            emotion_label = ranking.emotion
            if ranking.intensity is not None:
                emotion_label = f"{emotion_label} ({ranking.intensity})"
        label_suffix = f" -> {emotion_label}" if emotion_label else ""
        if args.scores and selection.score is not None:
            score = selection.score
            print(f"{rank}. {char} {alias}{label_suffix} (score={score:.4f})")
        else:
            print(f"{rank}. {char} {alias}{label_suffix}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
