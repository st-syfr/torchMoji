"""Command line interface for TorchMoji."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import emoji
import numpy as np
import torch

from .emojis import EMOJI_ALIASES
from .global_variables import PRETRAINED_PATH, VOCAB_PATH
from .model_def import torchmoji_emojis
from .sentence_tokenizer import SentenceTokenizer

_HELP_URL = "https://github.com/huggingface/torchMoji#download-the-pretrained-weights"


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
    emojize_parser.set_defaults(func=_run_emojize)

    return parser


def _load_vocabulary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_tokenizer(vocabulary: dict, maxlen: int) -> SentenceTokenizer:
    return SentenceTokenizer(vocabulary, maxlen)


def _load_model(weights_path: Path):
    return torchmoji_emojis(str(weights_path))


def _emoji_from_alias(alias: str) -> str:
    try:
        return emoji.emojize(alias, language="alias")
    except TypeError:
        # emoji<2 uses the use_aliases flag instead of the language argument
        return emoji.emojize(alias, use_aliases=True)


def _top_indices(array: np.ndarray, limit: int) -> Sequence[int]:
    limit = max(1, min(limit, array.size))
    indices = np.argpartition(array, -limit)[-limit:]
    return indices[np.argsort(array[indices])][::-1]


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

    try:
        vocabulary = _load_vocabulary(vocab_path)
    except FileNotFoundError:
        print(f"Error: could not open vocabulary file '{vocab_path}'.", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Error: failed to parse vocabulary JSON: {exc}", file=sys.stderr)
        return 2

    tokenizer = _build_tokenizer(vocabulary, args.maxlen)

    try:
        tokenized, _, _ = tokenizer.tokenize_sentences([args.text])
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Error tokenizing text: {exc}", file=sys.stderr)
        return 2

    try:
        model = _load_model(weights_path)
    except FileNotFoundError:
        print(f"Error: pretrained weights not found at '{weights_path}'.", file=sys.stderr)
        print(f"Download them following the instructions at {_HELP_URL}", file=sys.stderr)
        return 2

    model.eval()
    with torch.no_grad():
        probabilities = model(tokenized)[0]
    if hasattr(probabilities, "cpu"):
        probabilities = probabilities.cpu()
    if hasattr(probabilities, "numpy"):
        probabilities = probabilities.numpy()

    probabilities = np.asarray(probabilities, dtype=np.float32)

    if probabilities.size != len(EMOJI_ALIASES):
        print(
            f"Warning: expected {len(EMOJI_ALIASES)} emoji classes but received {probabilities.size}.",
            file=sys.stderr,
        )

    top_indices = _top_indices(probabilities, args.top_k)

    print(f"Input: {args.text}")
    print("Top predictions:")
    for rank, index in enumerate(top_indices, start=1):
        alias = EMOJI_ALIASES[index]
        char = _emoji_from_alias(alias)
        if args.scores and index < probabilities.size:
            score = float(probabilities[index])
            print(f"{rank}. {char} {alias} (score={score:.4f})")
        else:
            print(f"{rank}. {char} {alias}")

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
