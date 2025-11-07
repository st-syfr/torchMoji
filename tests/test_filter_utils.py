import test_helper
import pytest

from torchmoji.filter_utils import (
    extract_emojis,
    remove_variation_selectors,
    separate_emojis_and_text,
)


def test_remove_variation_selectors_strips_variants():
    text = "Play ⚽️ and ✌️!"
    cleaned = remove_variation_selectors(text)
    assert cleaned == "Play ⚽ and ✌!"


def test_remove_variation_selectors_preserves_text():
    text = "Cafe ☕️ break"
    cleaned = remove_variation_selectors(text)
    assert cleaned == "Cafe ☕ break"


def test_separate_emojis_and_text_with_skin_tone_and_text():
    emoji_part, text_part = separate_emojis_and_text("hi👍🏽there")
    assert emoji_part == "👍🏽"
    assert text_part == "hithere"


def test_separate_emojis_and_text_retains_joiners():
    emoji_part, text_part = separate_emojis_and_text("hi👨‍👩‍👧‍👦!")
    assert emoji_part == "👨👩👧👦"
    assert text_part == "hi\u200d\u200d\u200d!"


@pytest.mark.parametrize(
    "wanted, expected",
    [
        (["⚽️", "👍"], ["⚽", "👍"]),
        (None, ["⚽", "👍", "🏽"]),
    ],
)
def test_extract_emojis_respects_variation_selectors(wanted, expected):
    text = "Play ⚽️ or 👍🏽?"
    assert extract_emojis(text, wanted) == expected


def test_extract_emojis_preserves_order_and_duplicates():
    text = "wow 😂😂"
    assert extract_emojis(text, None) == ["😂", "😂"]
