from __future__ import annotations

import pytest

from torchmoji.emojis import (
    EMOJI_EMOTION_WEIGHTS,
    EmotionName,
    get_emotion_rankings,
    select_accessible_ranking,
)


def test_emotion_rankings_include_flat_intensity():
    rankings = get_emotion_rankings(":joy:")
    non_neutral_intensities = {
        ranking.intensity for ranking in rankings if ranking.emotion != "neutral"
    }

    assert "flat" in non_neutral_intensities
    assert "strong" in non_neutral_intensities


def test_select_accessible_ranking_returns_flat_when_only_option():
    ranking = select_accessible_ranking(
        ":joy:",
        ["sadness"],
        weak_emotions=[],
        strong_emotions=[],
    )

    assert ranking is not None
    assert ranking.emotion == "sadness"
    assert ranking.intensity == "flat"


def test_emotion_weight_distributions_are_normalised():
    for alias, weights in EMOJI_EMOTION_WEIGHTS.items():
        assert set(weights) == set(EmotionName)
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_emotion_rankings_expose_weight_information():
    rankings = get_emotion_rankings(":joy:")
    total = sum(ranking.weight for ranking in rankings if ranking.weight is not None)

    assert total == pytest.approx(1.0, rel=1e-6, abs=1e-6)
