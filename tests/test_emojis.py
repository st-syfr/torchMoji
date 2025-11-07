from __future__ import annotations

from torchmoji.emojis import get_emotion_rankings, select_accessible_ranking


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
