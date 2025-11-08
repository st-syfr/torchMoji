"""Emoji aliases and Ekman emotion metadata for the TorchMoji model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Sequence

EMOJI_ALIASES = [
    ":joy:",
    ":unamused:",
    ":weary:",
    ":sob:",
    ":heart_eyes:",
    ":pensive:",
    ":ok_hand:",
    ":blush:",
    ":heart:",
    ":smirk:",
    ":grin:",
    ":notes:",
    ":flushed:",
    ":100:",
    ":sleeping:",
    ":relieved:",
    ":relaxed:",
    ":raised_hands:",
    ":two_hearts:",
    ":expressionless:",
    ":sweat_smile:",
    ":pray:",
    ":confused:",
    ":kissing_heart:",
    ":heartbeat:",
    ":neutral_face:",
    ":information_desk_person:",
    ":disappointed:",
    ":see_no_evil:",
    ":tired_face:",
    ":v:",
    ":sunglasses:",
    ":rage:",
    ":thumbsup:",
    ":cry:",
    ":sleepy:",
    ":yum:",
    ":triumph:",
    ":hand:",
    ":mask:",
    ":clap:",
    ":eyes:",
    ":gun:",
    ":persevere:",
    ":smiling_imp:",
    ":sweat:",
    ":broken_heart:",
    ":yellow_heart:",
    ":musical_note:",
    ":speak_no_evil:",
    ":wink:",
    ":skull:",
    ":confounded:",
    ":smile:",
    ":stuck_out_tongue_winking_eye:",
    ":angry:",
    ":no_good:",
    ":muscle:",
    ":facepunch:",
    ":purple_heart:",
    ":sparkling_heart:",
    ":blue_heart:",
    ":grimacing:",
    ":sparkles:",
]


EmotionName = (
    "happiness",
    "sadness",
    "fear",
    "disgust",
    "anger",
    "surprise",
    "contempt",
    "neutral",
)

NonNeutralEmotionName = tuple(name for name in EmotionName if name != "neutral")


@dataclass(frozen=True)
class EmotionRanking:
    """Ranking information for mapping an emoji to an Ekman core emotion."""

    emotion: str
    intensity: str | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        if self.emotion not in EmotionName:
            raise ValueError(f"Unknown emotion '{self.emotion}'.")
        if self.emotion == "neutral":
            if self.intensity is not None:
                raise ValueError("Neutral emotion cannot have an intensity.")
        elif self.intensity not in {"flat", "weak", "strong"}:
            raise ValueError(f"Invalid intensity '{self.intensity}'.")
        if self.weight is not None and not (0.0 <= self.weight <= 1.0):
            raise ValueError("Emotion weights must fall within the [0, 1] range.")


EmotionWeights = Mapping[str, float]


def _validate_weights(weights: EmotionWeights) -> dict[str, float]:
    """Normalise and validate emotion weight mappings."""

    missing = [emotion for emotion in EmotionName if emotion not in weights]
    if missing:
        raise ValueError(
            f"Emotion weight definition is missing: {', '.join(sorted(missing))}."
        )

    normalised = {emotion: float(weight) for emotion, weight in weights.items()}
    total = sum(normalised.values())
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Emotion weights must sum to 1.0, received {total:.6f}."
        )

    for emotion, weight in normalised.items():
        if weight < 0.0:
            raise ValueError(
                f"Emotion '{emotion}' has a negative weight ({weight:.6f})."
            )

    return normalised


def _intensity_from_weight(weight: float) -> str:
    if weight >= 0.45:
        return "strong"
    if weight >= 0.18:
        return "weak"
    return "flat"


def _build_rankings(weights: EmotionWeights) -> tuple[EmotionRanking, ...]:
    """Convert a weight distribution into ordered emotion rankings."""

    normalised = _validate_weights(weights)
    ordered = sorted(normalised.items(), key=lambda item: item[1], reverse=True)

    rankings: list[EmotionRanking] = []
    strong_found = False
    weak_found = False

    for emotion, weight in ordered:
        if emotion == "neutral":
            rankings.append(EmotionRanking(emotion, None, weight))
            continue

        intensity = _intensity_from_weight(weight)
        if intensity == "strong":
            strong_found = True
        elif intensity == "weak":
            weak_found = True
        rankings.append(EmotionRanking(emotion, intensity, weight))

    for index, ranking in enumerate(rankings):
        if ranking.emotion == "neutral":
            continue
        if not strong_found:
            rankings[index] = EmotionRanking(ranking.emotion, "strong", ranking.weight)
            strong_found = True
        elif not weak_found and ranking.intensity == "flat":
            rankings[index] = EmotionRanking(ranking.emotion, "weak", ranking.weight)
            weak_found = True
        if strong_found and weak_found:
            break

    return tuple(rankings)


EMOTION_WEIGHT_TEMPLATES: Dict[str, Dict[str, float]] = {
    "joy_hysterical": {
        "happiness": 0.58,
        "sadness": 0.04,
        "fear": 0.03,
        "disgust": 0.04,
        "anger": 0.04,
        "surprise": 0.18,
        "contempt": 0.04,
        "neutral": 0.05,
    },
    "joy_playful": {
        "happiness": 0.50,
        "sadness": 0.04,
        "fear": 0.03,
        "disgust": 0.04,
        "anger": 0.04,
        "surprise": 0.20,
        "contempt": 0.10,
        "neutral": 0.05,
    },
    "joy_content": {
        "happiness": 0.55,
        "sadness": 0.05,
        "fear": 0.04,
        "disgust": 0.03,
        "anger": 0.03,
        "surprise": 0.10,
        "contempt": 0.05,
        "neutral": 0.15,
    },
    "joy_relief": {
        "happiness": 0.45,
        "sadness": 0.04,
        "fear": 0.20,
        "disgust": 0.03,
        "anger": 0.03,
        "surprise": 0.15,
        "contempt": 0.02,
        "neutral": 0.08,
    },
    "love_romantic": {
        "happiness": 0.60,
        "sadness": 0.05,
        "fear": 0.03,
        "disgust": 0.02,
        "anger": 0.02,
        "surprise": 0.15,
        "contempt": 0.03,
        "neutral": 0.10,
    },
    "gratitude_hope": {
        "happiness": 0.40,
        "sadness": 0.12,
        "fear": 0.12,
        "disgust": 0.05,
        "anger": 0.06,
        "surprise": 0.08,
        "contempt": 0.07,
        "neutral": 0.10,
    },
    "broken_heart": {
        "happiness": 0.04,
        "sadness": 0.60,
        "fear": 0.12,
        "disgust": 0.06,
        "anger": 0.10,
        "surprise": 0.04,
        "contempt": 0.02,
        "neutral": 0.02,
    },
    "sadness_intense": {
        "happiness": 0.03,
        "sadness": 0.58,
        "fear": 0.18,
        "disgust": 0.05,
        "anger": 0.06,
        "surprise": 0.05,
        "contempt": 0.02,
        "neutral": 0.03,
    },
    "sadness_melancholy": {
        "happiness": 0.05,
        "sadness": 0.45,
        "fear": 0.15,
        "disgust": 0.05,
        "anger": 0.08,
        "surprise": 0.05,
        "contempt": 0.05,
        "neutral": 0.12,
    },
    "sadness_exhausted": {
        "happiness": 0.04,
        "sadness": 0.50,
        "fear": 0.18,
        "disgust": 0.05,
        "anger": 0.08,
        "surprise": 0.04,
        "contempt": 0.04,
        "neutral": 0.07,
    },
    "tired_neutral": {
        "happiness": 0.08,
        "sadness": 0.20,
        "fear": 0.08,
        "disgust": 0.05,
        "anger": 0.04,
        "surprise": 0.07,
        "contempt": 0.03,
        "neutral": 0.45,
    },
    "sarcasm_disdain": {
        "happiness": 0.07,
        "sadness": 0.05,
        "fear": 0.03,
        "disgust": 0.12,
        "anger": 0.18,
        "surprise": 0.05,
        "contempt": 0.45,
        "neutral": 0.05,
    },
    "sarcasm_side_eye": {
        "happiness": 0.04,
        "sadness": 0.12,
        "fear": 0.04,
        "disgust": 0.08,
        "anger": 0.12,
        "surprise": 0.05,
        "contempt": 0.35,
        "neutral": 0.20,
    },
    "cool_confident": {
        "happiness": 0.40,
        "sadness": 0.05,
        "fear": 0.04,
        "disgust": 0.05,
        "anger": 0.06,
        "surprise": 0.10,
        "contempt": 0.20,
        "neutral": 0.10,
    },
    "celebration": {
        "happiness": 0.52,
        "sadness": 0.03,
        "fear": 0.03,
        "disgust": 0.04,
        "anger": 0.07,
        "surprise": 0.20,
        "contempt": 0.05,
        "neutral": 0.06,
    },
    "positive_gesture": {
        "happiness": 0.48,
        "sadness": 0.05,
        "fear": 0.04,
        "disgust": 0.04,
        "anger": 0.05,
        "surprise": 0.12,
        "contempt": 0.04,
        "neutral": 0.18,
    },
    "music": {
        "happiness": 0.50,
        "sadness": 0.05,
        "fear": 0.04,
        "disgust": 0.03,
        "anger": 0.03,
        "surprise": 0.15,
        "contempt": 0.05,
        "neutral": 0.15,
    },
    "attention": {
        "happiness": 0.16,
        "sadness": 0.04,
        "fear": 0.05,
        "disgust": 0.03,
        "anger": 0.08,
        "surprise": 0.32,
        "contempt": 0.20,
        "neutral": 0.12,
    },
    "surprise_embarrassed": {
        "happiness": 0.12,
        "sadness": 0.08,
        "fear": 0.20,
        "disgust": 0.07,
        "anger": 0.05,
        "surprise": 0.35,
        "contempt": 0.05,
        "neutral": 0.08,
    },
    "surprise_shock": {
        "happiness": 0.04,
        "sadness": 0.10,
        "fear": 0.30,
        "disgust": 0.15,
        "anger": 0.07,
        "surprise": 0.25,
        "contempt": 0.04,
        "neutral": 0.05,
    },
    "confusion": {
        "happiness": 0.08,
        "sadness": 0.15,
        "fear": 0.15,
        "disgust": 0.07,
        "anger": 0.07,
        "surprise": 0.30,
        "contempt": 0.08,
        "neutral": 0.10,
    },
    "struggle": {
        "happiness": 0.03,
        "sadness": 0.38,
        "fear": 0.25,
        "disgust": 0.08,
        "anger": 0.12,
        "surprise": 0.06,
        "contempt": 0.03,
        "neutral": 0.05,
    },
    "sick": {
        "happiness": 0.05,
        "sadness": 0.15,
        "fear": 0.20,
        "disgust": 0.30,
        "anger": 0.08,
        "surprise": 0.07,
        "contempt": 0.05,
        "neutral": 0.10,
    },
    "skull_laugh": {
        "happiness": 0.50,
        "sadness": 0.03,
        "fear": 0.03,
        "disgust": 0.02,
        "anger": 0.05,
        "surprise": 0.20,
        "contempt": 0.12,
        "neutral": 0.05,
    },
    "anger_fury": {
        "happiness": 0.01,
        "sadness": 0.04,
        "fear": 0.05,
        "disgust": 0.18,
        "anger": 0.55,
        "surprise": 0.03,
        "contempt": 0.12,
        "neutral": 0.02,
    },
    "anger_frustrated": {
        "happiness": 0.03,
        "sadness": 0.08,
        "fear": 0.10,
        "disgust": 0.12,
        "anger": 0.45,
        "surprise": 0.05,
        "contempt": 0.15,
        "neutral": 0.02,
    },
    "determined": {
        "happiness": 0.32,
        "sadness": 0.03,
        "fear": 0.04,
        "disgust": 0.03,
        "anger": 0.28,
        "surprise": 0.12,
        "contempt": 0.10,
        "neutral": 0.08,
    },
    "dark_frustration": {
        "happiness": 0.03,
        "sadness": 0.30,
        "fear": 0.15,
        "disgust": 0.10,
        "anger": 0.25,
        "surprise": 0.07,
        "contempt": 0.05,
        "neutral": 0.05,
    },
    "refusal": {
        "happiness": 0.03,
        "sadness": 0.08,
        "fear": 0.05,
        "disgust": 0.32,
        "anger": 0.25,
        "surprise": 0.06,
        "contempt": 0.18,
        "neutral": 0.03,
    },
    "mischief": {
        "happiness": 0.35,
        "sadness": 0.05,
        "fear": 0.04,
        "disgust": 0.04,
        "anger": 0.10,
        "surprise": 0.12,
        "contempt": 0.22,
        "neutral": 0.08,
    },
}

_EMOJI_TO_TEMPLATE: Dict[str, str] = {
    ":joy:": "joy_hysterical",
    ":unamused:": "sarcasm_disdain",
    ":weary:": "sadness_exhausted",
    ":sob:": "sadness_intense",
    ":heart_eyes:": "love_romantic",
    ":pensive:": "sadness_melancholy",
    ":ok_hand:": "positive_gesture",
    ":blush:": "joy_content",
    ":heart:": "love_romantic",
    ":smirk:": "sarcasm_disdain",
    ":grin:": "joy_hysterical",
    ":notes:": "music",
    ":flushed:": "surprise_embarrassed",
    ":100:": "celebration",
    ":sleeping:": "tired_neutral",
    ":relieved:": "joy_relief",
    ":relaxed:": "joy_content",
    ":raised_hands:": "celebration",
    ":two_hearts:": "love_romantic",
    ":expressionless:": "sarcasm_side_eye",
    ":sweat_smile:": "joy_relief",
    ":pray:": "gratitude_hope",
    ":confused:": "confusion",
    ":kissing_heart:": "love_romantic",
    ":heartbeat:": "love_romantic",
    ":neutral_face:": "sarcasm_side_eye",
    ":information_desk_person:": "sarcasm_disdain",
    ":disappointed:": "sadness_melancholy",
    ":see_no_evil:": "surprise_embarrassed",
    ":tired_face:": "sadness_exhausted",
    ":v:": "positive_gesture",
    ":sunglasses:": "cool_confident",
    ":rage:": "anger_fury",
    ":thumbsup:": "positive_gesture",
    ":cry:": "sadness_intense",
    ":sleepy:": "tired_neutral",
    ":yum:": "joy_content",
    ":triumph:": "anger_frustrated",
    ":hand:": "positive_gesture",
    ":mask:": "sick",
    ":clap:": "celebration",
    ":eyes:": "attention",
    ":gun:": "dark_frustration",
    ":persevere:": "struggle",
    ":smiling_imp:": "mischief",
    ":sweat:": "struggle",
    ":broken_heart:": "broken_heart",
    ":yellow_heart:": "love_romantic",
    ":musical_note:": "music",
    ":speak_no_evil:": "surprise_embarrassed",
    ":wink:": "joy_playful",
    ":skull:": "skull_laugh",
    ":confounded:": "struggle",
    ":smile:": "joy_content",
    ":stuck_out_tongue_winking_eye:": "joy_playful",
    ":angry:": "anger_fury",
    ":no_good:": "refusal",
    ":muscle:": "determined",
    ":facepunch:": "anger_frustrated",
    ":purple_heart:": "love_romantic",
    ":sparkling_heart:": "love_romantic",
    ":blue_heart:": "love_romantic",
    ":grimacing:": "surprise_shock",
    ":sparkles:": "celebration",
}

_NORMALISED_TEMPLATES = {
    name: _validate_weights(weights)
    for name, weights in EMOTION_WEIGHT_TEMPLATES.items()
}

EMOJI_EMOTION_WEIGHTS: Dict[str, Dict[str, float]] = {}
for alias in EMOJI_ALIASES:
    template_name = _EMOJI_TO_TEMPLATE.get(alias)
    if template_name is None:
        raise ValueError(f"Missing emotion template for alias '{alias}'.")
    EMOJI_EMOTION_WEIGHTS[alias] = dict(_NORMALISED_TEMPLATES[template_name])

_missing_aliases = set(EMOJI_ALIASES) - set(EMOJI_EMOTION_WEIGHTS)
if _missing_aliases:
    raise ValueError(
        "Emotion weight mapping does not cover every emoji alias: "
        + ", ".join(sorted(_missing_aliases))
    )

EMOJI_EMOTION_RANKS: Dict[str, tuple[EmotionRanking, ...]] = {
    alias: _build_rankings(weights)
    for alias, weights in EMOJI_EMOTION_WEIGHTS.items()
}


def get_emotion_rankings(alias: str) -> tuple[EmotionRanking, ...]:
    """Return the ordered emotion rankings for an emoji alias."""

    return EMOJI_EMOTION_RANKS[alias]


def iter_emotion_rankings() -> Iterator[tuple[str, tuple[EmotionRanking, ...]]]:
    """Iterate through aliases paired with their emotion rankings."""

    for alias in EMOJI_ALIASES:
        yield alias, EMOJI_EMOTION_RANKS[alias]


def select_accessible_ranking(
    alias: str,
    allowed_emotions: Iterable[str],
    weak_emotions: Iterable[str] | None = None,
    strong_emotions: Iterable[str] | None = None,
) -> EmotionRanking | None:
    """Return the highest ranked emotion the alias can map to under the filters."""

    allowed = set(allowed_emotions)
    weak_allowed = set(weak_emotions) if weak_emotions is not None else set(NonNeutralEmotionName)
    strong_allowed = set(strong_emotions) if strong_emotions is not None else set(NonNeutralEmotionName)

    for ranking in get_emotion_rankings(alias):
        if ranking.emotion not in allowed:
            continue
        if ranking.intensity is None or ranking.intensity == "flat":
            return ranking
        if ranking.intensity == "weak" and ranking.emotion in weak_allowed:
            return ranking
        if ranking.intensity == "strong" and ranking.emotion in strong_allowed:
            return ranking
    return None


def filter_emojis_by_emotion(
    probabilities: Sequence[float],
    limit: int,
    allowed_emotions: Iterable[str],
    weak_emotions: Iterable[str] | None = None,
    strong_emotions: Iterable[str] | None = None,
) -> list[tuple[int, EmotionRanking]]:
    """Select the highest probability emoji indices under the provided emotion filters."""

    if len(probabilities) != len(EMOJI_ALIASES):
        raise ValueError(
            "Probability count does not match number of emoji aliases; cannot filter by emotion."
        )

    limit = max(1, min(limit, len(EMOJI_ALIASES)))
    ordered_indices = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)

    selections: list[tuple[int, EmotionRanking]] = []
    for index in ordered_indices:
        ranking = select_accessible_ranking(
            EMOJI_ALIASES[index],
            allowed_emotions,
            weak_emotions,
            strong_emotions,
        )
        if ranking is None:
            continue
        selections.append((index, ranking))
        if len(selections) >= limit:
            break
    return selections


__all__ = [
    "EMOJI_ALIASES",
    "EmotionRanking",
    "EmotionName",
    "NonNeutralEmotionName",
    "EMOTION_WEIGHT_TEMPLATES",
    "EMOJI_EMOTION_WEIGHTS",
    "EMOJI_EMOTION_RANKS",
    "filter_emojis_by_emotion",
    "get_emotion_rankings",
    "iter_emotion_rankings",
    "select_accessible_ranking",
]
