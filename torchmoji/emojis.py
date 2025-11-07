"""Emoji aliases and Ekman emotion metadata for the TorchMoji model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Sequence

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

    def __post_init__(self) -> None:
        if self.emotion not in EmotionName:
            raise ValueError(f"Unknown emotion '{self.emotion}'.")
        if self.emotion == "neutral":
            if self.intensity is not None:
                raise ValueError("Neutral emotion cannot have an intensity.")
            return
        if self.intensity not in {"flat", "weak", "strong"}:
            raise ValueError(f"Invalid intensity '{self.intensity}'.")


def _apply_intensity_scheme(rankings: Sequence[EmotionRanking]) -> tuple[EmotionRanking, ...]:
    """Assign flat/weak/strong intensities based on relative ranking order."""

    result: list[EmotionRanking] = []
    non_neutral_seen = 0
    for ranking in rankings:
        if ranking.emotion == "neutral":
            result.append(EmotionRanking("neutral", None))
            continue

        non_neutral_seen += 1
        if non_neutral_seen <= 2:
            intensity = ranking.intensity if ranking.intensity is not None else "weak"
        else:
            intensity = "flat"
        result.append(EmotionRanking(ranking.emotion, intensity))
    return tuple(result)


def _ranking(
    *rankings: tuple[str, str | None] | EmotionRanking,
) -> tuple[EmotionRanking, ...]:
    ordered: list[EmotionRanking] = []
    seen: set[str] = set()
    for value in rankings:
        ranking = value if isinstance(value, EmotionRanking) else EmotionRanking(*value)
        if ranking.emotion in seen:
            raise ValueError(f"Duplicate emotion '{ranking.emotion}' in ranking.")
        seen.add(ranking.emotion)
        ordered.append(ranking)
    missing = {emotion for emotion in EmotionName if emotion not in seen}
    if missing:
        raise ValueError(f"Missing emotions in ranking: {', '.join(sorted(missing))}.")
    return _apply_intensity_scheme(ordered)


EMOJI_EMOTION_RANKS: Dict[str, tuple[EmotionRanking, ...]] = {
    ":joy:": _ranking(
        ("happiness", "strong"),
        ("surprise", "strong"),
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":unamused:": _ranking(
        ("contempt", "strong"),
        ("anger", "weak"),
        ("disgust", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("surprise", "weak"),
        ("happiness", "weak"),
    ),
    ":weary:": _ranking(
        ("sadness", "strong"),
        ("fear", "strong"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":sob:": _ranking(
        ("sadness", "strong"),
        ("fear", "strong"),
        ("anger", "weak"),
        ("disgust", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":heart_eyes:": _ranking(
        ("happiness", "strong"),
        ("surprise", "strong"),
        ("neutral", None),
        ("contempt", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
    ),
    ":pensive:": _ranking(
        ("sadness", "weak"),
        ("neutral", None),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("happiness", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
        ("surprise", "weak"),
    ),
    ":ok_hand:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("contempt", "weak"),
        ("anger", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
    ),
    ":blush:": _ranking(
        ("happiness", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("contempt", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
    ),
    ":heart:": _ranking(
        ("happiness", "strong"),
        ("neutral", None),
        ("sadness", "weak"),
        ("surprise", "weak"),
        ("contempt", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
    ),
    ":smirk:": _ranking(
        ("contempt", "strong"),
        ("happiness", "weak"),
        ("anger", "weak"),
        ("disgust", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("sadness", "weak"),
    ),
    ":grin:": _ranking(
        ("happiness", "strong"),
        ("surprise", "weak"),
        ("neutral", None),
        ("contempt", "weak"),
        ("anger", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("sadness", "weak"),
    ),
    ":notes:": _ranking(
        ("happiness", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("contempt", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
    ),
    ":flushed:": _ranking(
        ("surprise", "strong"),
        ("fear", "strong"),
        ("happiness", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":100:": _ranking(
        ("happiness", "strong"),
        ("anger", "strong"),
        ("surprise", "weak"),
        ("neutral", None),
        ("contempt", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
    ),
    ":sleeping:": _ranking(
        ("neutral", None),
        ("sadness", "weak"),
        ("happiness", "weak"),
        ("fear", "weak"),
        ("surprise", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":relieved:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("sadness", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":relaxed:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":raised_hands:": _ranking(
        ("happiness", "strong"),
        ("surprise", "strong"),
        ("neutral", None),
        ("anger", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("contempt", "weak"),
    ),
    ":two_hearts:": _ranking(
        ("happiness", "strong"),
        ("neutral", None),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":expressionless:": _ranking(
        ("neutral", None),
        ("contempt", "weak"),
        ("sadness", "weak"),
        ("anger", "weak"),
        ("disgust", "weak"),
        ("fear", "weak"),
        ("surprise", "weak"),
        ("happiness", "weak"),
    ),
    ":sweat_smile:": _ranking(
        ("happiness", "weak"),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":pray:": _ranking(
        ("sadness", "weak"),
        ("fear", "weak"),
        ("happiness", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":confused:": _ranking(
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":kissing_heart:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":heartbeat:": _ranking(
        ("happiness", "strong"),
        ("neutral", None),
        ("sadness", "weak"),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":neutral_face:": _ranking(
        ("neutral", None),
        ("sadness", "weak"),
        ("contempt", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("happiness", "weak"),
        ("surprise", "weak"),
    ),
    ":information_desk_person:": _ranking(
        ("contempt", "strong"),
        ("happiness", "weak"),
        ("neutral", None),
        ("anger", "weak"),
        ("disgust", "weak"),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("sadness", "weak"),
    ),
    ":disappointed:": _ranking(
        ("sadness", "strong"),
        ("neutral", None),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("surprise", "weak"),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":see_no_evil:": _ranking(
        ("surprise", "weak"),
        ("fear", "strong"),
        ("happiness", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":tired_face:": _ranking(
        ("sadness", "strong"),
        ("neutral", None),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("surprise", "weak"),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":v:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("anger", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("contempt", "weak"),
    ),
    ":sunglasses:": _ranking(
        ("happiness", "weak"),
        ("contempt", "strong"),
        ("neutral", None),
        ("anger", "weak"),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
    ),
    ":rage:": _ranking(
        ("anger", "strong"),
        ("disgust", "strong"),
        ("contempt", "strong"),
        ("fear", "weak"),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("happiness", "weak"),
        ("neutral", None),
    ),
    ":thumbsup:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("contempt", "weak"),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
    ),
    ":cry:": _ranking(
        ("sadness", "strong"),
        ("fear", "strong"),
        ("neutral", None),
        ("anger", "weak"),
        ("surprise", "weak"),
        ("disgust", "weak"),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":sleepy:": _ranking(
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("surprise", "weak"),
        ("happiness", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":yum:": _ranking(
        ("happiness", "weak"),
        ("surprise", "weak"),
        ("disgust", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":triumph:": _ranking(
        ("anger", "strong"),
        ("happiness", "strong"),
        ("contempt", "strong"),
        ("surprise", "weak"),
        ("disgust", "weak"),
        ("fear", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
    ),
    ":hand:": _ranking(
        ("neutral", None),
        ("anger", "weak"),
        ("contempt", "weak"),
        ("disgust", "weak"),
        ("surprise", "weak"),
        ("happiness", "weak"),
        ("fear", "weak"),
        ("sadness", "weak"),
    ),
    ":mask:": _ranking(
        ("fear", "strong"),
        ("disgust", "strong"),
        ("sadness", "weak"),
        ("neutral", None),
        ("anger", "weak"),
        ("surprise", "weak"),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":clap:": _ranking(
        ("happiness", "strong"),
        ("surprise", "strong"),
        ("neutral", None),
        ("contempt", "weak"),
        ("anger", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
    ),
    ":eyes:": _ranking(
        ("surprise", "weak"),
        ("fear", "weak"),
        ("neutral", None),
        ("anger", "weak"),
        ("disgust", "weak"),
        ("contempt", "weak"),
        ("sadness", "weak"),
        ("happiness", "weak"),
    ),
    ":gun:": _ranking(
        ("anger", "strong"),
        ("fear", "strong"),
        ("disgust", "strong"),
        ("contempt", "strong"),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
        ("happiness", "weak"),
    ),
    ":persevere:": _ranking(
        ("sadness", "strong"),
        ("anger", "weak"),
        ("fear", "weak"),
        ("neutral", None),
        ("disgust", "weak"),
        ("surprise", "weak"),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":smiling_imp:": _ranking(
        ("anger", "strong"),
        ("contempt", "strong"),
        ("happiness", "weak"),
        ("disgust", "weak"),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
    ),
    ":sweat:": _ranking(
        ("fear", "weak"),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":broken_heart:": _ranking(
        ("sadness", "strong"),
        ("anger", "strong"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("contempt", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("happiness", "weak"),
    ),
    ":yellow_heart:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":musical_note:": _ranking(
        ("happiness", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("contempt", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
    ),
    ":speak_no_evil:": _ranking(
        ("fear", "strong"),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":wink:": _ranking(
        ("happiness", "weak"),
        ("contempt", "strong"),
        ("surprise", "weak"),
        ("neutral", None),
        ("anger", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
    ),
    ":skull:": _ranking(
        ("fear", "strong"),
        ("sadness", "strong"),
        ("disgust", "strong"),
        ("anger", "weak"),
        ("contempt", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("happiness", "weak"),
    ),
    ":confounded:": _ranking(
        ("sadness", "strong"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":smile:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("contempt", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
    ),
    ":stuck_out_tongue_winking_eye:": _ranking(
        ("happiness", "strong"),
        ("surprise", "strong"),
        ("contempt", "strong"),
        ("neutral", None),
        ("anger", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("sadness", "weak"),
    ),
    ":angry:": _ranking(
        ("anger", "strong"),
        ("disgust", "strong"),
        ("contempt", "strong"),
        ("fear", "weak"),
        ("sadness", "weak"),
        ("surprise", "weak"),
        ("happiness", "weak"),
        ("neutral", None),
    ),
    ":no_good:": _ranking(
        ("disgust", "strong"),
        ("anger", "strong"),
        ("contempt", "strong"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("happiness", "weak"),
    ),
    ":muscle:": _ranking(
        ("happiness", "strong"),
        ("anger", "strong"),
        ("contempt", "strong"),
        ("neutral", None),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("sadness", "weak"),
        ("disgust", "weak"),
    ),
    ":facepunch:": _ranking(
        ("anger", "strong"),
        ("contempt", "strong"),
        ("fear", "strong"),
        ("disgust", "strong"),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
        ("happiness", "weak"),
    ),
    ":purple_heart:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("surprise", "weak"),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":sparkling_heart:": _ranking(
        ("happiness", "strong"),
        ("surprise", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":blue_heart:": _ranking(
        ("happiness", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("surprise", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
    ":grimacing:": _ranking(
        ("fear", "strong"),
        ("surprise", "strong"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("sadness", "weak"),
        ("neutral", None),
        ("contempt", "weak"),
        ("happiness", "weak"),
    ),
    ":sparkles:": _ranking(
        ("happiness", "weak"),
        ("surprise", "weak"),
        ("neutral", None),
        ("sadness", "weak"),
        ("fear", "weak"),
        ("disgust", "weak"),
        ("anger", "weak"),
        ("contempt", "weak"),
    ),
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
    "EMOJI_EMOTION_RANKS",
    "filter_emojis_by_emotion",
    "get_emotion_rankings",
    "iter_emotion_rankings",
    "select_accessible_ranking",
]
