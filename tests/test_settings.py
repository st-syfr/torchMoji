from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchmoji import settings as settings_module


def test_settings_round_trip(tmp_path):
    settings = settings_module.TorchMojiSettings(
        top_k=7,
        maxlen=64,
        scores=True,
        mode="simple",
        emotions=["happiness", "sadness"],
        weak_emotions=["happiness"],
        strong_emotions=["happiness"],
        weights=str(tmp_path / "weights.bin"),
        vocab=str(tmp_path / "vocab.json"),
    )

    payload = settings.to_dict()
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = settings_module.load_settings(path)
    assert loaded == settings


def test_merge_with_namespace_updates_fields(tmp_path):
    base = settings_module.TorchMojiSettings()
    namespace = argparse.Namespace(
        top_k=11,
        scores=True,
        weights=Path(tmp_path / "weights.bin"),
        emotions=("joy", "anger"),
    )

    updated, touched = base.merge_with_namespace(namespace)
    assert touched == {"top_k", "scores", "weights", "emotions"}
    assert updated.top_k == 11
    assert updated.scores is True
    assert updated.weights.endswith("weights.bin")
    assert updated.emotions == ["joy", "anger"]

    # Ensure the original instance remains unchanged
    assert base.top_k == 5
    assert base.scores is False
    assert base.emotions is None
