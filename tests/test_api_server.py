"""Tests for the TorchMoji API server."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from torchmoji.api_server import TorchMojiAPIServer
from torchmoji.runtime import EmojiPredictionResult, EmojiSelection
from torchmoji.emojis import EmotionRanking
from torchmoji.settings import TorchMojiSettings


@pytest.fixture
def api_server():
    """Create a test API server instance."""
    with patch("torchmoji.api_server.load_settings") as mock_load:
        mock_load.return_value = TorchMojiSettings()
        server = TorchMojiAPIServer()
        server.app.config["TESTING"] = True
        yield server


@pytest.fixture
def client(api_server):
    """Create a test client for the API server."""
    return api_server.app.test_client()


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "torchmoji-api"


def test_get_settings_endpoint(client, api_server):
    """Test retrieving current settings."""
    response = client.get("/settings")
    assert response.status_code == 200
    data = response.get_json()
    assert "settings" in data
    settings = data["settings"]
    assert settings["top_k"] == 5
    assert settings["maxlen"] == 30
    assert settings["mode"] == "standard"


def test_update_settings_endpoint(client, api_server):
    """Test updating settings via API."""
    new_settings = {
        "top_k": 10,
        "maxlen": 50,
        "scores": True,
    }
    response = client.post(
        "/settings",
        data=json.dumps(new_settings),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "Settings updated successfully"
    assert data["settings"]["top_k"] == 10
    assert data["settings"]["maxlen"] == 50
    assert data["settings"]["scores"] is True


def test_update_settings_with_persist(client, api_server):
    """Test updating and persisting settings."""
    with patch("torchmoji.api_server.save_settings") as mock_save:
        new_settings = {
            "top_k": 3,
            "persist": True,
        }
        response = client.post(
            "/settings",
            data=json.dumps(new_settings),
            content_type="application/json",
        )
        assert response.status_code == 200
        # Verify save_settings was called
        assert mock_save.called


def test_update_settings_invalid_data(client):
    """Test updating settings with invalid data."""
    response = client.post(
        "/settings",
        data=json.dumps(None),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_cli_preview_endpoint(client, api_server):
    """Test CLI command preview generation."""
    response = client.get("/cli-preview?text=hello world")
    assert response.status_code == 200
    data = response.get_json()
    assert "command" in data
    assert "tokens" in data
    assert "settings" in data
    assert isinstance(data["tokens"], list)


def test_cli_preview_without_text(client):
    """Test CLI preview without text."""
    response = client.get("/cli-preview")
    assert response.status_code == 200
    data = response.get_json()
    assert "command" in data


def test_predict_missing_text(client):
    """Test prediction endpoint with missing text."""
    response = client.post(
        "/predict",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "text" in data["error"]


def test_predict_empty_text(client):
    """Test prediction endpoint with empty text."""
    response = client.post(
        "/predict",
        data=json.dumps({"text": ""}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_invalid_text_type(client):
    """Test prediction endpoint with invalid text type."""
    response = client.post(
        "/predict",
        data=json.dumps({"text": 123}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_with_mock_runtime(client, api_server, tmp_path):
    """Test prediction endpoint with mocked runtime."""
    # Create temporary files for weights and vocab
    weights_file = tmp_path / "weights.bin"
    vocab_file = tmp_path / "vocab.json"
    weights_file.write_text("fake weights")
    vocab_file.write_text('{"hello": 0, "world": 1}')
    
    # Update server settings to use temp files
    api_server.settings.weights = str(weights_file)
    api_server.settings.vocab = str(vocab_file)
    
    # Create mock prediction result
    mock_result = EmojiPredictionResult(
        probabilities=np.array([0.9, 0.8, 0.7]),
        selections=[
            EmojiSelection(
                index=0,
                ranking=EmotionRanking(emotion="happiness", intensity="strong"),
                score=0.9,
            ),
            EmojiSelection(
                index=1,
                ranking=EmotionRanking(emotion="sadness", intensity="weak"),
                score=0.8,
            ),
        ],
    )
    
    with patch("torchmoji.api_server.get_runtime") as mock_runtime:
        mock_runtime.return_value.predict.return_value = mock_result
        
        response = client.post(
            "/predict",
            data=json.dumps({"text": "test text"}),
            content_type="application/json",
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "text" in data
        assert data["text"] == "test text"
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["rank"] == 1
        assert data["predictions"][0]["score"] == 0.9
        assert data["predictions"][0]["emotion"] == "happiness"
        assert data["predictions"][0]["intensity"] == "strong"


def test_predict_with_settings_override(client, api_server, tmp_path):
    """Test prediction with settings override in request."""
    weights_file = tmp_path / "weights.bin"
    vocab_file = tmp_path / "vocab.json"
    weights_file.write_text("fake weights")
    vocab_file.write_text('{"hello": 0}')
    
    api_server.settings.weights = str(weights_file)
    api_server.settings.vocab = str(vocab_file)
    
    mock_result = EmojiPredictionResult(
        probabilities=np.array([0.9]),
        selections=[EmojiSelection(index=0, ranking=None, score=0.9)],
    )
    
    with patch("torchmoji.api_server.get_runtime") as mock_runtime:
        mock_runtime.return_value.predict.return_value = mock_result
        
        response = client.post(
            "/predict",
            data=json.dumps({
                "text": "test",
                "settings": {"top_k": 3},
            }),
            content_type="application/json",
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["settings_used"]["top_k"] == 3


def test_predict_weights_not_found(client, api_server):
    """Test prediction when weights file doesn't exist."""
    api_server.settings.weights = "/nonexistent/weights.bin"
    api_server.settings.vocab = "/nonexistent/vocab.json"
    
    response = client.post(
        "/predict",
        data=json.dumps({"text": "test"}),
        content_type="application/json",
    )
    
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data


def test_main_function_help():
    """Test that main function can parse help argument."""
    from torchmoji.api_server import main
    
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    
    assert exc_info.value.code == 0


def test_main_function_arguments():
    """Test main function argument parsing."""
    from torchmoji.api_server import main
    
    with patch("torchmoji.api_server.TorchMojiAPIServer") as mock_server_class:
        mock_instance = MagicMock()
        mock_server_class.return_value = mock_instance
        mock_instance.run.side_effect = KeyboardInterrupt()
        
        result = main(["--host", "0.0.0.0", "--port", "8080"])
        
        assert result == 0
        mock_server_class.assert_called_once_with(host="0.0.0.0", port=8080)
