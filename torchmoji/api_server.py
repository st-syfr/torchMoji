"""HTTP API server for TorchMoji predictions and settings management."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from flask import Flask, jsonify, request

from .cli import EMOJI_ALIASES, _emoji_from_alias, _ensure_file, _resolve_emotion_filters
from .runtime import PredictionSettings, get_runtime
from .settings import TorchMojiSettings, load_settings, save_settings
from .gui.utils import build_cli_command, format_cli_command

_HELP_URL = "https://github.com/huggingface/torchMoji#download-the-pretrained-weights"


class TorchMojiAPIServer:
    """HTTP API server for TorchMoji."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.settings = load_settings()
        
        # Register routes
        self.app.add_url_rule("/predict", "predict", self._predict_endpoint, methods=["POST"])
        self.app.add_url_rule("/settings", "get_settings", self._get_settings_endpoint, methods=["GET"])
        self.app.add_url_rule("/settings", "update_settings", self._update_settings_endpoint, methods=["POST"])
        self.app.add_url_rule("/cli-preview", "cli_preview", self._cli_preview_endpoint, methods=["GET"])
        self.app.add_url_rule("/health", "health", self._health_endpoint, methods=["GET"])

    def _predict_endpoint(self) -> tuple[dict[str, Any], int]:
        """Handle prediction requests."""
        data = request.get_json()
        if not data or "text" not in data:
            return {"error": "Missing 'text' field in request body"}, 400

        text = data["text"]
        if not isinstance(text, str) or not text.strip():
            return {"error": "'text' must be a non-empty string"}, 400

        # Allow optional settings override in the request
        request_settings = self.settings
        if "settings" in data and isinstance(data["settings"], dict):
            settings_dict = self.settings.to_dict()
            settings_dict.update(data["settings"])
            try:
                request_settings = TorchMojiSettings.from_dict(settings_dict)
            except Exception as exc:
                return {"error": f"Invalid settings: {exc}"}, 400

        weights_path = Path(request_settings.weights)
        vocab_path = Path(request_settings.vocab)

        if not weights_path.exists():
            return {"error": f"Pretrained weights not found at '{weights_path}'"}, 500
        if not vocab_path.exists():
            return {"error": f"Vocabulary file not found at '{vocab_path}'"}, 500

        allowed_emotions, weak_emotions, strong_emotions = _resolve_emotion_filters(
            request_settings.mode,
            request_settings.emotions,
            request_settings.weak_emotions,
            request_settings.strong_emotions,
        )

        try:
            runtime = get_runtime(weights_path, vocab_path, request_settings.maxlen)
        except FileNotFoundError:
            return {"error": f"Pretrained weights not found at '{weights_path}'"}, 500
        except json.JSONDecodeError as exc:
            return {"error": f"Failed to parse vocabulary JSON: {exc}"}, 500
        except Exception as exc:
            return {"error": f"Could not initialise TorchMoji runtime: {exc}"}, 500

        try:
            result = runtime.predict(
                text,
                PredictionSettings(
                    top_k=request_settings.top_k,
                    allowed_emotions=allowed_emotions,
                    weak_emotions=weak_emotions,
                    strong_emotions=strong_emotions,
                ),
            )
        except RuntimeError as exc:
            return {"error": str(exc)}, 500

        predictions = []
        for rank, selection in enumerate(result.selections, start=1):
            index = selection.index
            alias = EMOJI_ALIASES[index] if index < len(EMOJI_ALIASES) else str(index)
            char = _emoji_from_alias(alias)
            ranking = selection.ranking
            
            prediction_data = {
                "rank": rank,
                "emoji": char,
                "alias": alias,
                "index": index,
            }
            
            if selection.score is not None:
                prediction_data["score"] = float(selection.score)
            
            if ranking is not None:
                prediction_data["emotion"] = ranking.emotion
                if ranking.intensity is not None:
                    prediction_data["intensity"] = ranking.intensity
            
            predictions.append(prediction_data)

        return {
            "text": text,
            "predictions": predictions,
            "settings_used": request_settings.to_dict(),
        }, 200

    def _get_settings_endpoint(self) -> tuple[dict[str, Any], int]:
        """Return current settings."""
        return {"settings": self.settings.to_dict()}, 200

    def _update_settings_endpoint(self) -> tuple[dict[str, Any], int]:
        """Update server settings."""
        data = request.get_json()
        if not data:
            return {"error": "Missing request body"}, 400

        try:
            settings_dict = self.settings.to_dict()
            settings_dict.update(data)
            new_settings = TorchMojiSettings.from_dict(settings_dict)
            self.settings = new_settings
            
            # Optionally persist the settings
            if data.get("persist", False):
                save_settings(self.settings)
            
            return {
                "message": "Settings updated successfully",
                "settings": self.settings.to_dict(),
            }, 200
        except Exception as exc:
            return {"error": f"Failed to update settings: {exc}"}, 400

    def _cli_preview_endpoint(self) -> tuple[dict[str, Any], int]:
        """Return CLI command preview for current settings."""
        text = request.args.get("text", "")
        tokens = build_cli_command(self.settings, text.strip() or None)
        return {
            "command": format_cli_command(tokens),
            "tokens": list(tokens),
            "settings": self.settings.to_dict(),
        }, 200

    def _health_endpoint(self) -> tuple[dict[str, Any], int]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "torchmoji-api"}, 200

    def run(self, debug: bool = False) -> None:
        """Start the Flask server."""
        print(f"Starting TorchMoji API server on http://{self.host}:{self.port}")
        print(f"Endpoints:")
        print(f"  POST   /predict         - Predict emojis for text")
        print(f"  GET    /settings        - Get current settings")
        print(f"  POST   /settings        - Update settings")
        print(f"  GET    /cli-preview     - Get CLI command preview")
        print(f"  GET    /health          - Health check")
        self.app.run(host=self.host, port=self.port, debug=debug)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="torchmoji-api",
        description="HTTP API server for TorchMoji predictions and settings management.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind the server to (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    args = parser.parse_args(argv)
    
    server = TorchMojiAPIServer(host=args.host, port=args.port)
    try:
        server.run(debug=args.debug)
        return 0
    except KeyboardInterrupt:
        print("\nShutting down server...")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
