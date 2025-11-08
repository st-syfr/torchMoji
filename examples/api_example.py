#!/usr/bin/env python
"""Example script demonstrating the TorchMoji API server."""
from __future__ import annotations

import json
import time

import requests

API_URL = "http://127.0.0.1:5000"


def main():
    """Run API demonstration."""
    print("TorchMoji API Example")
    print("=" * 60)
    
    # Check health
    print("\n1. Checking API health...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.json()}")
    
    # Get current settings
    print("\n2. Getting current settings...")
    response = requests.get(f"{API_URL}/settings")
    settings = response.json()["settings"]
    print(f"   Top K: {settings['top_k']}")
    print(f"   Max Length: {settings['maxlen']}")
    print(f"   Mode: {settings['mode']}")
    
    # Update settings
    print("\n3. Updating settings (top_k=3, scores=True)...")
    response = requests.post(
        f"{API_URL}/settings",
        json={"top_k": 3, "scores": True},
    )
    print(f"   Updated: {response.json()['message']}")
    
    # Get CLI preview
    print("\n4. Getting CLI preview for text 'I love this!'...")
    response = requests.get(
        f"{API_URL}/cli-preview",
        params={"text": "I love this!"},
    )
    print(f"   CLI Command: {response.json()['command']}")
    
    # Make prediction
    print("\n5. Making prediction for 'I love this!'...")
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": "I love this!"},
    )
    result = response.json()
    print(f"   Input: {result['text']}")
    print("   Predictions:")
    for pred in result["predictions"]:
        score_str = f" (score={pred['score']:.4f})" if "score" in pred else ""
        emotion_str = ""
        if "emotion" in pred:
            emotion_str = f" -> {pred['emotion']}"
            if "intensity" in pred:
                emotion_str += f" ({pred['intensity']})"
        print(f"      {pred['rank']}. {pred['emoji']} {pred['alias']}{emotion_str}{score_str}")
    
    # Make prediction with settings override
    print("\n6. Making prediction with settings override (top_k=5)...")
    response = requests.post(
        f"{API_URL}/predict",
        json={
            "text": "This is terrible!",
            "settings": {"top_k": 5},
        },
    )
    result = response.json()
    print(f"   Input: {result['text']}")
    print(f"   Top K used: {result['settings_used']['top_k']}")
    print("   Predictions:")
    for pred in result["predictions"][:3]:  # Show first 3
        print(f"      {pred['rank']}. {pred['emoji']} {pred['alias']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Make sure the server is running with: torchmoji-api")
        print("Or: python -m torchmoji.api_server")
    except Exception as e:
        print(f"Error: {e}")
