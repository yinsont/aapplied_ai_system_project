"""
api.py — Flask API server for the AI Music Mood Recommender.

Exposes two endpoints:
  GET  /api/songs       → returns the full song catalog as JSON
  POST /api/analyze      → accepts { "text": "..." }, returns mood analysis + recommendations

This wraps the existing recommender.py logic so the React frontend
can call your Python code directly.

Usage (from project root):
    pip install flask flask-cors
    python src/api.py
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import from the same src/ directory
from recommender import (
    load_songs_as_objects,
    Recommender,
    validate_input_text,
    sanitize_text,
    analyze_mood_keywords,
    logger,
)

app = Flask(__name__)
CORS(app)  # Allow React frontend to call this

# ── Load songs once at startup ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "songs.csv")

try:
    songs = load_songs_as_objects(CSV_PATH)
    recommender = Recommender(songs)
    print(f"✅ Loaded {len(songs)} songs from {CSV_PATH}")
except FileNotFoundError:
    print(f"❌ Could not find {CSV_PATH}")
    print("   Make sure songs.csv is in the data/ folder at the project root.")
    sys.exit(1)


@app.route("/api/songs", methods=["GET"])
def get_songs():
    """Return the full song catalog as JSON."""
    return jsonify([
        {
            "id": s.id, "title": s.title, "artist": s.artist,
            "genre": s.genre, "mood": s.mood, "energy": s.energy,
            "tempo_bpm": s.tempo_bpm, "valence": s.valence,
            "danceability": s.danceability, "acousticness": s.acousticness,
        }
        for s in songs
    ])


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accept user text, run mood analysis + recommendation through Python,
    and return the full result.

    Request body: { "text": "I feel sad and lonely", "k": 6 }
    Response: { "analysis": {...}, "recommendations": [...] }
    """
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"]
    k = data.get("k", 6)

    # Validate input (guardrails)
    is_valid, error_msg = validate_input_text(text)
    if not is_valid:
        logger.warning(f"API guardrail rejected input: {error_msg}")
        return jsonify({"error": error_msg, "guardrail": True}), 400

    # Sanitize and analyze
    clean_text = sanitize_text(text)

    try:
        analysis, recs = recommender.recommend_from_text(clean_text, k=k)
    except ValueError as e:
        return jsonify({"error": str(e), "guardrail": True}), 400

    # Build response
    response = {
        "analysis": {
            "detected_mood": analysis.detected_mood,
            "energy_level": analysis.energy_level,
            "valence_level": analysis.valence_level,
            "suggested_genre": analysis.suggested_genre,
            "likes_acoustic": analysis.likes_acoustic,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
            "source": analysis.source,
        },
        "recommendations": [
            {
                "song": {
                    "id": song.id, "title": song.title, "artist": song.artist,
                    "genre": song.genre, "mood": song.mood, "energy": song.energy,
                    "tempo_bpm": song.tempo_bpm, "valence": song.valence,
                    "danceability": song.danceability, "acousticness": song.acousticness,
                },
                "score": round(score, 2),
                "explanation": explanation,
            }
            for song, score, explanation in recs
        ],
    }

    return jsonify(response)


if __name__ == "__main__":
    print("\n🎧 Mood Recommender API running at http://localhost:5000")
    print("   POST /api/analyze  — analyze mood from text")
    print("   GET  /api/songs    — get song catalog\n")
    app.run(debug=True, port=5000)
