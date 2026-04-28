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
    and return the full result including a processing trail for the debug panel.

    Request body: { "text": "I feel sad and lonely", "k": 6 }
    Response: { "analysis": {...}, "recommendations": [...], "processing": [...] }
    """
    import time
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"]
    k = data.get("k", 10)
    processing = []  # step-by-step trail
    t0 = time.time()

    # Step 1: Input validation (guardrails)
    is_valid, error_msg = validate_input_text(text)
    processing.append({
        "step": 1,
        "name": "Input Validation",
        "icon": "🛡️",
        "status": "pass" if is_valid else "fail",
        "detail": f"Length: {len(text)} chars" if is_valid else f"Rejected: {error_msg}",
        "checks": [
            {"label": "Not empty", "pass": bool(text.strip())},
            {"label": f"Length ≥ 2 chars", "pass": len(text.strip()) >= 2 if text.strip() else False},
            {"label": f"Length ≤ 2000 chars", "pass": len(text.strip()) <= 2000 if text.strip() else True},
            {"label": "No script injection", "pass": not bool(__import__('re').search(r'<script|<iframe|javascript:', text, __import__('re').IGNORECASE))},
        ],
    })

    if not is_valid:
        logger.warning(f"API guardrail rejected input: {error_msg}")
        return jsonify({"error": error_msg, "guardrail": True, "processing": processing}), 400

    # Step 2: Sanitization
    clean_text = sanitize_text(text)
    processing.append({
        "step": 2,
        "name": "Text Sanitization",
        "icon": "🧹",
        "status": "pass",
        "detail": f"Cleaned: \"{clean_text[:80]}{'...' if len(clean_text) > 80 else ''}\"",
    })

    # Step 3: Mood analysis
    try:
        analysis, recs = recommender.recommend_from_text(clean_text, k=k)
    except ValueError as e:
        return jsonify({"error": str(e), "guardrail": True, "processing": processing}), 400

    processing.append({
        "step": 3,
        "name": "Keyword Mood Analysis",
        "icon": "🧠",
        "status": "pass",
        "detail": f"Matched mood: {analysis.detected_mood} ({analysis.confidence:.0%} confidence)",
        "keywords_matched": analysis.reasoning,
        "parameters": {
            "detected_mood": analysis.detected_mood,
            "energy_level": analysis.energy_level,
            "valence_level": analysis.valence_level,
            "suggested_genre": analysis.suggested_genre,
            "likes_acoustic": analysis.likes_acoustic,
        },
    })

    # Step 4: Profile conversion
    processing.append({
        "step": 4,
        "name": "UserProfile Conversion",
        "icon": "👤",
        "status": "pass",
        "detail": f"Genre={analysis.suggested_genre}, Mood={analysis.detected_mood}, Energy={analysis.energy_level}, Acoustic={analysis.likes_acoustic}",
    })

    # Step 5: Scoring
    top_score = recs[0][1] if recs else 0
    low_score = recs[-1][1] if recs else 0
    processing.append({
        "step": 5,
        "name": f"Scoring Engine ({len(recommender.songs)} songs)",
        "icon": "📊",
        "status": "pass",
        "detail": f"Scored all songs, top={top_score:.2f}, lowest returned={low_score:.2f}",
        "formula": "genre(+3.0) + mood(+2.0) + energy_sim(×2.0) + valence_sim(×1.5) + dance(×1.0) + acoustic(+0.5) = 10.0 max",
    })

    # Step 6: Results
    elapsed = round((time.time() - t0) * 1000, 1)
    processing.append({
        "step": 6,
        "name": "Results Returned",
        "icon": "✅",
        "status": "pass",
        "detail": f"{len(recs)} songs returned in {elapsed}ms",
    })

    # Build response
    low_confidence = analysis.confidence < 0.3
    is_blended = "Blended moods" in analysis.reasoning

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
            "low_confidence": low_confidence,
            "blended": is_blended,
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
        "processing": processing,
    }

    return jsonify(response)


if __name__ == "__main__":
    print("\n🎧 Mood Recommender API running at http://localhost:5000")
    print("   POST /api/analyze  — analyze mood from text")
    print("   GET  /api/songs    — get song catalog\n")
    app.run(debug=True, port=5000)
