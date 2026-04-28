"""
main.py — Command-line runner for the AI Music Mood Recommender.

Demonstrates three modes:
1. Traditional user profiles (original Module 1-3 behavior)
2. AI mood detection from text (new applied AI feature)
3. Guardrail / validation demos

Usage (from project root):
    python src/main.py
"""

import sys
import os

from recommender import (
    load_songs, load_songs_as_objects, recommend_songs,
    Recommender, analyze_mood_keywords, validate_input_text, sanitize_text,
    logger,
)


def demo_original_profiles(songs_dicts):
    """Original demo with manually defined user profiles."""
    user_profiles = [
        {
            "name": "Angry Calm-Seeker",
            "prefs": {
                "favorite_genre": "rock",
                "favorite_mood": "angry",
                "target_energy": 0.1,
                "target_valence": 0.2,
                "likes_acoustic": False,
            },
        },
        {
            "name": "Hyperactive Acoustic Lover",
            "prefs": {
                "favorite_genre": "pop",
                "favorite_mood": "energetic",
                "target_energy": 0.95,
                "target_valence": 0.85,
                "likes_acoustic": True,
            },
        },
        {
            "name": "Zero Energy Minimalist",
            "prefs": {
                "favorite_genre": "lofi",
                "favorite_mood": "peaceful",
                "target_energy": 0.0,
                "target_valence": 0.6,
                "likes_acoustic": True,
            },
        },
    ]

    for profile in user_profiles:
        recommendations = recommend_songs(profile["prefs"], songs_dicts, k=10)
        print("=" * 70)
        genre = profile["prefs"]["favorite_genre"].upper()
        mood = profile["prefs"]["favorite_mood"].upper()
        print(f"🎵 TOP RECOMMENDATIONS FOR {profile['name'].upper()}")
        print(f"   ({genre} • {mood} MOOD • Energy: {profile['prefs']['target_energy']:.2f})")
        print("=" * 70 + "\n")
        for i, rec in enumerate(recommendations, 1):
            song, score, explanation = rec
            print(f"#{i} {song['title']}")
            print(f"    Artist: {song['artist']}")
            print(f"    Score:  {score:.2f}/10")
            print(f"    Reasons: {explanation}")
            print()
        print()


def demo_mood_detection(songs_objects):
    """Demonstrate the AI mood detection → recommendations pipeline."""
    rec = Recommender(songs_objects)

    sample_texts = [
        "I just had a terrible breakup and I can't stop crying",
        "Heading to the gym, need something to get me pumped up!",
        "It's a rainy Sunday morning, I'm making coffee and reading a book",
        "I'm pulling an all-nighter coding this project, need to stay focused",
        "Just got a promotion! Feeling on top of the world!",
    ]

    print("\n" + "=" * 70)
    print("  🧠 AI MOOD DETECTION → PLAYLIST GENERATOR")
    print("=" * 70 + "\n")

    for text in sample_texts:
        analysis, recs = rec.recommend_from_text(text, k=3)
        print("=" * 70)
        print(f'📝 "{text}"')
        print("-" * 70)
        print(f"   Detected mood:   {analysis.detected_mood}")
        print(f"   Energy level:    {analysis.energy_level:.0%}")
        print(f"   Valence:         {analysis.valence_level:.0%}")
        print(f"   Suggested genre: {analysis.suggested_genre}")
        print(f"   Acoustic:        {'Yes' if analysis.likes_acoustic else 'No'}")
        print(f"   Confidence:      {analysis.confidence:.0%}")
        print(f"   Reasoning:       {analysis.reasoning}")
        print("-" * 70)
        for i, (song, score, explanation) in enumerate(recs, 1):
            print(f"   #{i} {song.title} — {song.artist}")
            print(f"      Score: {score:.2f} | {explanation}")
        print()


def demo_guardrails(songs_objects):
    """Demonstrate input validation and guardrail behavior."""
    rec = Recommender(songs_objects)

    print("\n" + "=" * 70)
    print("  🛡️  GUARDRAIL & VALIDATION DEMOS")
    print("=" * 70 + "\n")

    test_inputs = [
        ("", "Empty input"),
        ("x", "Too short"),
        ("a" * 2500, "Too long (2500 chars)"),
        ("<script>alert('xss')</script>", "Script injection attempt"),
        ("I feel happy and want to dance!", "Valid input"),
    ]

    for text, label in test_inputs:
        print(f"Test: {label}")
        is_valid, error = validate_input_text(text)
        if is_valid:
            analysis, recs = rec.recommend_from_text(text, k=2)
            print(f"  ✅ Accepted → mood={analysis.detected_mood}, confidence={analysis.confidence:.0%}")
            print(f"     Top pick: {recs[0][0].title} (score={recs[0][1]:.2f})")
        else:
            print(f"  🚫 Rejected: {error}")
        print()


def interactive_mode(songs_objects):
    """Interactive CLI: type how you feel, get a playlist."""
    rec = Recommender(songs_objects)

    print("\n" + "=" * 70)
    print("  🎧 MOOD-BASED MUSIC RECOMMENDER")
    print("  Tell me how you're feeling and I'll pick your playlist.")
    print("  Type 'quit' to exit.")
    print("=" * 70 + "\n")

    while True:
        try:
            text = input("How are you feeling? → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 🎶")
            break
        if text.lower() in ("quit", "exit", "q"):
            print("Bye! 🎶")
            break
        if not text:
            continue
        try:
            analysis, recs = rec.recommend_from_text(text, k=10)
        except ValueError as e:
            print(f"  ⚠️  {e}\n")
            continue
        print(f"\n  🧠 Detected: {analysis.detected_mood} "
              f"(energy {analysis.energy_level:.0%}, "
              f"valence {analysis.valence_level:.0%}) "
              f"→ {analysis.suggested_genre}\n")
        for i, (song, score, explanation) in enumerate(recs, 1):
            print(f"  #{i} {song.title} — {song.artist} ({score:.2f})")
            print(f"      {explanation}")
        print()


def main() -> None:
    # Resolve path to data/songs.csv relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # go up from src/ to root
    csv_path = os.path.join(project_root, "data", "songs.csv")

    try:
        songs_dicts = load_songs(csv_path)
        songs_objects = load_songs_as_objects(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        print("Make sure songs.csv is in the data/ folder at the project root.")
        sys.exit(1)

    print(f"Loaded songs: {len(songs_dicts)}\n")

    demo_original_profiles(songs_dicts)
    demo_mood_detection(songs_objects)
    demo_guardrails(songs_objects)

    # Uncomment for interactive mode:
    # interactive_mode(songs_objects)


if __name__ == "__main__":
    main()
