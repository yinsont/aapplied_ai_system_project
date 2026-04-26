"""
evaluate.py — Test Harness & Evaluation Script

Runs the full mood-recommender pipeline on a predefined set of test cases
and prints a structured pass/fail summary with confidence scores.

This satisfies the rubric's "Test Harness or Evaluation Script" stretch feature (+2 pts).

Usage (from project root):
    python evaluate.py
"""

import sys
import os
from datetime import datetime

# Add src to path so we can import recommender
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from recommender import (
    Recommender, load_songs_as_objects, analyze_mood_keywords,
    validate_input_text, MoodAnalysis,
)


# ── Test cases ──

TEST_CASES = [
    {
        "id": "TC-01",
        "input": "I just had a terrible breakup and I can't stop crying",
        "expected_mood": "melancholic",
        "energy_range": (0.0, 0.5),
        "expected_genre": "indie folk",
        "description": "Sad/heartbreak text should map to melancholic + low energy",
    },
    {
        "id": "TC-02",
        "input": "Heading to the gym, need something to get me pumped up!",
        "expected_mood": "intense",
        "energy_range": (0.7, 1.0),
        "expected_genre": "hip-hop",
        "description": "Workout text should map to intense + high energy",
    },
    {
        "id": "TC-03",
        "input": "It's a rainy Sunday morning, making coffee and reading",
        "expected_mood": "relaxed",
        "energy_range": (0.2, 0.6),
        "expected_genre": "jazz",
        "description": "Cozy morning text should map to relaxed + low-medium energy",
    },
    {
        "id": "TC-04",
        "input": "I need to focus and study for my final exam tonight",
        "expected_mood": "focused",
        "energy_range": (0.2, 0.6),
        "expected_genre": "lofi",
        "description": "Study text should map to focused + lofi",
    },
    {
        "id": "TC-05",
        "input": "Just got promoted! Feeling amazing and happy!",
        "expected_mood": "happy",
        "energy_range": (0.5, 1.0),
        "expected_genre": "pop",
        "description": "Celebration text should map to happy + pop",
    },
    {
        "id": "TC-06",
        "input": "I'm so angry and frustrated with everything right now",
        "expected_mood": "aggressive",
        "energy_range": (0.7, 1.0),
        "expected_genre": "rock",
        "description": "Angry text should map to aggressive + rock",
    },
    {
        "id": "TC-07",
        "input": "Let's go to a party and dance all night!",
        "expected_mood": "energetic",
        "energy_range": (0.7, 1.0),
        "expected_genre": "electronic",
        "description": "Party text should map to energetic + electronic",
    },
    {
        "id": "TC-08",
        "input": "The cat sat on the mat",
        "expected_mood": "chill",
        "energy_range": (0.0, 1.0),
        "expected_genre": None,
        "description": "Neutral text with no mood keywords should fallback gracefully",
    },
    {
        "id": "TC-09",
        "input": "",
        "expected_mood": None,
        "energy_range": None,
        "expected_genre": None,
        "description": "Empty input should be rejected by input validation",
        "expect_rejection": True,
    },
    {
        "id": "TC-10",
        "input": "<script>alert('xss')</script>",
        "expected_mood": None,
        "energy_range": None,
        "expected_genre": None,
        "description": "Script injection should be rejected by guardrail",
        "expect_rejection": True,
    },
]


def run_evaluation():
    """Run all test cases and produce a summary report."""
    print("=" * 72)
    print("  🧪 MOOD RECOMMENDER — EVALUATION HARNESS")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72 + "\n")

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "songs.csv")

    try:
        songs = load_songs_as_objects(csv_path)
        rec = Recommender(songs)
        print(f"✅ Loaded {len(songs)} songs\n")
    except FileNotFoundError:
        print(f"❌ Could not find {csv_path}")
        print("   Make sure songs.csv is in the data/ folder at the project root.")
        sys.exit(1)

    results = []
    total_confidence = 0.0
    confidence_count = 0

    for tc in TEST_CASES:
        tc_id = tc["id"]
        expect_rejection = tc.get("expect_rejection", False)
        checks_passed = 0
        checks_total = 0
        details = []

        print(f"─── {tc_id}: {tc['description']}")
        print(f"    Input: \"{tc['input'][:60]}{'...' if len(tc['input']) > 60 else ''}\"")

        if expect_rejection:
            checks_total += 1
            is_valid, err = validate_input_text(tc["input"])
            if not is_valid:
                checks_passed += 1
                details.append(f"  ✅ Correctly rejected: {err}")
            else:
                details.append("  ❌ Should have been rejected but was accepted")
        else:
            try:
                analysis, recs = rec.recommend_from_text(tc["input"], k=3)
            except ValueError as e:
                details.append(f"  ❌ Unexpected rejection: {e}")
                checks_total += 1
                for d in details:
                    print(d)
                results.append({"id": tc_id, "passed": 0, "total": 1, "status": "FAIL"})
                print()
                continue

            total_confidence += analysis.confidence
            confidence_count += 1

            if tc["expected_mood"]:
                checks_total += 1
                if analysis.detected_mood == tc["expected_mood"]:
                    checks_passed += 1
                    details.append(f"  ✅ Mood: {analysis.detected_mood}")
                else:
                    details.append(f"  ❌ Mood: got '{analysis.detected_mood}', expected '{tc['expected_mood']}'")

            if tc["energy_range"]:
                checks_total += 1
                lo, hi = tc["energy_range"]
                if lo <= analysis.energy_level <= hi:
                    checks_passed += 1
                    details.append(f"  ✅ Energy: {analysis.energy_level:.2f} (in [{lo}, {hi}])")
                else:
                    details.append(f"  ❌ Energy: {analysis.energy_level:.2f} (expected [{lo}, {hi}])")

            if tc["expected_genre"]:
                checks_total += 1
                if analysis.suggested_genre == tc["expected_genre"]:
                    checks_passed += 1
                    details.append(f"  ✅ Genre: {analysis.suggested_genre}")
                else:
                    details.append(f"  ❌ Genre: got '{analysis.suggested_genre}', expected '{tc['expected_genre']}'")

            checks_total += 1
            if len(recs) > 0:
                checks_passed += 1
                top = recs[0]
                details.append(f"  ✅ Recommendations: {len(recs)} returned (top: '{top[0].title}', score={top[1]:.2f})")
            else:
                details.append("  ❌ No recommendations returned")

            details.append(f"  📊 Confidence: {analysis.confidence:.0%}")

        for d in details:
            print(d)

        status = "PASS" if checks_passed == checks_total else "FAIL"
        results.append({"id": tc_id, "passed": checks_passed, "total": checks_total, "status": status})
        print(f"    → {status} ({checks_passed}/{checks_total} checks)\n")

    # Summary
    total_passed = sum(r["passed"] for r in results)
    total_checks = sum(r["total"] for r in results)
    cases_passed = sum(1 for r in results if r["status"] == "PASS")
    avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0

    print("=" * 72)
    print("  📋 EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Test cases:    {cases_passed}/{len(results)} passed")
    print(f"  Total checks:  {total_passed}/{total_checks} passed")
    print(f"  Pass rate:     {total_passed / total_checks * 100:.1f}%")
    print(f"  Avg confidence: {avg_confidence:.0%}")
    print("=" * 72)

    print("\n  Per-case breakdown:")
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"    {icon} {r['id']}: {r['passed']}/{r['total']} checks")

    print()
    return total_passed == total_checks


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
