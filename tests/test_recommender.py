"""
tests/test_recommender.py — Test suite for the AI Music Mood Recommender.

Covers:
- Song / UserProfile / MoodAnalysis data classes
- Keyword-based mood analysis
- Recommender scoring, ranking, and explanations
- Input validation and guardrails
- AI response parsing
- Full text-to-recommendation pipeline

Run from project root:
    python -m pytest tests/ -v
"""

from src.recommender import (
    Song, UserProfile, MoodAnalysis, Recommender,
    analyze_mood_keywords, build_mood_prompt, parse_mood_response,
    validate_input_text, sanitize_text,
    load_songs, recommend_songs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _sample_songs() -> list[Song]:
    return [
        Song(id=1, title="Test Pop Track", artist="Test Artist",
             genre="pop", mood="happy", energy=0.8, tempo_bpm=120,
             valence=0.9, danceability=0.8, acousticness=0.2),
        Song(id=2, title="Chill Lofi Loop", artist="Test Artist",
             genre="lofi", mood="chill", energy=0.4, tempo_bpm=80,
             valence=0.6, danceability=0.5, acousticness=0.9),
        Song(id=3, title="Heavy Rock Riff", artist="Rock Band",
             genre="rock", mood="aggressive", energy=0.92, tempo_bpm=155,
             valence=0.35, danceability=0.65, acousticness=0.08),
        Song(id=4, title="Ambient Dreams", artist="Drift",
             genre="ambient", mood="peaceful", energy=0.2, tempo_bpm=60,
             valence=0.55, danceability=0.3, acousticness=0.95),
        Song(id=5, title="Jazz Café", artist="Smooth Keys",
             genre="jazz", mood="relaxed", energy=0.38, tempo_bpm=92,
             valence=0.72, danceability=0.52, acousticness=0.88),
    ]


def make_small_recommender() -> Recommender:
    return Recommender(_sample_songs())


# ── Recommender.recommend ────────────────────────────────────────────────

def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, target_valence=0.9, likes_acoustic=False)
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)
    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommend_k_greater_than_catalog():
    rec = make_small_recommender()
    user = UserProfile("pop", "happy", 0.5, 0.9, False)
    results = rec.recommend(user, k=999)
    assert len(results) == len(rec.songs)


def test_recommend_respects_k():
    rec = make_small_recommender()
    user = UserProfile("pop", "happy", 0.5, 0.9, False)
    assert len(rec.recommend(user, k=1)) == 1
    assert len(rec.recommend(user, k=3)) == 3


# ── Recommender.explain_recommendation ───────────────────────────────────

def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, target_valence=0.9, likes_acoustic=False)
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_explain_includes_genre_match():
    user = UserProfile("pop", "happy", 0.8, 0.9, False)
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert "genre match" in explanation.lower()


def test_explain_includes_mood_match():
    user = UserProfile("pop", "happy", 0.8, 0.9, False)
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert "mood match" in explanation.lower()


def test_explain_mentions_acoustic_when_relevant():
    user = UserProfile("lofi", "chill", 0.4, 0.6, likes_acoustic=True)
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[1])
    assert "acoustic" in explanation.lower()


# ── Scoring logic ────────────────────────────────────────────────────────

def test_genre_match_boosts_score():
    rec = make_small_recommender()
    user_pop = UserProfile("pop", "happy", 0.5, 0.9, False)
    user_rock = UserProfile("rock", "happy", 0.5, 0.9, False)
    pop_song = rec.songs[0]
    assert rec.score_song(user_pop, pop_song) > rec.score_song(user_rock, pop_song)


def test_mood_match_boosts_score():
    rec = make_small_recommender()
    user_happy = UserProfile("jazz", "happy", 0.5, 0.9, False)
    user_chill = UserProfile("jazz", "chill", 0.5, 0.6, False)
    happy_song = rec.songs[0]
    assert rec.score_song(user_happy, happy_song) > rec.score_song(user_chill, happy_song)


def test_acoustic_bonus_applied():
    rec = make_small_recommender()
    acoustic_song = rec.songs[1]
    user_yes = UserProfile("lofi", "chill", 0.4, 0.6, likes_acoustic=True)
    user_no = UserProfile("lofi", "chill", 0.4, 0.6, likes_acoustic=False)
    assert rec.score_song(user_yes, acoustic_song) > rec.score_song(user_no, acoustic_song)


def test_energy_similarity_affects_score():
    rec = make_small_recommender()
    high_energy_song = rec.songs[0]
    user_close = UserProfile("jazz", "relaxed", 0.8, 0.7, False)
    user_far = UserProfile("jazz", "relaxed", 0.1, 0.7, False)
    assert rec.score_song(user_close, high_energy_song) > rec.score_song(user_far, high_energy_song)


# ── Keyword mood analysis ────────────────────────────────────────────────

def test_analyze_mood_sad_text():
    result = analyze_mood_keywords("I feel so sad and lonely tonight")
    assert result.detected_mood == "melancholic"
    assert result.energy_level < 0.5
    assert result.valence_level < 0.5
    assert result.confidence > 0.0


def test_analyze_mood_party_text():
    result = analyze_mood_keywords("Let's go to a party and dance all night!")
    assert result.detected_mood == "energetic"
    assert result.energy_level > 0.7
    assert result.suggested_genre == "electronic"


def test_analyze_mood_study_text():
    result = analyze_mood_keywords("I need to focus and study for my exam")
    assert result.detected_mood == "focused"
    assert result.suggested_genre == "lofi"


def test_analyze_mood_angry_text():
    result = analyze_mood_keywords("I'm so angry and frustrated right now")
    assert result.detected_mood == "aggressive"
    assert result.energy_level > 0.7
    assert result.suggested_genre == "rock"


def test_analyze_mood_no_keywords():
    result = analyze_mood_keywords("The cat sat on the mat")
    assert result.confidence < 0.5
    assert result.detected_mood == "chill"


def test_analyze_mood_returns_mood_analysis():
    result = analyze_mood_keywords("happy day")
    assert isinstance(result, MoodAnalysis)
    assert 0.0 <= result.energy_level <= 1.0
    assert 0.0 <= result.valence_level <= 1.0
    assert 0.0 <= result.confidence <= 1.0


# ── Input validation / guardrails ────────────────────────────────────────

def test_validate_empty_input():
    valid, msg = validate_input_text("")
    assert not valid
    assert "empty" in msg.lower()


def test_validate_too_short():
    valid, msg = validate_input_text("x")
    assert not valid


def test_validate_too_long():
    valid, msg = validate_input_text("a" * 3000)
    assert not valid
    assert "long" in msg.lower()


def test_validate_script_injection():
    valid, msg = validate_input_text("<script>alert('xss')</script>")
    assert not valid


def test_validate_normal_input():
    valid, msg = validate_input_text("I feel happy today")
    assert valid
    assert msg == ""


def test_sanitize_strips_html():
    result = sanitize_text("Hello <b>world</b>!")
    assert "<b>" not in result
    assert "Hello" in result


# ── MoodAnalysis → UserProfile conversion ────────────────────────────────

def test_mood_analysis_to_user_profile():
    analysis = MoodAnalysis(text="test", detected_mood="chill", energy_level=0.4,
                            valence_level=0.6, suggested_genre="lofi",
                            likes_acoustic=True, confidence=0.8, reasoning="test")
    profile = analysis.to_user_profile()
    assert isinstance(profile, UserProfile)
    assert profile.favorite_genre == "lofi"
    assert profile.target_energy == 0.4


def test_mood_analysis_to_prefs_dict():
    analysis = MoodAnalysis(text="test", detected_mood="energetic", energy_level=0.9,
                            valence_level=0.8, suggested_genre="pop",
                            likes_acoustic=False, confidence=0.7, reasoning="test")
    prefs = analysis.to_prefs_dict()
    assert prefs["favorite_genre"] == "pop"
    assert prefs["target_energy"] == 0.9


# ── AI response parsing ─────────────────────────────────────────────────

def test_parse_mood_response_valid_json():
    raw = '{"detected_mood":"happy","energy_level":0.8,"valence_level":0.9,"suggested_genre":"pop","likes_acoustic":false,"confidence":0.9,"reasoning":"upbeat"}'
    result = parse_mood_response("I feel great!", raw)
    assert result.detected_mood == "happy"
    assert result.source == "ai"


def test_parse_mood_response_with_markdown_fences():
    raw = '```json\n{"detected_mood":"chill","energy_level":0.3,"valence_level":0.5,"suggested_genre":"lofi","likes_acoustic":true,"confidence":0.7,"reasoning":"relaxed"}\n```'
    result = parse_mood_response("just vibing", raw)
    assert result.detected_mood == "chill"


def test_parse_mood_response_invalid_json_falls_back():
    result = parse_mood_response("I'm really angry!!!", "not valid json {{{")
    assert isinstance(result, MoodAnalysis)
    assert result.detected_mood == "aggressive"
    assert result.source == "keyword"


def test_parse_mood_response_unknown_mood_falls_back():
    raw = '{"detected_mood":"UNKNOWNMOOD","energy_level":0.5,"valence_level":0.5,"suggested_genre":"pop","likes_acoustic":false,"confidence":0.5,"reasoning":"test"}'
    result = parse_mood_response("test text", raw)
    assert result.source == "keyword"


# ── Prompt building ──────────────────────────────────────────────────────

def test_build_mood_prompt_contains_text():
    prompt = build_mood_prompt("I feel so happy today")
    assert "I feel so happy today" in prompt
    assert "pop" in prompt
    assert "happy" in prompt


# ── Full pipeline ────────────────────────────────────────────────────────

def test_recommend_from_text_returns_results():
    rec = make_small_recommender()
    analysis, recs = rec.recommend_from_text("I want to dance at a party!", k=3)
    assert isinstance(analysis, MoodAnalysis)
    assert len(recs) == 3


def test_recommend_from_text_sad_prefers_low_energy():
    rec = make_small_recommender()
    analysis, recs = rec.recommend_from_text("I'm feeling lonely and sad")
    assert analysis.energy_level < 0.5
    assert recs[0][0].energy < 0.6


def test_recommend_from_text_workout_prefers_high_energy():
    rec = make_small_recommender()
    analysis, recs = rec.recommend_from_text("heading to the gym for a workout")
    assert analysis.energy_level > 0.7
    assert recs[0][0].energy > 0.5


def test_recommend_from_text_rejects_empty():
    rec = make_small_recommender()
    try:
        rec.recommend_from_text("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── Functional API ───────────────────────────────────────────────────────

def test_functional_recommend_songs():
    songs = [
        {"id": 1, "title": "A", "artist": "X", "genre": "pop", "mood": "happy",
         "energy": 0.8, "tempo_bpm": 120, "valence": 0.9, "danceability": 0.8, "acousticness": 0.2},
        {"id": 2, "title": "B", "artist": "Y", "genre": "lofi", "mood": "chill",
         "energy": 0.4, "tempo_bpm": 80, "valence": 0.6, "danceability": 0.5, "acousticness": 0.9},
    ]
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "target_valence": 0.9, "likes_acoustic": False}
    results = recommend_songs(prefs, songs, k=2)
    assert len(results) == 2
    assert results[0][0]["genre"] == "pop"
