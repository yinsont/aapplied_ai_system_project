"""
recommender.py — AI-Powered Music Mood Recommender

Core module containing:
- Data classes (Song, UserProfile, MoodAnalysis)
- Keyword-based mood detection (offline fallback)
- AI-powered mood detection (Anthropic API integration)
- Recommendation engine with scoring, ranking, and explanations
- Input validation, guardrails, and structured logging
"""

import csv
import json
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("recommender")
logger.setLevel(logging.DEBUG)

# File handler — persistent log of all decisions
_file_handler = logging.FileHandler("recommender.log", mode="a")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_file_handler)

# Console handler — info and above
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_console_handler)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """Represents a song and its audio attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

    def __post_init__(self):
        """Validate that numerical fields are within expected ranges."""
        for attr in ("energy", "valence", "danceability", "acousticness"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                logger.warning(f"Song '{self.title}': {attr}={val} outside [0,1] range")


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

    def __post_init__(self):
        """Clamp energy to valid range and log guardrail activations."""
        if not (0.0 <= self.target_energy <= 1.0):
            clamped = max(0.0, min(1.0, self.target_energy))
            logger.warning(
                f"Guardrail: target_energy={self.target_energy} clamped to {clamped}"
            )
            self.target_energy = clamped


@dataclass
class MoodAnalysis:
    """
    Result of AI mood detection from free-form text.
    Contains the derived emotional profile and the raw user text.
    """
    text: str
    detected_mood: str
    energy_level: float
    valence_level: float
    suggested_genre: str
    likes_acoustic: bool
    confidence: float
    reasoning: str
    source: str = "keyword"  # "keyword" or "ai"

    def __post_init__(self):
        """Clamp all float fields to [0, 1]."""
        for attr in ("energy_level", "valence_level", "confidence"):
            val = getattr(self, attr)
            clamped = max(0.0, min(1.0, float(val)))
            if clamped != val:
                logger.warning(f"MoodAnalysis: {attr}={val} clamped to {clamped}")
            setattr(self, attr, round(clamped, 2))

    def to_user_profile(self) -> "UserProfile":
        """Convert a mood analysis into a UserProfile for the recommender."""
        return UserProfile(
            favorite_genre=self.suggested_genre,
            favorite_mood=self.detected_mood,
            target_energy=self.energy_level,
            likes_acoustic=self.likes_acoustic,
        )

    def to_prefs_dict(self) -> Dict:
        """Convert to the dict format used by recommend_songs()."""
        return {
            "favorite_genre": self.suggested_genre,
            "favorite_mood": self.detected_mood,
            "target_energy": self.energy_level,
            "likes_acoustic": self.likes_acoustic,
        }


# ---------------------------------------------------------------------------
# Input Validation / Guardrails
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 2000  # characters
MIN_INPUT_LENGTH = 2


def validate_input_text(text: str) -> Tuple[bool, str]:
    """
    Validate user input text. Returns (is_valid, error_message).
    Guardrails:
    - Reject empty or too-short input
    - Reject excessively long input
    - Strip HTML/script tags as a safety measure
    """
    if not text or not text.strip():
        return False, "Input text is empty"

    stripped = text.strip()

    if len(stripped) < MIN_INPUT_LENGTH:
        return False, f"Input too short (min {MIN_INPUT_LENGTH} characters)"

    if len(stripped) > MAX_INPUT_LENGTH:
        return False, f"Input too long (max {MAX_INPUT_LENGTH} characters)"

    # Basic sanitization — strip potential injection patterns
    if re.search(r"<script|<iframe|javascript:", stripped, re.IGNORECASE):
        logger.warning("Guardrail: rejected potentially unsafe input")
        return False, "Input contains disallowed content"

    return True, ""


def sanitize_text(text: str) -> str:
    """Clean user input text for safe processing."""
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Mood Detection — keyword-based fallback (no API required)
# ---------------------------------------------------------------------------

_KEYWORD_RULES: List[Tuple[List[str], str, float, float, str, bool]] = [
    # High-energy / positive
    (["party", "dance", "club", "wild", "hype"],          "energetic",    0.90, 0.80, "electronic", False),
    (["workout", "gym", "run", "exercise", "pump"],        "intense",      0.92, 0.70, "hip-hop",    False),
    (["happy", "great", "amazing", "wonderful", "joy"],    "happy",        0.75, 0.90, "pop",        False),
    (["excited", "thrilled", "pumped", "stoked"],          "energetic",    0.85, 0.85, "pop",        False),

    # Medium energy
    (["focus", "study", "concentrate", "work", "coding"],  "focused",      0.40, 0.55, "lofi",       False),
    (["chill", "relax", "unwind", "calm", "mellow"],       "chill",        0.35, 0.60, "lofi",       True),
    (["drive", "road", "travel", "journey"],               "adventurous",  0.70, 0.65, "indie rock", False),
    (["coffee", "morning", "cozy", "warm"],                "relaxed",      0.40, 0.70, "jazz",       True),
    (["romantic", "love", "date", "crush"],                "romantic",     0.45, 0.75, "soul",       True),

    # Low energy / negative
    (["sad", "cry", "heartbreak", "miss", "lonely"],       "melancholic",  0.30, 0.25, "indie folk", True),
    (["angry", "furious", "rage", "frustrated", "mad"],    "aggressive",   0.90, 0.20, "rock",       False),
    (["anxious", "stress", "overwhelm", "nervous"],        "introspective",0.35, 0.30, "ambient",    True),
    (["tired", "exhausted", "sleepy", "drained"],          "peaceful",     0.20, 0.40, "ambient",    True),
    (["nostalgic", "remember", "memories", "past"],        "moody",        0.45, 0.50, "synthwave",  False),
    (["dark", "gloomy", "void", "empty", "numb"],          "introspective",0.25, 0.20, "dark ambient",True),

    # Creative / abstract
    (["inspired", "creative", "flow", "art", "paint"],     "inspirational",0.55, 0.75, "classical",  True),
    (["dream", "float", "space", "ethereal"],              "dreamy",       0.40, 0.65, "ambient",    True),
    (["rain", "thunder", "storm"],                         "moody",        0.50, 0.40, "jazz",       True),
]


def analyze_mood_keywords(text: str) -> MoodAnalysis:
    """
    Analyze mood from text using keyword matching.
    Works offline — no API needed. Used as a fallback when AI is unavailable.
    """
    logger.info(f"Keyword analysis on: '{text[:80]}...'")
    lower = text.lower()
    best_score = 0
    best_match = None

    for keywords, mood, energy, valence, genre, acoustic in _KEYWORD_RULES:
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_score:
            best_score = hits
            best_match = (mood, energy, valence, genre, acoustic, keywords)

    if best_match:
        mood, energy, valence, genre, acoustic, matched_kws = best_match
        matched = [kw for kw in matched_kws if kw in lower]
        confidence = min(0.4 + (best_score * 0.15), 0.85)
        reasoning = f"Keyword matches: {', '.join(matched)}"
        logger.info(f"Mood detected: {mood} (confidence={confidence:.2f}, keywords={matched})")
    else:
        mood, energy, valence, genre, acoustic = "chill", 0.50, 0.55, "lofi", False
        confidence = 0.2
        reasoning = "No strong keyword signals detected — defaulting to neutral/chill"
        logger.info("No keyword matches found — using default profile")

    return MoodAnalysis(
        text=text,
        detected_mood=mood,
        energy_level=round(energy, 2),
        valence_level=round(valence, 2),
        suggested_genre=genre,
        likes_acoustic=acoustic,
        confidence=round(confidence, 2),
        reasoning=reasoning,
        source="keyword",
    )


# ---------------------------------------------------------------------------
# Mood Detection — AI-powered (uses Anthropic API)
# ---------------------------------------------------------------------------

KNOWN_GENRES = [
    "pop", "rock", "lofi", "jazz", "electronic", "hip-hop", "soul", "metal",
    "classical", "reggae", "indie rock", "indie pop", "ambient", "synthwave",
    "dark ambient", "country", "blues", "punk", "house", "dubstep", "techno",
    "afrobeats", "downtempo", "indie folk", "progressive rock", "folk",
    "bossa nova", "funk", "gospel", "singer-songwriter",
]

KNOWN_MOODS = [
    "happy", "chill", "intense", "relaxed", "moody", "focused", "dreamy",
    "energetic", "romantic", "aggressive", "introspective", "inspirational",
    "playful", "adventurous", "melancholic", "peaceful", "dark", "nostalgic",
]

MOOD_ANALYSIS_PROMPT = """You are a music mood analyst. Given the user's text describing how they feel or what they're doing, analyze their emotional state and suggest music parameters.

Available genres: {genres}
Available moods: {moods}

Respond with ONLY a JSON object (no markdown, no explanation) with these fields:
- "detected_mood": one of the available moods above
- "energy_level": float 0.0-1.0 (0=very calm, 1=very intense)
- "valence_level": float 0.0-1.0 (0=very negative/sad, 1=very positive/happy)
- "suggested_genre": one of the available genres above
- "likes_acoustic": boolean
- "confidence": float 0.0-1.0
- "reasoning": brief explanation of your analysis

User text: "{text}"
"""


def build_mood_prompt(text: str) -> str:
    """Build the prompt for the AI mood analysis."""
    return MOOD_ANALYSIS_PROMPT.format(
        genres=", ".join(KNOWN_GENRES),
        moods=", ".join(KNOWN_MOODS),
        text=text,
    )


def parse_mood_response(text: str, raw_response: str) -> MoodAnalysis:
    """
    Parse a JSON response from the AI into a MoodAnalysis.
    Falls back to keyword analysis if parsing fails.
    """
    try:
        cleaned = re.sub(r"```json\s*|```\s*", "", raw_response).strip()
        data = json.loads(cleaned)

        required = ["detected_mood", "energy_level", "suggested_genre"]
        for field_name in required:
            if field_name not in data:
                raise KeyError(f"Missing required field: {field_name}")

        detected_mood = data["detected_mood"]
        if detected_mood not in KNOWN_MOODS:
            logger.warning(f"AI returned unknown mood '{detected_mood}', falling back")
            raise ValueError(f"Unknown mood: {detected_mood}")

        suggested_genre = data["suggested_genre"]
        if suggested_genre not in KNOWN_GENRES:
            logger.warning(f"AI returned unknown genre '{suggested_genre}', falling back")
            raise ValueError(f"Unknown genre: {suggested_genre}")

        result = MoodAnalysis(
            text=text,
            detected_mood=detected_mood,
            energy_level=float(data.get("energy_level", 0.5)),
            valence_level=float(data.get("valence_level", 0.5)),
            suggested_genre=suggested_genre,
            likes_acoustic=bool(data.get("likes_acoustic", False)),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", "AI analysis"),
            source="ai",
        )
        logger.info(f"AI analysis successful: mood={result.detected_mood}, confidence={result.confidence}")
        return result

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"AI response parsing failed ({e}), falling back to keywords")
        return analyze_mood_keywords(text)


# ---------------------------------------------------------------------------
# Recommender Class
# ---------------------------------------------------------------------------

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Scores songs against a user profile and returns ranked results with explanations.
    """
    def __init__(self, songs: List[Song]):
        if not songs:
            logger.warning("Recommender initialized with empty song list")
        self.songs = songs
        logger.info(f"Recommender initialized with {len(songs)} songs")

    def score_song(self, user: UserProfile, song: Song) -> float:
        """Calculate a recommendation score for a song given a user profile."""
        score = 0.0

        if song.genre.lower() == user.favorite_genre.lower():
            score += 2.0
        if song.mood.lower() == user.favorite_mood.lower():
            score += 1.0

        energy_sim = 1.0 - abs(user.target_energy - song.energy)
        score += energy_sim * 1.5

        valence_sim = 1.0 - abs(user.target_energy - song.valence)
        score += valence_sim * 0.75

        score += song.danceability * 0.5

        if user.likes_acoustic and song.acousticness > 0.7:
            score += 0.25

        return round(score, 4)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs sorted by score descending."""
        if k < 1:
            logger.warning(f"Guardrail: k={k} is invalid, setting to 1")
            k = 1

        scored = [(song, self.score_song(user, song)) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = [song for song, _score in scored[:k]]
        if results:
            logger.debug(f"recommend(k={k}): top song='{results[0].title}' score={scored[0][1]:.2f}")
        return results

    def recommend_with_scores(self, user: UserProfile, k: int = 5) -> List[Tuple[Song, float, str]]:
        """Return top-k songs with scores and explanations."""
        results = []
        for song in self.songs:
            score = self.score_song(user, song)
            explanation = self.explain_recommendation(user, song)
            results.append((song, score, explanation))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Generate a human-readable explanation of why this song was recommended."""
        reasons = []

        if song.genre.lower() == user.favorite_genre.lower():
            reasons.append(f"genre match ({song.genre})")
        if song.mood.lower() == user.favorite_mood.lower():
            reasons.append(f"mood match ({song.mood})")

        energy_sim = 1.0 - abs(user.target_energy - song.energy)
        if energy_sim > 0.8:
            reasons.append(f"energy closely matches ({song.energy:.0%})")
        elif energy_sim > 0.5:
            reasons.append(f"decent energy fit ({song.energy:.0%})")

        if user.likes_acoustic and song.acousticness > 0.7:
            reasons.append("acoustic vibes")

        if song.danceability > 0.75:
            reasons.append("very danceable")

        if not reasons:
            reasons.append("general audio profile match")

        return " + ".join(reasons)

    def recommend_from_text(self, text: str, k: int = 5) -> Tuple[MoodAnalysis, List[Tuple[Song, float, str]]]:
        """
        Full pipeline: text → mood analysis → recommendations.
        Includes input validation and guardrails.
        """
        is_valid, error_msg = validate_input_text(text)
        if not is_valid:
            logger.error(f"Input validation failed: {error_msg}")
            raise ValueError(f"Invalid input: {error_msg}")

        clean_text = sanitize_text(text)
        logger.info(f"Pipeline started for: '{clean_text[:80]}...'")

        analysis = analyze_mood_keywords(clean_text)
        profile = analysis.to_user_profile()
        recs = self.recommend_with_scores(profile, k=k)

        logger.info(
            f"Pipeline complete: mood={analysis.detected_mood}, "
            f"confidence={analysis.confidence}, top_song='{recs[0][0].title if recs else 'N/A'}'"
        )
        return analysis, recs


# ---------------------------------------------------------------------------
# Functional API (dict-based, backward compatible with original project)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Load and parse songs from a CSV file with type conversions."""
    logger.info(f"Loading songs from {csv_path}...")

    songs = []
    try:
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row["id"] = int(row["id"])
                row["energy"] = float(row["energy"])
                row["tempo_bpm"] = float(row["tempo_bpm"])
                row["valence"] = float(row["valence"])
                row["danceability"] = float(row["danceability"])
                row["acousticness"] = float(row["acousticness"])
                songs.append(row)
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise

    logger.info(f"Loaded {len(songs)} songs successfully")
    return songs


def load_songs_as_objects(csv_path: str) -> List[Song]:
    """Load songs as Song dataclass instances (for OOP usage)."""
    raw = load_songs(csv_path)
    return [
        Song(
            id=row["id"], title=row["title"], artist=row["artist"],
            genre=row["genre"], mood=row["mood"], energy=row["energy"],
            tempo_bpm=row["tempo_bpm"], valence=row["valence"],
            danceability=row["danceability"], acousticness=row["acousticness"],
        )
        for row in raw
    ]


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score and rank songs by user preference, returning top k recommendations with explanations."""

    def calculate_score(user: Dict, song: Dict) -> Tuple[float, str]:
        score = 0.0
        reasons = []

        if song["genre"].lower() == user["favorite_genre"].lower():
            score += 2.0
            reasons.append("genre match")
        if song["mood"].lower() == user["favorite_mood"].lower():
            score += 1.0
            reasons.append("mood match")

        energy_sim = 1.0 - abs(user["target_energy"] - song["energy"])
        score += energy_sim * 1.5

        valence_sim = 1.0 - abs(user["target_energy"] - song["valence"])
        score += valence_sim * 0.75

        score += song["danceability"] * 0.5

        if user.get("likes_acoustic", False):
            if song["acousticness"] > 0.7:
                score += 0.25
                reasons.append("acoustic preference")

        explanation = " + ".join(reasons) if reasons else "energy/audio profile match"
        return score, explanation

    scored_songs = [(song, *calculate_score(user_prefs, song)) for song in songs]
    top_k = sorted(scored_songs, key=lambda x: x[1], reverse=True)[:k]
    return top_k
