import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Load and parse songs from a CSV file with type conversions."""
    print(f"Loading songs from {csv_path}...")
    
    songs = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert numerical columns to appropriate types
            row['id'] = int(row['id'])
            row['energy'] = float(row['energy'])
            row['tempo_bpm'] = float(row['tempo_bpm'])
            row['valence'] = float(row['valence'])
            row['danceability'] = float(row['danceability'])
            row['acousticness'] = float(row['acousticness'])
            
            songs.append(row)
    
    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score and rank songs by user preference, returning top k recommendations with explanations."""
    def calculate_score(user: Dict, song: Dict) -> Tuple[float, str]:
        """Calculate score and explanation for a single song."""
        score = 0.0
        reasons = []
        
        # Genre match: +2.0
        if song['genre'].lower() == user['favorite_genre'].lower():
            score += 2.0
            reasons.append("genre match")
        
        # Mood match: +1.0
        if song['mood'].lower() == user['favorite_mood'].lower():
            score += 1.0
            reasons.append("mood match")
        
        # Energy similarity: ×1.5
        energy_sim = 1.0 - abs(user['target_energy'] - song['energy'])
        energy_score = energy_sim * 1.5
        score += energy_score
        
        # Valence similarity: ×0.75
        valence_sim = 1.0 - abs(user['target_energy'] - song['valence'])
        valence_score = valence_sim * 0.75
        score += valence_score
        
        # Danceability: ×0.5
        danceability_score = song['danceability'] * 0.5
        score += danceability_score
        
        # Acoustic bonus: +0.25
        if user.get('likes_acoustic', False):
            if song['acousticness'] > 0.7:
                score += 0.25
                reasons.append("acoustic preference")
        
        explanation = " + ".join(reasons) if reasons else "energy/audio profile match"
        return score, explanation
    
    scored_songs = [
        (song, *calculate_score(user_prefs, song))
        for song in songs
    ]
    
    top_k = sorted(scored_songs, key=lambda x: x[1], reverse=True)[:k]
    
    return top_k

def calculate_score(user: UserProfile, song: Song) -> float:
    score = 0.0
    
    # Categorical matches
    if song.genre.lower() == user.favorite_genre.lower():
        score += 2.0
    if song.mood.lower() == user.favorite_mood.lower():
        score += 1.0
    
    # Continuous similarities
    energy_similarity = 1.0 - abs(user.target_energy - song.energy)
    score += energy_similarity * 1.5
    
    valence_similarity = 1.0 - abs(user.target_energy - song.valence)  # approximate
    score += valence_similarity * 0.75
    
    score += song.danceability * 0.5
    
    # Optional fine-tuning
    if abs(user.target_tempo - song.tempo_bpm) <= 15:  # within 15 bpm
        score += 0.25
    
    if user.likes_acoustic and song.acousticness > 0.7:
        score += 0.25
    
    return score