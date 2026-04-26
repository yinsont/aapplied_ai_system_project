"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}\n")

    # Define distinct user profiles
    user_profiles = [
        {
            "name": "Angry Calm-Seeker",
            "prefs": {
                "favorite_genre": "rock",
                "favorite_mood": "angry",
                "target_energy": 0.1,  # Contradiction: angry music is typically high-energy
                "likes_acoustic": False
            }
        },
        {
            "name": "Hyperactive Acoustic Lover",
            "prefs": {
                "favorite_genre": "pop",
                "favorite_mood": "energetic",
                "target_energy": 0.95,  # Extremely high
                "likes_acoustic": True   # Acoustic songs are typically low-energy
            }
        },
        {
            "name": "Zero Energy Minimalist",
            "prefs": {
                "favorite_genre": "lofi",
                "favorite_mood": "peaceful",
                "target_energy": 0.0,  # Minimum boundary
                "likes_acoustic": True
            }
        }
    ]

    # Generate recommendations for each profile
    for profile in user_profiles:
        recommendations = recommend_songs(profile["prefs"], songs, k=5)

        print("="*70)
        genre = profile["prefs"]['favorite_genre'].upper()
        mood = profile["prefs"]['favorite_mood'].upper()
        print(f"🎵 TOP RECOMMENDATIONS FOR {profile['name'].upper()}")
        print(f"   ({genre} • {mood} MOOD • Energy: {profile['prefs']['target_energy']:.2f})")
        print("="*70 + "\n")
        
        for i, rec in enumerate(recommendations, 1):
            song, score, explanation = rec
            print(f"#{i} {song['title']}")
            print(f"    Artist: {song['artist']}")
            print(f"    Score:  {score:.2f}/6.25")
            print(f"    Reasons: {explanation}")
            print()
        print()


if __name__ == "__main__":
    main()
