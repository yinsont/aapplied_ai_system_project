# 🎧 AI Music Mood Recommender

An AI-powered music recommendation system that detects your mood from natural language text and generates personalized playlists. Built as a final project for **AI110: Foundations of AI Engineering** (CodePath, Spring 2026).

## 📌 Base Project

This project extends **Project 2 (Module 2): Music Song Recommender**, which originally scored songs using a weighted formula based on manually defined user profiles (genre, mood, energy, acoustic preferences). The original system required users to specify numerical parameters — this extension replaces that with natural language understanding.

## 🎯 What It Does

Instead of manually configuring `target_energy: 0.8` and `favorite_genre: "lofi"`, you simply tell the system how you're feeling:

> *"I just had a terrible breakup and can't stop crying"*

The AI analyzes your text, detects emotional parameters (mood: melancholic, energy: 30%, genre: indie folk), and feeds them into the recommendation engine to generate a personalized playlist with scored explanations.

## 🏗️ Architecture Overview

![System Architecture](assets/architecture-diagram.png)

```
User Input (text) → Input Validator (guardrails) → Mood Analyzer (keyword/AI)
                                                          ↓
Song Catalog (CSV) → Scoring Engine (weighted formula) ← MoodAnalysis → UserProfile
                          ↓
                    Ranked Results → Explanation Generator → Playlist Output
```

**Components:**

- **Input Validator** — Guardrails that reject empty, too-short, too-long, or potentially unsafe inputs before processing
- **Mood Analyzer** — Two modes: (1) keyword-based analysis using 18 rule sets mapping emotional language to parameters, (2) AI-powered analysis via Anthropic API for nuanced understanding
- **Scoring Engine** — Weighted formula: genre match (+2.0), mood match (+1.0), energy similarity (×1.5), valence similarity (×0.75), danceability (×0.5), acoustic bonus (+0.25)
- **Explanation Generator** — Produces human-readable reasons for each recommendation
- **Logger** — All decisions, guardrail activations, and errors are recorded to `recommender.log`

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- No external dependencies required (uses only standard library)
- pytest (optional, for running tests)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/applied-ai-system-project.git
cd applied-ai-system-project
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
source. venv/bin/activate pip install -r requirements.txt
```

### Running the System

```bash
# Full demo (profiles + mood detection + guardrails)
python src/main.py

# Evaluation harness (structured pass/fail report)
python evaluate.py

# Run tests
python -m pytest tests/ -v
```

### Project Structure

```
├── .venv/
├── assets/                      # Architecture diagrams and screenshots
│   └── architecture-diagram.png
├── data/
│   └── songs.csv                # 100-song catalog with audio features
├── src/
│   ├── main.py                  # CLI runner with all demo modes
│   └── recommender.py           # Core AI module (mood detection, scoring, guardrails)
├── tests/
│   └── test_recommender.py      # 35 unit tests
├── evaluate.py                  # Test harness / evaluation script
├── model_card.md                # Reflections, ethics, AI collaboration
├── requirements.txt
└── README.md
```

## 💬 Sample Interactions

### Example 1: Sad / Heartbreak
```
📝 "I just had a terrible breakup and I can't stop crying"
─────────────────────────────────────────────
   Detected mood:   melancholic
   Energy level:    30%
   Suggested genre: indie folk
   Confidence:      55%
─────────────────────────────────────────────
   #1 Mountain Echo — Highland Voices
      Score: 4.36 | genre match (indie folk) + acoustic vibes
   #2 Autumn Leaves — Acoustic Dreams
      Score: 4.33 | genre match (indie folk) + acoustic vibes
   #3 Heartstrings — Luna Rivers
      Score: 3.67 | acoustic vibes
```

### Example 2: Workout / High Energy
```
📝 "Heading to the gym, need something to get me pumped up!"
─────────────────────────────────────────────
   Detected mood:   intense
   Energy level:    92%
   Suggested genre: hip-hop
   Confidence:      70%
─────────────────────────────────────────────
   #1 Bass Drop Thunder — Cipher Crew
      Score: 5.45 | genre match (hip-hop) + energy closely matches + very danceable
   #2 Urban Flow — Street Beats
      Score: 5.21 | genre match (hip-hop) + very danceable
   #3 Hip Hop Hustle — Cash Flow
      Score: 5.10 | genre match (hip-hop) + very danceable
```

### Example 3: Guardrail Rejection
```
Test: Script injection attempt
  🚫 Rejected: Input contains disallowed content

Test: Empty input
  🚫 Rejected: Input text is empty
```

## 🧠 Design Decisions

**Why keyword-based analysis as the primary mode?** The keyword system works offline with zero latency and no API costs. It provides a reliable fallback when the AI API is unavailable, and for common emotional expressions it's highly accurate (70% average confidence on test cases). The AI mode is available as an enhancement for more nuanced inputs.

**Why a weighted scoring formula instead of embeddings?** The scoring formula is fully transparent — users can see exactly why each song was recommended (genre match, energy fit, etc.). This supports the rubric's requirement for "clear explanation of how the AI works and why it's trustworthy." Embedding-based approaches would be more powerful but less explainable.

**Why input validation as a separate layer?** Separating guardrails from business logic follows defensive programming principles. The validator catches malformed, empty, or potentially unsafe inputs before they reach the mood analyzer, preventing garbage-in-garbage-out failures.

## 🧪 Testing Summary

**35 unit tests** covering all components — 100% pass rate.

**10 evaluation harness cases** — 33/33 checks passed, 100% pass rate, 62% average confidence.

| Area | Tests | Result |
|------|-------|--------|
| Mood keyword analysis | 6 tests | All pass |
| Input validation/guardrails | 5 tests | All pass |
| Scoring logic | 4 tests | All pass |
| Recommendation ranking | 3 tests | All pass |
| Explanation generation | 4 tests | All pass |
| AI response parsing | 4 tests | All pass |
| Full pipeline (text → recs) | 4 tests | All pass |
| Functional API backward compat | 1 test | All pass |
| Data class conversions | 2 tests | All pass |
| Prompt building | 1 test | All pass |
| Evaluation harness cases | 10 cases | All pass |

The system handled edge cases well: neutral text without mood keywords gracefully defaults to "chill/lofi", script injection is caught and rejected, and AI parsing failures fall back to keyword analysis without crashing.

## 🎥 Demo Walkthrough

> [Loom video link here]

## 📄 License

This project was built for educational purposes as part of CodePath AI110.
