# Model Card: AI Music Mood Recommender

## Model Overview

This system uses natural language processing to detect a user's emotional state from free-form text, then maps that emotional profile onto a weighted scoring formula to recommend songs from a 250 song catalog.

**Input:** Free form text describing how the user feels (e.g., "I'm feeling sad and lonely")  
**Output:** Ranked list of song recommendations with scores and human readable explanations

**AI Components Used:**
- Keyword based mood detection (18 rule sets, offline)
- LLM powered mood analysis via Anthropic API (optional, for nuanced inputs)
- Confidence scoring on all mood detections
- Input validation guardrails

---

## Limitations and Biases

**Keyword coverage is English-centric.** The 18 keyword rule sets only cover English emotional vocabulary. Non-English text or culturally specific emotional expressions will fall through to the neutral default, producing generic recommendations.

**Genre representation is uneven.** The 100-song catalog over-represents Western genres (pop, rock, electronic, hip-hop) and under-represents genres like K-pop, Afrobeats, Latin, and classical Indian music. Users whose taste falls outside the catalog's range will get lower-quality recommendations.

**Mood is simplified to a single category.** Real human emotion is complex and layered. Someone can feel "nostalgic but also energized." The system maps to a single mood label, which loses nuance. The confidence score partially addresses this by signaling when the system is uncertain.

**Energy/valence are proxies, not measurements.** The audio features (energy, valence, danceability) are synthetic values in this dataset, not derived from actual audio analysis. In a production system these would come from a service like Spotify's audio features API.

**No personalization over time.** The system treats each interaction independently. It doesn't learn from feedback or build a user taste profile across sessions.

---

## Potential for Misuse

**Mood manipulation:** A recommendation system that responds to emotional state could theoretically be used to reinforce negative emotional patterns (e.g., always recommending sad music to a sad user). The current system matches mood-to-mood, which could deepen negative spirals.

**Mitigation:** A production version should include a "mood lift" option that recommends slightly more positive music than the detected mood, and should flag when users repeatedly express distress.

**Data privacy:** The system logs all user inputs to `recommender.log` for debugging. In a production setting, this would need proper anonymization, retention policies, and user consent.

**Mitigation:** The guardrail layer already strips HTML and rejects script injection. For production, input logging should be opt-in and PII should be redacted.

---

## Testing Surprises

**The neutral fallback was more useful than expected.** Initially I thought "The cat sat on the mat" producing a chill/lofi playlist was a failure; but in practice, users who type ambiguous or non-emotional text probably do want something neutral and unobtrusive. The 20% confidence score correctly signals low certainty.

**Keyword overlap caused unexpected behavior.** The word "run" appears in both workout contexts ("going for a run") and escape contexts ("I want to run away from everything"). The system maps both to high energy workout music, which is wrong for the second case. This showed me that keyword-based NLP has real limits compared to LLM-based understanding.

**Guardrails caught real issues.** During testing, I pasted a long paragraph from a news article (>2000 chars) and the system correctly rejected it rather than trying to extract mood from irrelevant text. The length limit turned out to be a practical quality filter, not just a safety measure.

---

## AI Collaboration During This Project

**Helpful suggestion:** When I was designing the mood detection system, Claude suggested structuring the keyword rules as tuples with all parameters inline rather than using separate dictionaries for each field. This made the rules much more readable and easy to extend, I could see all 18 mood mappings at a glance and quickly spot gaps in coverage.

**Flawed suggestion:** Claude initially generated a `calculate_score` function at module level that referenced `user.target_tempo`, a field that doesn't exist on the `UserProfile` dataclass. This would have caused an `AttributeError` at runtime. I caught it because the test suite flagged the inconsistency between the dataclass definition and the function's assumptions. This reinforced why writing tests alongside code matters, especially when using AI generated code.

---

## Reflection: What This Project Says About Me as an AI Engineer

The development of this project and the previous projects are a stepping stone to what me and others may develop in the near future. This keyword based mood analyzer, although conceptually simple, also challenged my innate abilities to develop input validation, confidence scoring, fallbacks, tests, and debug explanations. These practices are what we need to make AI systems trustworthy. At this moment, rather than "making AI smarter", we can just use it to automate simple tasks like this, or even deleting spam emails. 