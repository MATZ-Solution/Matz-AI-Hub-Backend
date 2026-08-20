"""
src/utils/spell_correct.py
---------------------------
Spell correction for English queries only.
Skips correction if Urdu words are detected.
"""
from textblob import TextBlob

URDU_MARKERS = [
    "kya", "hai", "kiya", "batao", "mujhe", "mujhey",
    "karo", "hain", "ka", "ki", "ke", "ho", "aur",
    "nahi", "kuch", "bhi", "se", "mein", "ko",
]

def correct_query(text: str) -> str:
    """Correct spelling only for English queries. Skip if Roman Urdu detected."""
    text_lower = text.lower()

    # If any Urdu marker found → skip correction entirely
    if any(marker in text_lower.split() for marker in URDU_MARKERS):
        return text

    try:
        corrected = str(TextBlob(text).correct())
        return corrected
    except Exception:
        return text