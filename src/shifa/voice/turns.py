"""Semantic end-of-turn detection: silence alone doesn't mean the caller is done.

The VAD ends a turn after ~1.3 s of silence, but people pause mid-sentence ("mujhe kal ... aur ..."). If the
transcript so far ends on a word that can't end an Urdu/English sentence (a conjunction or postposition), we wait
a little for the rest instead of answering half a sentence. Production systems use a small model for this
(e.g. LiveKit's turn detector); a word list is the zero-cost version and covers the common cases.
"""

import re

WAIT_FOR_REST_S = 1.8

_CONTINUATION = {
    # Urdu script: conjunctions and postpositions that need something after them
    "اور", "لیکن", "مگر", "کہ", "کیونکہ", "یا", "اگر", "تو", "کے", "کی", "کا", "کو", "سے", "میں", "پر", "لیے", "ساتھ",
    "جب", "بلکہ",
    # Roman Urdu
    "aur", "lekin", "magar", "ke", "ki", "ka", "ko", "se", "ya", "agar", "kyunke", "kyunki", "liye", "saath",
    # English
    "and", "but", "or", "so", "because", "if", "the", "a", "an", "to", "with", "for", "of", "at", "on", "my", "is",
}


def looks_incomplete(text: str) -> bool:
    words = re.sub(r"[^\w\s]", " ", text).split()
    return bool(words) and words[-1].lower() in _CONTINUATION
