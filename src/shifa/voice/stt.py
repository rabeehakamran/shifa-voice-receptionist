"""Speech-to-text: Groq whisper-large-v3 (free tier), with a Devanagari guard for Urdu.

Whisper often hears Urdu as Hindi and writes it in Devanagari, which our Urdu pipeline can't use.
If that happens we re-run with language="ur" to force Urdu script.
"""

import re
from functools import lru_cache

from shifa.config import get_settings

STT_MODEL = "whisper-large-v3"
# A short prompt biases Whisper toward the clinic's vocabulary and mixed Urdu/English.
PROMPT = "Shifa Family Clinic, Dr. Ayesha, Dr. Usman, Dr. Sana, appointment, fees, کل، ڈاکٹر صاحب"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@lru_cache
def _client():
    from groq import Groq

    return Groq(api_key=get_settings().groq_api_key)


def transcribe(wav: bytes, language: str | None = None) -> tuple[str, str]:
    """Returns (text, language). Pass the returned language back on the next turn of the same call:
    once a caller is known to speak Urdu we force language="ur" and skip the second Whisper call
    (measured: 1.85 s with the retry vs ~0.3-0.7 s without)."""
    kwargs = {"language": language} if language else {}
    res = _client().audio.transcriptions.create(file=("turn.wav", wav), model=STT_MODEL, prompt=PROMPT,
                                                response_format="verbose_json", temperature=0.0, **kwargs)
    text, lang = res.text.strip(), language or (getattr(res, "language", "") or "").lower()
    if not language and (_DEVANAGARI.search(text) or lang in {"hindi", "hi"}):
        return transcribe(wav, language="ur")
    return text, "ur" if lang in {"urdu", "ur"} else lang
