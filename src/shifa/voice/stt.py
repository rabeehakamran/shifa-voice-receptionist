"""Speech-to-text: Groq whisper-large-v3 (free tier), with a Devanagari guard and a noise filter.

Two Whisper habits we handle:
- It often hears Urdu as Hindi and writes Devanagari. Then we re-run with language="ur" for that turn only.
  (A per-call "sticky Urdu" shortcut was removed: once forced, an English caller's speech came back in Urdu
  script, and the agent replied "I can only speak Urdu".)
- It invents words from noise and breathing ("ایک سیکنڈ دنیا"). Segments with high no_speech_prob or very
  low avg_logprob are dropped, using the same thresholds as OpenAI's reference implementation.
"""

import re
from functools import lru_cache

from shifa.config import get_settings

STT_MODEL = "whisper-large-v3"
# A short prompt biases Whisper toward clinic vocabulary and the doctors' correct spellings in both scripts.
PROMPT = ("Shifa Family Clinic, Dr. Ayesha Malik, Dr. Usman Tariq, Dr. Sana Iqbal, appointment, fees. "
          "شفا فیملی کلینک، ڈاکٹر عائشہ ملک، ڈاکٹر عثمان طارق، ڈاکٹر ثنا اقبال، اپائنٹمنٹ")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0
# Pure filler sounds; not "جی"/"ہاں", which can be real answers to "should I book it?"
FILLERS = {"ہم", "ہمم", "اوہ", "آہ", "ام", "hmm", "hm", "um", "uh", "oh", "ah"}


@lru_cache
def _client():
    from groq import Groq

    return Groq(api_key=get_settings().groq_api_key)


def _field(seg, name):
    return seg.get(name) if isinstance(seg, dict) else getattr(seg, name, None)


# Phrases Whisper produces from silence or noise. "کل، ڈاکٹر صاحب" came from our OWN old hint text, which
# Whisper echoed back three times in one call. These need much higher confidence to count as speech.
_KNOWN_PHANTOMS = {"ڈاکٹرصاحب", "کلڈاکٹرصاحب", "شکریہ", "thankyou", "thanksforwatching", "you", "bye"}


def _norm(text: str) -> str:
    return re.sub(r"[^\w]", "", text).lower()


def _speech_text(res) -> str:
    segments = getattr(res, "segments", None) or []
    if not segments:
        return res.text.strip()
    kept = [s for s in segments
            if not ((_field(s, "no_speech_prob") or 0) > NO_SPEECH_PROB
                    or (_field(s, "avg_logprob") or 0) < MIN_AVG_LOGPROB)]
    text = " ".join((_field(s, "text") or "").strip() for s in kept).strip()
    if _norm(text) in _KNOWN_PHANTOMS and any((_field(s, "no_speech_prob") or 0) > 0.2
                                              or (_field(s, "avg_logprob") or 0) < -0.4 for s in kept):
        return ""
    return text


def is_filler(text: str) -> bool:
    return re.sub(r"[^\w\s]", "", text).strip().lower() in FILLERS


# A Pakistani caller speaks Urdu, English or Punjabi (Whisper may label Urdu as Hindi, Arabic or Persian).
# Anything else is almost certainly a TV or video in the background: "¡Ay, Dios, qué bonita era!" got through.
ALLOWED_LANGUAGES = {"urdu", "ur", "english", "en", "hindi", "hi", "punjabi", "pa", "arabic", "ar", "persian", "fa"}


def transcribe(wav: bytes, language: str | None = None) -> tuple[str, str]:
    """Returns (text, language). Empty text means noise, a filler sound or background media: don't answer it."""
    kwargs = {"language": language} if language else {}
    res = _client().audio.transcriptions.create(file=("turn.wav", wav), model=STT_MODEL, prompt=PROMPT,
                                                response_format="verbose_json", temperature=0.0, **kwargs)
    text, lang = _speech_text(res), language or (getattr(res, "language", "") or "").lower()
    if not language and lang and lang not in ALLOWED_LANGUAGES:
        return "", lang
    if not language and (_DEVANAGARI.search(text) or lang in {"hindi", "hi"}):
        return transcribe(wav, language="ur")
    if is_filler(text):
        text = ""
    return text, "ur" if lang in {"urdu", "ur"} else lang
