"""Text-to-speech with Edge-TTS (free neural voices, including Pakistani Urdu). Output: MP3.

Measured from Lahore (5 samples each): first audio chunk ~1 s for ur-PK-UzmaNeural (980 ms median, faster than
ur-IN-GulNeural at 1162 ms). Multilingual voices were rejected: Whisper understood their Urdu far worse
(0.15-0.79 vs 0.88-0.89 similarity) and they were slower.

`stream` yields audio chunks as they arrive instead of waiting for whole sentences: the wait for a full long
Urdu sentence was 3-5 s, while the first chunk takes ~1 s regardless of length.
"""

import asyncio
import re
from collections.abc import AsyncIterator

import edge_tts

from shifa.agents.guardrails import _SENTENCE

# Voice calls only ever speak Urdu (in Urdu script) or English. Roman Urdu is a way of *writing* Urdu; spoken, it
# is just Urdu, and the Urdu voice reads Latin letters badly, so it must never reach TTS.
VOICES = {"ur": "ur-PK-UzmaNeural", "en": "en-IN-NeerjaNeural"}
FILLERS = {"ur": "جی، ایک سیکنڈ۔", "en": "Sure, one second."}


def voice_lang(lang: str) -> str:
    return "en" if lang == "en" else "ur"


def sentences(text: str) -> list[str]:
    # Same splitter as the guardrail: Urdu "۔" with no following space still splits; "Dr." and "Rs." don't.
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


URDU_PIECE_CHARS = 50
MIN_PIECE_CHARS = 15
_CLAUSE = re.compile(r"(?<=[،,;؛:])\s*")


def _split_words(text: str, limit: int) -> list[str]:
    out, buf = [], ""
    for word in text.split():
        if buf and len(buf) + 1 + len(word) > limit:
            out.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
    return [*out, buf] if buf else out


def pieces(text: str, lang: str) -> list[str]:
    """Units sent to TTS. Measured from Lahore: a 128-char Urdu sentence took 982-4332 ms to first audio (median
    3886), a 51-char clause 933-966 ms. So Urdu is split at clause punctuation, and pieces still over ~50 chars at
    word boundaries (the model often writes long Urdu sentences with no commas). English showed no length effect
    (990 vs 924 ms), so it stays sentence-level for smoother prosody."""
    if voice_lang(lang) == "en":
        return sentences(text)
    out: list[str] = []
    for sentence in sentences(text):
        for clause in (c.strip() for c in _CLAUSE.split(sentence) if c.strip()):
            for piece in _split_words(clause, URDU_PIECE_CHARS) if len(clause) > URDU_PIECE_CHARS else [clause]:
                if out and len(piece) < MIN_PIECE_CHARS:  # a tiny fragment isn't worth a ~1 s request
                    out[-1] = f"{out[-1]} {piece}"
                else:
                    out.append(piece)
    return out


_PARALLEL = asyncio.Semaphore(4)  # a few concurrent Edge requests measured fine; dozens would be rude to a free service


async def _produce(text: str, voice: str, queue: asyncio.Queue) -> None:
    try:
        async with _PARALLEL:
            async for chunk in edge_tts.Communicate(text, voice, rate="+5%").stream():
                if chunk["type"] == "audio":
                    await queue.put(chunk["data"])
    finally:
        await queue.put(None)


async def stream(text: str, lang: str) -> AsyncIterator[bytes]:
    """All pieces start synthesizing in parallel (each Edge request has ~1 s setup); chunks are yielded in
    order as soon as they arrive. Cancelling the consumer (barge-in) cancels the producers.

    Tried and reverted: request hedging (sending the first piece twice and keeping the faster). Measured over
    8 runs it didn't help on this link (median 958 -> 972 ms, worst case 1581 -> 2069 ms). Running pieces in
    parallel vs one at a time also made no difference (~950 vs ~935 ms), so contention isn't the issue."""
    voice = VOICES[voice_lang(lang)]
    parts = pieces(text, lang)
    queues = [asyncio.Queue() for _ in parts]
    tasks = [asyncio.create_task(_produce(p, voice, q)) for p, q in zip(parts, queues)]
    try:
        for q in queues:
            while (chunk := await q.get()) is not None:
                yield chunk
    finally:
        for t in tasks:
            t.cancel()


async def synthesize(text: str, lang: str) -> bytes:
    return b"".join([c async for c in stream(text, lang)])
