"""Text-to-speech with Edge-TTS (free neural voices, including Pakistani Urdu). Output: MP3 bytes.

Replies are split into sentences so the first one can start playing while later ones synthesize.
"""

import re

import edge_tts

from shifa.agents.safety import detect_script

VOICES = {"ur": "ur-PK-UzmaNeural", "roman_ur": "ur-PK-UzmaNeural", "en": "en-IN-NeerjaNeural"}
FILLERS = {"ur": "جی، ایک سیکنڈ۔", "en": "Sure, one second."}

_SENTENCE_END = re.compile(r"(?<=[.!?۔؟])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


async def synthesize(text: str, lang: str | None = None) -> bytes:
    voice = VOICES[lang or detect_script(text)]
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate="+5%").stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return bytes(audio)
