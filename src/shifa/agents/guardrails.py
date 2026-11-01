"""Output guardrail: every reply goes to a caller's ear via TTS, so never trust raw model text.

Found in the first Arena run: gpt-oss leaked its scratchpad into a reply ("We need to respond properly:
confirm appointment details in 2 short sentences...") after a wall of whitespace, plus fragments like "?>"
and "(Waiting for name)". `clean` removes harmless artifacts; `problems` flags replies that must be regenerated.
"""

import difflib
import re
from collections import Counter

from shifa.agents.safety import detect_script

TRIGGERS: Counter = Counter()  # issue -> count; reported by the eval runner
MAX_CHARS = 400  # two short spoken sentences; anything far longer is a leak or a lecture

_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
_ARTIFACTS = re.compile(r"\?>|<\|[^|]*\|>|\((?:waiting|pause|note|thinking)[^)]*\)", re.I)
_META = re.compile(
    r"\b(we need to|the (?:response|answer|reply) (?:must|should)|i should (?:respond|reply|answer)|"
    r"let'?s (?:respond|reply|answer)|as an ai|system prompt|my instructions|the user (?:says|wants|asked))\b", re.I)
_ARABIC = re.compile(r"[؀-ۿ]")
# Prompting alone didn't stop "Bhai sahab": the caller's gender is unknown, so flag gendered honorifics.
# Deliberately NOT verb forms: "karengi"/"سکتی ہیں" is also correct about a female doctor ("Dr. Ayesha dekh sakti
# hain"), and mirroring a caller who said "main aa sakti hoon" is polite Urdu. Precision over recall here.
_GENDERED_ADDRESS = re.compile(
    r"\b(bhai ?sa+h?a*b|bhai jaan|baji|behen ?ji|bahan ?ji|sir ?ji|madam|uncle ?ji|aunty ?ji)\b"
    r"|بھائی صاحب|باجی|بہن جی", re.I)


MAX_SENTENCES = 2  # paraphrased repeats score only 0.33-0.49 similarity (measured), so a hard cap is the real fix
# Urdu full stop / question mark always end a sentence (replies arrive glued with no space). A Latin ". ! ?"
# ends one only before whitespace and not after an abbreviation: "Dr. Ayesha ki fee Rs. 2000 hai." is ONE sentence.
_SENTENCE = re.compile(r"(?<=[۔؟])\s*|(?<=[.!?])(?<!\bDr\.)(?<!\bRs\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bSt\.)\s+", re.I)
_IDS = re.compile(r"\b(?:slot_id|appointment_id|slot id|appointment id|id\s*[:=#]?\s*\d+|slot\s*[:=#]?\s*\d+)\b"
                  r"|آئی ?ڈی|سلاٹ ?\d+", re.I)  # "(slot 148)" leaked from a Gemini fallback reply


def _norm(s: str) -> str:
    return re.sub(r"[^\w]", "", s).lower()


def clean(text: str) -> str:
    """Strip artifacts, drop repeated sentences, and cap length IN CODE.

    "Max 2 sentences" in the prompt wasn't enough: voice replies grew to 4-5 sentences (and TTS time with
    them), and some came back glued twice ("...بتائیں۔براہِ کرم اپنا نام بتائیں...").
    """
    text = re.sub(r"\s+", " ", _ARTIFACTS.sub("", _ZERO_WIDTH.sub("", text))).strip()
    kept: list[str] = []
    for sentence in (s.strip() for s in _SENTENCE.split(text) if s.strip()):
        n = _norm(sentence)
        if any(n == _norm(k) or (len(n) > 12 and difflib.SequenceMatcher(None, n, _norm(k)).ratio() > 0.75)
               for k in kept):
            continue  # near-duplicate of a sentence we already have
        kept.append(sentence)
    return " ".join(kept[:MAX_SENTENCES])


def problems(text: str, lang: str) -> list[str]:
    issues = []
    if not text:
        issues.append("empty reply")
    if len(text) > MAX_CHARS:
        issues.append(f"too long ({len(text)} chars); max 2 short sentences")
    if _META.search(text) or re.search(r"[.…]{6,}", text):
        issues.append("contains internal reasoning or meta-commentary")
    if lang == "ur" and text and not _ARABIC.search(text):
        issues.append("must be written in Urdu script")
    if lang == "en" and text and (_ARABIC.search(text) or detect_script(text) == "roman_ur"):
        # Seen in a voice call: an English caller got "Kal subah 9 baje... pasand karengi?", read by the Urdu voice.
        issues.append("must be written in English")
    if _IDS.search(text):
        issues.append("mentioned an internal id; describe by doctor, day and time instead")
    if _GENDERED_ADDRESS.search(text):
        issues.append("assumed the caller's gender; use 'aap' and plural forms (karenge, batayein)")
    for issue in issues:
        TRIGGERS[issue.split(" (")[0]] += 1
    return issues


def acceptable(issues: list[str]) -> bool:
    """After one retry, a wrong script or gendered word is tolerable (still a useful reply); a leak or empty reply
    is not."""
    return all(i.startswith(("must be written in", "assumed the caller's gender")) for i in issues)


def correction_note(issues: list[str]) -> str:
    return ("(system: your previous reply was rejected: " + "; ".join(issues) +
            ". Write only the words Hina says to the caller, nothing else.)")
