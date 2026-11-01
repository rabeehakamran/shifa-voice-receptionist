"""Layer 1 of safety: a deterministic emergency lexicon (English, Roman Urdu, Urdu script).

Runs before any LLM, in microseconds, and can't be talked out of a match. Layer 2 is the router's
`risk` field, which catches phrasings the lexicon misses. Recall matters more than precision
here: a false alarm costs one extra sentence, a miss can cost a life.
"""

import re

_PATTERNS = [
    # English
    r"chest pain", r"heart attack", r"can'?t breathe", r"cannot breathe", r"not breathing",
    r"difficulty breathing", r"unconscious", r"fainted", r"passed out", r"seizure", r"\bhaving (a )?fits?\b",
    r"heavy bleeding", r"bleeding (a lot|heavily|won'?t stop)", r"stroke", r"overdose", r"poison",
    r"suicid", r"kill myself", r"swallowed",
    # Roman Urdu (spelling varies a lot, so patterns are loose)
    r"seen[ae]y? (main|mein|me|mai) (dard|dart|takleef)", r"dil ka dor[ae]",
    r"saa?ns? (nahi|nhi|nahin|ruk|band|lene (main|mein|me) (mushkil|takleef))", r"dam ghut",
    r"behosh", r"be hosh", r"hosh (nahi|nhi|nahin)", r"jhatk[ea]", r"mirgi", r"fit (par|pad|pr)",
    r"khoo?n (beh|ruk nahi|nahi ruk|bohat|bahut)", r"zeh[ae]r", r"khud ?kushi", r"lakw[ae]",
    r"munh tera", r"gir (gaye|gayi|gaya) .*(hosh|bol nahi)",
    # Urdu script
    r"سینے (میں|مے) درد", r"دل کا دورہ", r"سانس (نہیں|رک|بند|لینے میں)", r"بے ?ہوش", r"جھٹکے",
    r"مرگی", r"خون (بہ|نہیں رک|رک نہیں)", r"زہر", r"خودکشی", r"فالج",
]
_EMERGENCY = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def emergency_match(text: str) -> str | None:
    m = _EMERGENCY.search(text)
    return m.group(0) if m else None


ESCALATION_MESSAGES = {
    "en": ("This sounds like an emergency. Please call Rescue 1122 right now or go to the nearest hospital "
           "emergency. I am alerting the doctor immediately."),
    "roman_ur": ("Ye emergency lag rahi hai. Foran 1122 par call karein ya qareebi hospital ki emergency "
                 "jayein. Main abhi doctor sahab ko alert kar rahi hoon."),
    "ur": ("یہ ایمرجنسی لگ رہی ہے۔ فوراً 1122 پر کال کریں یا قریبی ہسپتال کی ایمرجنسی جائیں۔ "
           "میں ابھی ڈاکٹر صاحب کو اطلاع دے رہی ہوں۔"),
}


_ROMAN_UR = re.compile(
    r"\b(hai|hain|ha|hy|hun|hoon|kya|kia|kyun|kab|kahan|kaun|kaise|kitni|kitna|kitne|mujhe|mujhay|mera|meri|mere|"
    r"hamara|aap|ap|apka|aapka|unka|unki|unke|un|wo|woh|ye|yeh|mein|main|mai|nahi|nhi|nahin|kal|aaj|parson|abhi|"
    r"karna|karo|kardo|kar|kardein|karwa|dein|do|chahiye|chahye|ji|jee|bata|batao|batayen|sahab|baji|bhai|"
    r"wala|wali|hoga|hogi|tha|thi|thay|ho|gaye|gaya|gayi|raha|rahi|rahe|sirf|paas|pass|bhi|aur|lekin|acha|"
    r"theek|shukriya|ka|ki|ke|ko|se|par|pe|tou|to|bohat|bahut|abbu|ammi|beta|beti|bacha|bachay|paise)\b")
_EN = re.compile(
    r"\b(the|is|are|am|was|you|your|do|does|did|what|when|where|which|how|my|i|me|can|could|please|have|has|"
    r"need|want|would|will|appointment|doctor|thanks|thank|hello|hi|yes|no|okay|ok|it|this|that|for|with|of|"
    r"and|to|a|an|in|on|at)\b")


_ASK_ENGLISH = re.compile(r"\b(in english|speak english|english (mein|me|main|men)|angrezi)\b|انگلش|انگلیش|انگلس|انگریزی",
                          re.I)
_ASK_URDU = re.compile(r"\b(in urdu|speak urdu|urdu (mein|me|main|men))\b|اردو میں", re.I)


def detect_script(text: str, previous: str | None = None) -> str:
    """Cheap language guess: 'ur' (Urdu script), 'roman_ur', or 'en'.

    Callers code-switch constantly ("mere pass to sirf 500 rupees hain"), so we score Roman Urdu
    vs English function words. With hysteresis: once a call has a language, switching needs a margin
    of 2+, so one English-ish line ("hina baji are you drunk") doesn't flip Hina mid-call.
    """
    # An explicit request beats every heuristic. Real failure: a caller said (transcribed in Urdu script)
    # "I'm talking to you in English, why aren't you?" and kept getting Urdu.
    if _ASK_ENGLISH.search(text):
        return "en"
    if _ASK_URDU.search(text):
        return "ur" if re.search(r"[؀-ۿ]", text) else "roman_ur"
    if re.search(r"[؀-ۿ]", text):
        return "ur"
    low = text.lower()
    ur, en = len(_ROMAN_UR.findall(low)), len(_EN.findall(low))
    margin = 2 if previous in ("roman_ur", "en", "ur") else 1
    if ur - en >= margin:
        return "roman_ur"
    if en - ur >= margin:
        return "en"
    return previous or ("roman_ur" if ur > en else "en")
