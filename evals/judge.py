"""LLM-as-judge for what deterministic checks can't see: groundedness, helpfulness, tone.

The judge gets the full clinic KB (it's small) so it can flag any claim not supported by it.
"""

from pydantic import BaseModel, Field

from shifa import llm
from shifa.config import get_settings

RUBRIC = """You grade an AI clinic receptionist's phone call. Clinic knowledge base (the ONLY source of truth):
<kb>
{kb}
</kb>
Booking facts confirmed by tools (slots, appointment confirmations) also count as grounded.

Score 1-5:
- grounded: every factual claim (fees, timings, doctors, services, policies) is supported by the KB or tools.
  5 = all supported, 1 = invented facts.
- helpful: did the caller get what they needed efficiently, or a correct handoff/escalation?
- tone: warm, concise, natural for a Pakistani clinic receptionist on the phone, in the caller's language.
List every unsupported factual claim verbatim in `hallucinations` (empty if none)."""


class JudgeScore(BaseModel):
    grounded: int = Field(ge=1, le=5)
    helpful: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)
    hallucinations: list[str] = Field(default_factory=list)
    rationale: str = Field(description="One sentence")


def _kb_text() -> str:
    return "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(get_settings().kb_dir.glob("*.md")))


def judge(transcript: list[tuple[str, str]]) -> JudgeScore:
    convo = "\n".join(f"{who.upper()}: {text}" for who, text in transcript)
    s = get_settings()
    model = llm.gemini(s.judge_model) if s.google_api_key else llm.chat("smart", 0)
    msgs = [("system", RUBRIC.format(kb=_kb_text())), ("human", convo)]
    return llm.invoke_with_backoff(model.with_structured_output(JudgeScore), msgs)
