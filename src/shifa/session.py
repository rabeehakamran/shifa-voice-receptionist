"""One phone call. Shared entry point for the CLI, the voice server and the eval simulator."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from shifa import llm
from shifa.agents import prompts
from shifa.agents.graph import build_graph
from shifa.db import Appointment, Call, Doctor, Slot, Turn, get_engine

_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(checkpointer=MemorySaver())
    return _GRAPH


@dataclass
class TurnResult:
    reply: str
    route: str
    risk: str
    escalated: bool
    citations: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    lang: str = "en"  # the language this turn was answered in; picks the TTS voice


class CallSummary(BaseModel):
    summary: str = Field(description="1-2 sentences in English")
    sentiment: str = Field(description="positive | neutral | negative | distressed")
    # Only used when no tool result proves the outcome (see factual_outcome).
    outcome: Literal["answered", "unresolved"] = Field(description="answered = caller got what they wanted")
    tags: list[str] = Field(description="2-4 short tags, e.g. fees, booking, lab, complaint")


def caller_context(engine, phone: str) -> str:
    """Returning-caller memory: past patients on this number, so Hina can say 'pichli dafa...'."""
    with Session(engine) as s:
        rows = s.exec(select(Appointment, Slot, Doctor).join(Slot, Appointment.slot_id == Slot.id).join(Doctor)
                      .where(Appointment.phone == phone).order_by(Slot.start.desc()).limit(3)).all()
    if not rows:
        return ""
    seen = "; ".join(f"{a.patient_name} with {d.name} on {sl.start:%d %b} ({a.status})" for a, sl, d in rows)
    return f"Returning caller. Past appointments on this number: {seen}. Greet them warmly as a known patient."


def factual_outcome(messages: list, escalated: bool) -> str | None:
    """Outcomes we can prove from tool results beat the LLM's opinion: a summary once said 'booked' for a call
    where nothing was booked. Returns None when only the LLM can judge (answered vs unresolved)."""
    if escalated:
        return "escalated"
    # The LAST state change wins: a call that booked and then cancelled was summarized as "booked".
    actions = {"BOOKED": "booked", "RESCHEDULED": "rescheduled", "CANCELLED": "cancelled"}
    for m in reversed([m for m in messages if isinstance(m, ToolMessage)]):
        for prefix, outcome in actions.items():
            if m.text.startswith(prefix):
                return outcome
    if any(m.name == "leave_message_for_staff" for m in messages if isinstance(m, ToolMessage)):
        return "message_taken"
    return None


class CallSession:
    def __init__(self, phone: str, channel: str = "text", engine=None, call_id: str | None = None):
        self.engine = engine or get_engine()
        self.phone = phone
        self.channel = channel
        self.call_id = call_id or uuid.uuid4().hex[:12]
        self.config = {"configurable": {"thread_id": self.call_id, "engine": self.engine}}
        self.transcript: list[tuple[str, str]] = []
        self._first = True
        self._lock = threading.Lock()
        with Session(self.engine) as s:
            s.add(Call(id=self.call_id, phone=phone, channel=channel))
            s.commit()

    def turn(self, text: str) -> TurnResult:
        # A barge-in can cancel the voice task while a turn runs in a worker thread; the lock keeps
        # the next turn from racing the unfinished one on the same conversation state.
        with self._lock:
            return self._turn(text)

    def _turn(self, text: str) -> TurnResult:
        t0 = time.perf_counter()
        update = {"messages": [HumanMessage(text)], "timings": {}}
        if self._first:
            update |= {"phone": self.phone, "call_id": self.call_id, "channel": self.channel,
                       "caller_context": caller_context(self.engine, self.phone), "escalated": False}
            self._first = False
        state = _graph().invoke(update, self.config)
        timings = {**state.get("timings", {}), "total": round((time.perf_counter() - t0) * 1000)}
        result = TurnResult(reply=state["messages"][-1].text, route=state.get("route", ""),
                            risk=state.get("risk", "none"), escalated=state.get("escalated", False),
                            citations=state.get("citations", []), timings_ms=timings,
                            lang=state.get("lang", "en"))
        self._log(text, result)
        return result

    def _log(self, text: str, r: TurnResult) -> None:
        self.transcript += [("caller", text), ("agent", r.reply)]
        with Session(self.engine) as s:
            s.add(Turn(call_id=self.call_id, role="caller", text=text))
            turn = Turn(call_id=self.call_id, role="agent", text=r.reply, route=r.route, timings_ms=r.timings_ms)
            s.add(turn)
            s.commit()
            self._last_turn_id = turn.id

    def add_timings(self, extra: dict) -> None:
        """Voice adds stages measured outside the graph (STT, TTS first byte) to the last turn."""
        with Session(self.engine) as s:
            turn = s.get(Turn, self._last_turn_id)
            turn.timings_ms = {**turn.timings_ms, **extra}
            s.add(turn)
            s.commit()

    def end(self, summarize: bool = True) -> Call:
        with Session(self.engine) as s:
            call = s.get(Call, self.call_id)
            call.ended_at = datetime.now()
            state = _graph().get_state(self.config).values
            call.escalated = bool(state.get("escalated"))
            if summarize and self.transcript:
                convo = "\n".join(f"{who}: {text}" for who, text in self.transcript)
                try:
                    out: CallSummary = llm.structured(CallSummary, tier="fast").invoke(
                        [("system", prompts.POST_CALL), ("human", convo)])
                    call.summary, call.sentiment, call.outcome, call.tags = (
                        out.summary, out.sentiment, out.outcome, out.tags)
                except Exception as e:  # a failed summary must never lose the call record
                    call.summary = f"(summary failed: {type(e).__name__})"
            call.outcome = factual_outcome(state.get("messages", []), call.escalated) or call.outcome or "unresolved"
            s.add(call)
            s.commit()
            s.refresh(call)
            return call
