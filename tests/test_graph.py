"""Graph wiring tests with fake LLMs: no API keys, no quota."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from sqlmodel import Session, select

from shifa import llm
from shifa.agents import graph as g
from shifa.db import Call, InboxItem
from shifa.rag.retriever import Chunk
from shifa.session import CallSession


def _explode(*a, **k):
    raise AssertionError("LLM must not be called on this path")


def test_emergency_path_uses_no_llm(engine, monkeypatch):
    for name in ("chat", "structured", "with_tools"):
        monkeypatch.setattr(llm, name, _explode)
    call = CallSession("03001112222", engine=engine)
    r = call.turn("Abbu ko seenay mein dard ho raha hai aur saans nahi aa rahi")
    assert r.escalated and r.route == "escalate"
    assert "1122" in r.reply
    with Session(engine) as s:
        assert s.exec(select(InboxItem).where(InboxItem.kind == "emergency")).first()
    assert call.end(summarize=False).outcome == "escalated"


def test_router_risk_emergency_also_escalates(engine, monkeypatch):
    """A phrasing the lexicon misses is caught by the router's risk field (layer 2)."""
    monkeypatch.setattr(llm, "structured", lambda *a, **k: RunnableLambda(
        lambda _: g.RouterDecision(route="faq", risk="emergency")))
    r = CallSession("03001112222", engine=engine).turn("my wife's lips are turning blue")
    assert r.escalated and "1122" in r.reply


def test_faq_citations_and_unanswered_goes_to_inbox(engine, monkeypatch):
    monkeypatch.setattr(llm, "structured", lambda *a, **k: RunnableLambda(
        lambda _: g.RouterDecision(route="faq", search_query="insurance panel")))
    monkeypatch.setattr(g.retriever, "search", lambda q: [Chunk("03_fees_payment#payment-methods", "cash...")])
    monkeypatch.setattr(llm, "chat", lambda *a, **k: RunnableLambda(
        lambda _: AIMessage("Main ye sawal staff ko de deti hoon, woh call back karenge. [NO_ANSWER]")))
    r = CallSession("03001112222", engine=engine).turn("kya aap State Life insurance lete hain?")
    assert r.route == "faq" and r.citations == ["NO_ANSWER"]
    assert "[" not in r.reply
    with Session(engine) as s:
        item = s.exec(select(InboxItem).where(InboxItem.kind == "unanswered")).one()
        assert "insurance" in item.text


def test_split_citations():
    text, cites = g.split_citations("Fee Rs. 2000 hai. [02_doctors_timings#dr-ayesha-malik, 03_fees#x]")
    assert text == "Fee Rs. 2000 hai." and cites == ["02_doctors_timings#dr-ayesha-malik", "03_fees#x"]


def test_turn_records_latency_breakdown(engine, monkeypatch):
    monkeypatch.setattr(llm, "structured", _explode)
    r = CallSession("03001112222", engine=engine).turn("bachay ko jhatke lag rahe hain")
    assert {"safety", "escalate", "total"} <= set(r.timings_ms)
    with Session(engine) as s:
        assert s.exec(select(Call)).first() is not None


def test_outcome_comes_from_tool_results_not_llm_opinion():
    from langchain_core.messages import ToolMessage

    from shifa.session import factual_outcome

    booked = [ToolMessage("BOOKED appointment_id=3 for Ali", tool_call_id="1", name="book_appointment")]
    looked = [ToolMessage("slot_id=1 | ...", tool_call_id="1", name="get_available_slots")]
    msg = [ToolMessage("Message saved", tool_call_id="1", name="leave_message_for_staff")]
    assert factual_outcome(booked, escalated=False) == "booked"
    assert factual_outcome(booked, escalated=True) == "escalated"
    assert factual_outcome(msg, escalated=False) == "message_taken"
    assert factual_outcome(looked, escalated=False) is None  # only browsing slots is NOT a booking


def test_outcome_uses_last_action():
    from langchain_core.messages import ToolMessage

    from shifa.session import factual_outcome

    msgs = [ToolMessage("BOOKED appointment_id=3", tool_call_id="1", name="book_appointment"),
            ToolMessage("CANCELLED appointment_id=3", tool_call_id="2", name="cancel_appointment")]
    assert factual_outcome(msgs, escalated=False) == "cancelled"


def test_caller_name_is_remembered_across_turns(engine, monkeypatch):
    """Regression: 'Bro, do you have memory of a goldfish? I told you my name just a minute before.'"""
    decisions = iter([g.RouterDecision(route="faq", caller_name="Diya Aslam"), g.RouterDecision(route="faq")])
    monkeypatch.setattr(llm, "structured", lambda *a, **k: RunnableLambda(lambda _: next(decisions)))
    seen = []
    monkeypatch.setattr(llm, "chat", lambda *a, **k: RunnableLambda(
        lambda msgs: seen.append(msgs[0].content) or AIMessage("Ji, bataiye.")))
    call = CallSession("03001112222", engine=engine)
    call.turn("My name is Diya Aslam")
    call.turn("what are your timings?")  # this turn doesn't mention the name
    assert "The caller's name is Diya Aslam" in seen[-1]
