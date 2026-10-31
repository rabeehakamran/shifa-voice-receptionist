"""The receptionist as a LangGraph multi-agent graph (one graph invocation = one caller turn).

    START -> safety_rules --emergency--> escalate -> END
                  |
                router (+ speculative retrieval in parallel) --risk=emergency--> escalate
                  |
         faq | booking | handoff -> END

Each node records its own latency into state["timings"] so every turn has a latency breakdown.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from sqlmodel import Session

from shifa import llm
from shifa.agents import guardrails, prompts
from shifa.agents.safety import ESCALATION_MESSAGES, detect_script, emergency_match
from shifa.db import InboxItem
from shifa.rag import retriever
from shifa.tools.booking_tools import BookingError, leave_message, make_tools

MAX_TOOL_STEPS = 6
# Below this, the raw-text retrieval probably missed; retry with the router's English rewrite.
REWRITE_RETRY_SCORE = 0.70
_pool = ThreadPoolExecutor(max_workers=4)


class CallState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    phone: str
    call_id: str
    channel: str
    caller_context: str
    lang: str
    route: str
    risk: str
    search_query: str
    hits: list[dict]
    citations: list[str]
    escalated: bool
    timings: dict


class RouterDecision(BaseModel):
    route: Literal["booking", "faq", "handoff"]
    risk: Literal["none", "urgent", "medical_advice", "injection", "emergency"] = "none"
    search_query: str = Field("", description="English KB search query; empty for booking")


def timed(name: str):
    def deco(fn):
        @wraps(fn)
        def inner(state: CallState, config: RunnableConfig):
            t0 = time.perf_counter()
            out = fn(state, config)
            out["timings"] = {**state.get("timings", {}), **out.get("timings", {}),
                              name: round((time.perf_counter() - t0) * 1000)}
            return out
        return inner
    return deco


def _engine(config: RunnableConfig):
    return config["configurable"]["engine"]


def _last_user_text(state: CallState) -> str:
    return next((m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")


def _dialogue(state: CallState, n: int = 8) -> list[AnyMessage]:
    """Recent human/assistant turns only (no tool chatter): cheaper and clearer for router and FAQ."""
    msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)
            or (isinstance(m, AIMessage) and m.text and not m.tool_calls)]
    return msgs[-n:]


def _system(state: CallState, *parts: str) -> str:
    """Persona + language + agent instructions, plus post-escalation care if this call had an emergency."""
    extra = [prompts.POST_ESCALATION] if state.get("escalated") else []
    return "\n\n".join([prompts.PERSONA, prompts.LANG[state.get("lang", "en")], *parts, *extra,
                        state.get("caller_context", "")])


@timed("safety")
def safety_rules(state: CallState, config: RunnableConfig) -> dict:
    text = _last_user_text(state)
    lang = detect_script(text, previous=state.get("lang"))
    if state.get("channel") == "voice" and lang == "roman_ur":
        lang = "ur"  # the Urdu TTS voice reads Urdu script, not Roman Urdu
    out = {"lang": lang, "risk": "none", "citations": [], "hits": [], "timings": {}}
    if emergency_match(text):
        out["risk"] = "emergency"
    return out


def _timed_search(query: str) -> tuple[list[dict], int]:
    t0 = time.perf_counter()
    hits = [{"id": h.id, "text": h.text, "score": round(h.score, 3)} for h in retriever.search(query)]
    return hits, round((time.perf_counter() - t0) * 1000)


@timed("router")
def router(state: CallState, config: RunnableConfig) -> dict:
    # Speculative retrieval on the raw text runs while the router LLM thinks: hides ~700 ms of embedding latency.
    spec = _pool.submit(_timed_search, _last_user_text(state))
    convo = "\n".join(f"{'Caller' if isinstance(m, HumanMessage) else 'Receptionist'}: {m.text}"
                      for m in _dialogue(state, 6))
    t0 = time.perf_counter()
    decision: RouterDecision = llm.structured(RouterDecision, tier="fast").invoke(
        [SystemMessage(prompts.ROUTER), HumanMessage(convo)])
    llm_ms = round((time.perf_counter() - t0) * 1000)
    hits, retrieval_ms = spec.result()
    timings = {"router_llm": llm_ms, "retrieval": retrieval_ms}
    if decision.route == "faq" and decision.search_query and (not hits or hits[0]["score"] < REWRITE_RETRY_SCORE):
        hits, timings["retrieval_retry"] = _timed_search(decision.search_query)
    return {"route": decision.route, "risk": decision.risk, "search_query": decision.search_query,
            "hits": hits, "timings": timings}


@timed("escalate")
def escalate(state: CallState, config: RunnableConfig) -> dict:
    # Deterministic, no LLM: the most important message must be instant and can't be hallucinated.
    with Session(_engine(config)) as s:
        leave_message(s, state["phone"], state.get("call_id"),
                      f"EMERGENCY reported by caller: {_last_user_text(state)}", kind="emergency")
    msg = ESCALATION_MESSAGES.get(state.get("lang", "en"), ESCALATION_MESSAGES["en"])
    return {"messages": [AIMessage(msg)], "route": "escalate", "escalated": True}


_CITE = re.compile(r"\[([^\]]+)\]")


def split_citations(text: str) -> tuple[str, list[str]]:
    """'Fee is Rs 2000. [02_doctors#dr-ayesha]' -> ('Fee is Rs 2000.', ['02_doctors#dr-ayesha'])."""
    cites = [c.strip() for group in _CITE.findall(text) for c in group.split(",")]
    return _CITE.sub("", text).strip(), cites


def _fallback(state: CallState) -> str:
    return prompts.FALLBACK_REPLY.get(state.get("lang", "en"), prompts.FALLBACK_REPLY["en"])


@timed("faq")
def faq(state: CallState, config: RunnableConfig) -> dict:
    hits = state.get("hits") or []
    context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits) or "(no matching information)"
    system = _system(state, prompts.FAQ.format(context=context))
    if state.get("risk") == "medical_advice":
        system += "\nThe caller is asking for medical advice: politely decline and offer an appointment."
    model = llm.chat("smart")
    msgs = [SystemMessage(system), *_dialogue(state)]
    answer, cites = split_citations(guardrails.clean(model.invoke(msgs).text))
    issues = guardrails.problems(answer, state.get("lang", "en"))
    if issues:  # empty, leaked reasoning, too long, wrong script: regenerate once with the reason
        msgs.append(HumanMessage(guardrails.correction_note(issues)))
        answer, cites = split_citations(guardrails.clean(model.invoke(msgs).text))
        if not guardrails.acceptable(guardrails.problems(answer, state.get("lang", "en"))):
            answer, cites = _fallback(state), []
    # Only genuine KB gaps go to the owner's "teach Hina" inbox, not medical-advice or post-emergency turns.
    if "NO_ANSWER" in cites and state.get("risk") == "none" and not state.get("escalated"):
        with Session(_engine(config)) as s:
            s.add(InboxItem(call_id=state.get("call_id"), phone=state["phone"], kind="unanswered",
                            text=_last_user_text(state)))
            s.commit()
    valid = {h["id"] for h in hits}
    return {"messages": [AIMessage(answer)], "citations": [c for c in cites if c in valid] or
            (["NO_ANSWER"] if "NO_ANSWER" in cites else [])}


def _tool_loop(system: str, state: CallState, tools: list) -> list[AnyMessage]:
    """Minimal ReAct loop. Tool errors go back to the model as messages so it can self-correct."""
    by_name = {t.name: t for t in tools}
    model = llm.with_tools(tools)
    history = state["messages"][-20:]
    # Never start mid tool-exchange: an orphaned ToolMessage is rejected by providers.
    while history and not isinstance(history[0], HumanMessage):
        history = history[1:]
    new: list[AnyMessage] = []
    corrected = False
    for _ in range(MAX_TOOL_STEPS):
        ai: AIMessage = model.invoke([SystemMessage(system), *history, *new])
        if not ai.tool_calls:
            text = guardrails.clean(ai.text)
            issues = guardrails.problems(text, state.get("lang", "en"))
            if issues and not corrected:
                # Empty reply or leaked reasoning (both seen in testing): ask once more, with the reason.
                corrected = True
                new.append(HumanMessage(guardrails.correction_note(issues)))
                continue
            if issues and not guardrails.acceptable(issues):
                text = _fallback(state)
            new = [m for m in new if not (isinstance(m, HumanMessage) and m.text.startswith("(system:"))]
            new.append(AIMessage(text))  # plain string: portable across providers
            return new
        new.append(ai)
        for call in ai.tool_calls:
            try:
                result = by_name[call["name"]].invoke(call["args"])
            except KeyError:
                result = f"Error: unknown tool {call['name']}"
            except (BookingError, ValueError) as e:
                result = f"Error: {e}"
            new.append(ToolMessage(str(result), tool_call_id=call["id"], name=call["name"]))
    new = [m for m in new if not (isinstance(m, HumanMessage) and m.text.startswith("(system:"))]
    new.append(AIMessage("Maazrat, system mein masla aa raha hai. Main staff ko message de deti hoon."
                         if state.get("lang") != "en" else
                         "Sorry, I'm having trouble with the booking system. I'll ask the staff to call you back."))
    return new


def _tools(state: CallState, config: RunnableConfig):
    return make_tools(_engine(config), state["phone"], state.get("call_id"))


@timed("booking")
def booking(state: CallState, config: RunnableConfig) -> dict:
    urgent = prompts.URGENT if state.get("risk") == "urgent" or state.get("escalated") else ""
    system = _system(state, prompts.BOOKING.format(calendar=prompts.calendar_hint(), urgent=urgent,
                                                   caller_context=""))
    return {"messages": _tool_loop(system, state, _tools(state, config)["booking"])}


@timed("handoff")
def handoff(state: CallState, config: RunnableConfig) -> dict:
    return {"messages": _tool_loop(_system(state, prompts.HANDOFF), state, _tools(state, config)["handoff"])}


def after_safety(state: CallState) -> str:
    return "escalate" if state.get("risk") == "emergency" else "router"


def after_router(state: CallState) -> str:
    if state.get("risk") == "emergency":
        return "escalate"
    # After an emergency, worried follow-ups ("BP low tha", "ab theek hain") go to booking for an urgent check-up.
    if state.get("escalated") and state.get("risk") in ("medical_advice", "urgent"):
        return "booking"
    return state["route"]


def build_graph(checkpointer=None):
    g = StateGraph(CallState)
    for name, fn in [("safety_rules", safety_rules), ("router", router), ("escalate", escalate),
                     ("faq", faq), ("booking", booking), ("handoff", handoff)]:
        g.add_node(name, fn)
    g.add_edge(START, "safety_rules")
    g.add_conditional_edges("safety_rules", after_safety, ["escalate", "router"])
    g.add_conditional_edges("router", after_router, ["escalate", "faq", "booking", "handoff"])
    for leaf in ("escalate", "faq", "booking", "handoff"):
        g.add_edge(leaf, END)
    return g.compile(checkpointer=checkpointer)
