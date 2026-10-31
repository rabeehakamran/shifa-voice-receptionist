"""FastAPI app: browser voice calls over WebSocket + owner dashboard API.

Run: uvicorn shifa.api.main:app --reload   then open http://localhost:8000/call and /dashboard

WebSocket protocol (/ws/call):
  client -> server  text {"type":"start","phone":"0300..."} | {"type":"hangup"}
                    binary: one WAV utterance (16 kHz mono, already cut by the browser VAD)
  server -> client  text {"type":"transcript"|"reply"|"filler"|"ended"|"error", ...}
                    binary: MP3 audio, one message per sentence
Barge-in: the browser stops playback when the caller starts talking and sends the next utterance;
any audio still being synthesized for the old reply is cancelled.
"""

import asyncio
import json
import re
import statistics
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select

from shifa import llm
from shifa.agents.safety import detect_script
from shifa.config import get_settings
from shifa.db import Appointment, Call, Doctor, InboxItem, Slot, Turn, get_engine
from shifa.rag import retriever
from shifa.session import CallSession
from shifa.voice import stt, tts

STATIC = Path(__file__).with_name("static")
FILLER_AFTER_S = 0.8
GREETING = "السلام علیکم، شفا فیملی کلینک۔ میں حنا بات کر رہی ہوں، میں آپ کی کیا مدد کر سکتی ہوں؟"

app = FastAPI(title="Shifa Desk")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_filler_cache: dict[str, bytes] = {}


@app.on_event("startup")
async def _warmup():
    llm.warmup()
    for lang in ("ur", "en"):  # fillers must play instantly, so synthesize them before any call
        try:
            await _filler(lang)
        except Exception:
            pass


@app.get("/")
@app.get("/call")
def call_page():
    return FileResponse(STATIC / "call.html")


@app.get("/dashboard")
def dashboard_page():
    return FileResponse(STATIC / "dashboard.html")


async def _filler(lang: str) -> bytes:
    key = "en" if lang == "en" else "ur"
    if key not in _filler_cache:
        _filler_cache[key] = await tts.synthesize(tts.FILLERS[key], key)
    return _filler_cache[key]


async def _speak(ws: WebSocket, text: str, lang: str) -> int | None:
    """Synthesize all sentences in parallel (Edge-TTS has ~1 s connection setup each), send in order.
    Returns ms to the first audio sent."""
    voice_lang = "en" if lang == "en" else "ur"
    t0 = time.perf_counter()
    jobs = [asyncio.create_task(tts.synthesize(s, voice_lang)) for s in tts.sentences(text)]
    first = None
    try:
        for job in jobs:
            await ws.send_bytes(await job)
            first = first if first is not None else round((time.perf_counter() - t0) * 1000)
    finally:
        for job in jobs:  # on barge-in, stop synthesizing the rest
            job.cancel()
    return first


async def _handle_utterance(ws: WebSocket, call: CallSession, wav: bytes) -> None:
    t0 = time.perf_counter()
    text, lang = await asyncio.to_thread(stt.transcribe, wav, getattr(call, "stt_lang", None))
    if lang == "ur":
        call.stt_lang = "ur"  # sticky for the rest of the call: one Whisper request per turn
    stt_ms = round((time.perf_counter() - t0) * 1000)
    if not text:
        return
    await ws.send_json({"type": "transcript", "text": text})

    # Run the agent; if it's slow (tool calls), play a filler so the line isn't silent.
    turn_task = asyncio.create_task(asyncio.to_thread(call.turn, text))
    done, _ = await asyncio.wait({turn_task}, timeout=FILLER_AFTER_S)
    if not done:
        await ws.send_json({"type": "filler"})
        await ws.send_bytes(await _filler(detect_script(text)))
    result = await turn_task

    await ws.send_json({"type": "reply", "text": result.reply, "route": result.route, "risk": result.risk,
                        "escalated": result.escalated, "timings": {"stt": stt_ms, **result.timings_ms}})
    tts_first = await _speak(ws, result.reply, detect_script(result.reply))
    timings = {"stt": stt_ms, "tts_first_audio": tts_first,
               "voice_to_voice": round((time.perf_counter() - t0) * 1000) if tts_first is None else
               stt_ms + result.timings_ms["total"] + tts_first}
    call.add_timings(timings)
    await ws.send_json({"type": "timings", "timings": timings})


@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    await ws.accept()
    call: CallSession | None = None
    current: asyncio.Task | None = None
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                data = json.loads(msg["text"])
                if data["type"] == "start":
                    call = CallSession(data.get("phone") or "03001234567", channel="voice")
                    await ws.send_json({"type": "reply", "text": GREETING, "route": "greeting"})
                    await _speak(ws, GREETING, "ur")
                elif data["type"] == "hangup":
                    break
            elif msg.get("bytes") and call:
                if current and not current.done():
                    current.cancel()  # barge-in: drop the rest of the previous reply
                current = asyncio.create_task(_handle_utterance(ws, call, msg["bytes"]))
    except WebSocketDisconnect:
        pass
    finally:
        if current and not current.done():
            current.cancel()
        if call:
            summary = await asyncio.to_thread(call.end)
            try:
                await ws.send_json({"type": "ended", "outcome": summary.outcome, "summary": summary.summary})
            except Exception:
                pass


# ---------------- Owner dashboard API ----------------

def _session():
    return Session(get_engine())


@app.get("/api/calls")
def list_calls(limit: int = 50):
    with _session() as s:
        calls = s.exec(select(Call).where(Call.channel != "sim").order_by(Call.started_at.desc()).limit(limit)).all()
        return [c.model_dump() for c in calls]


@app.get("/api/calls/{call_id}")
def call_detail(call_id: str):
    with _session() as s:
        turns = s.exec(select(Turn).where(Turn.call_id == call_id).order_by(Turn.id)).all()
        return {"call": s.get(Call, call_id).model_dump(), "turns": [t.model_dump() for t in turns]}


@app.get("/api/appointments")
def appointments():
    with _session() as s:
        rows = s.exec(select(Appointment, Slot, Doctor).join(Slot, Appointment.slot_id == Slot.id).join(Doctor)
                      .where(Appointment.status == "booked").order_by(Slot.start)).all()
        return [{"id": a.id, "patient": a.patient_name, "phone": a.phone, "reason": a.reason, "doctor": d.name,
                 "start": sl.start.isoformat()} for a, sl, d in rows]


@app.get("/api/inbox")
def inbox():
    with _session() as s:
        return [i.model_dump() for i in s.exec(select(InboxItem).order_by(InboxItem.created_at.desc())).all()]


class Answer(BaseModel):
    answer: str


LEARNED = "90_learned.md"


@app.post("/api/inbox/{item_id}/answer")
def answer_inbox(item_id: int, body: Answer):
    """Self-improving KB: the owner answers once, it becomes a KB section, next caller gets it."""
    with _session() as s:
        item = s.get(InboxItem, item_id)
        item.answer, item.status = body.answer, "resolved"
        s.add(item)
        s.commit()
        if item.kind == "unanswered":
            path = get_settings().kb_dir / LEARNED
            if not path.exists():
                path.write_text("# Answers added by clinic staff\n", encoding="utf-8")
            heading = re.sub(r"\s+", " ", item.text).strip()[:80]
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n## {heading}\n{body.answer.strip()}\n")
            retriever.ingest()
    return {"ok": True, "added_to_kb": item.kind == "unanswered"}


@app.get("/api/stats")
def stats():
    with _session() as s:
        calls = s.exec(select(Call).where(Call.channel != "sim")).all()
        turns = s.exec(select(Turn).where(Turn.role == "agent")).all()
        inbox_items = s.exec(select(InboxItem)).all()
    v2v = [t.timings_ms["voice_to_voice"] for t in turns if t.timings_ms.get("voice_to_voice")]
    agent = [t.timings_ms["total"] for t in turns if t.timings_ms.get("total")]
    outcomes: dict[str, int] = {}
    for c in calls:
        outcomes[c.outcome or "in_progress"] = outcomes.get(c.outcome or "in_progress", 0) + 1
    return {
        "calls": len(calls),
        "escalations": sum(c.escalated for c in calls),
        "outcomes": outcomes,
        "open_inbox": sum(i.status == "open" for i in inbox_items),
        "unanswered_total": sum(i.kind == "unanswered" for i in inbox_items),
        "unanswered_resolved": sum(i.kind == "unanswered" and i.status == "resolved" for i in inbox_items),
        "latency_ms": {"agent_p50": statistics.median(agent) if agent else None,
                       "voice_to_voice_p50": statistics.median(v2v) if v2v else None},
    }
