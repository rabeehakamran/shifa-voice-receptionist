"""Caller Simulator Arena. Run: python -m evals.run_evals [--only id1,id2] [--label baseline] [--no-judge]

Every persona gets a fresh in-memory clinic DB seeded for pinned "today" (Mon 2026-11-02), so runs are
comparable across prompt/model changes. Writes reports/latest.{json,md} and appends reports/history.jsonl.
"""

import os

os.environ.setdefault("SHIFA_TODAY", "2026-11-02")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zlib  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import date, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402
from rich.console import Console  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

from evals.checks import run_checks  # noqa: E402
from evals.judge import judge  # noqa: E402
from evals.simulator import HANGUP, SimulatedCaller  # noqa: E402
from shifa import llm  # noqa: E402
from shifa.agents import guardrails  # noqa: E402
from shifa.config import ROOT, clinic_today, get_settings  # noqa: E402
from shifa.seed import seed  # noqa: E402
from shifa.session import CallSession  # noqa: E402
from shifa.tools import booking_tools as bt  # noqa: E402

console = Console()
REPORTS = ROOT / "reports"
PERSONAS = Path(__file__).with_name("personas.yaml")


def fresh_engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    seed(eng, start=clinic_today())
    return eng


def apply_setup(engine, setup: dict, phone: str) -> None:
    items = [(a, phone) for a in setup.get("appointments", [])]
    items += [(a, a["phone"]) for a in setup.get("other_patients", [])]
    with Session(engine) as s:
        for appt, ph in items:
            slots = bt.get_available_slots(s, appt["day"], doctor=appt["doctor"])
            slot_id = int(re.search(r"slot_id=(\d+)", slots).group(1))
            bt.book_appointment(s, ph, slot_id, appt["patient_name"])


def run_persona(p: dict, max_turns: int, pace: float, use_judge: bool, label: str = "") -> dict:
    engine = fresh_engine()
    phone = "0300" + str(zlib.crc32(p["id"].encode()) % 10_000_000).zfill(7)  # stable across runs
    apply_setup(engine, p.get("setup", {}), phone)
    call = CallSession(phone, channel="sim", engine=engine, trace_tags=["arena", p["category"], p["id"]],
                       trace_metadata={"persona": p["id"], "arena_label": label})
    caller = SimulatedCaller(p)
    replies, turns, reply = [], [], None
    for _ in range(max_turns):
        line = caller.next_line(reply)
        if HANGUP in line:
            break
        r = call.turn(line)
        reply = r.reply
        replies.append(reply)
        turns.append({"caller": line, "agent": reply, "route": r.route, "risk": r.risk,
                      "citations": r.citations, "timings_ms": r.timings_ms})
        time.sleep(pace)  # stay under free-tier rate limits
    record = call.end(summarize=False)
    checks = run_checks(p.get("expect", {}), engine, phone, call.call_id, replies)
    out = {"id": p["id"], "category": p["category"], "passed": all(c.passed for c in checks),
           "checks": [asdict(c) for c in checks], "turns": turns, "escalated": record.escalated}
    if use_judge and turns:
        try:
            out["judge"] = judge(call.transcript).model_dump()
        except Exception as e:
            out["judge_error"] = repr(e)
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0


def summarize(all_results: list[dict], personas: dict) -> dict:
    results = [r for r in all_results if r.get("status") != "error"]

    def rate(rs):
        return round(sum(r["passed"] for r in rs) / len(rs), 3) if rs else None

    cats = sorted({r["category"] for r in results})
    should_escalate = [r for r in results if personas[r["id"]].get("expect", {}).get("escalated") is True]
    must_not = [r for r in results if personas[r["id"]].get("expect", {}).get("escalated") is not True]
    totals = [t["timings_ms"]["total"] for r in results for t in r["turns"]]
    stages = {}
    for r in results:
        for t in r["turns"]:
            for k, v in t["timings_ms"].items():
                stages.setdefault(k, []).append(v)
    judged = [r["judge"] for r in results if "judge" in r]
    return {
        "evaluated": len(results),
        "not_run_infra_errors": len(all_results) - len(results),
        "task_success": rate(results),
        "by_category": {c: rate([r for r in results if r["category"] == c]) for c in cats},
        "emergency_recall": rate(should_escalate),
        "false_escalations": sum(r["escalated"] for r in must_not),
        "injection_resistance": rate([r for r in results if r["category"] == "security"]),
        "avg_turns": round(statistics.mean(len(r["turns"]) for r in results), 1) if results else 0,
        "latency_ms": {"p50": pct(totals, 0.5), "p95": pct(totals, 0.95),
                       "stages_p50": {k: pct(v, 0.5) for k, v in stages.items() if k != "total"}},
        "judge": {k: round(statistics.mean(j[k] for j in judged), 2) for k in ("grounded", "helpful", "tone")}
        if judged else None,
        "hallucinations": sum(len(j["hallucinations"]) for j in judged),
        "guardrail_triggers": dict(guardrails.TRIGGERS),
        "llm_fallbacks": dict(llm.FALLBACKS),
    }


def write_reports(label: str, summary: dict, results: list[dict], name: str = "latest") -> Path:
    """Full runs write latest.* (read by the CI gate); --only smoke runs write smoke.* so they can't move the gate."""
    REPORTS.mkdir(exist_ok=True)
    s = get_settings()
    meta = {"label": label, "at": datetime.now().isoformat(timespec="seconds"),
            "models": {"fast": s.fast_model, "smart": s.smart_model, "sim_judge": s.gemini_model}}
    (REPORTS / f"{name}.json").write_text(json.dumps({**meta, "summary": summary, "results": results},
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    with open(REPORTS / f"{name}_history.jsonl" if name != "latest" else REPORTS / "history.jsonl", "a",
              encoding="utf-8") as f:
        f.write(json.dumps({**meta, "summary": summary}) + "\n")

    lines = [f"# Caller Simulator Arena: {label}", f"_{meta['at']} · models: {meta['models']}_", "",
             "| Metric | Value |", "|---|---|"]
    for k in ("evaluated", "not_run_infra_errors", "task_success", "emergency_recall", "false_escalations",
              "injection_resistance", "avg_turns", "hallucinations"):
        lines.append(f"| {k} | {summary[k]} |")
    lines.append(f"| guardrail triggers | {summary['guardrail_triggers'] or 0} |")
    lines.append(f"| LLM provider fallbacks | {summary['llm_fallbacks'] or 0} |")
    lines.append(f"| latency p50 / p95 (ms) | {summary['latency_ms']['p50']} / {summary['latency_ms']['p95']} |")
    if summary["judge"]:
        lines.append(f"| judge grounded / helpful / tone (1-5) | {summary['judge']['grounded']} / "
                     f"{summary['judge']['helpful']} / {summary['judge']['tone']} |")
    lines += ["", "**By category:** " + ", ".join(f"{k} {v:.0%}" for k, v in summary["by_category"].items()
                                                   if v is not None),
              "", "**Stage latency p50 (ms):** " + ", ".join(f"{k} {v}" for k, v in
                                                            summary["latency_ms"]["stages_p50"].items()),
              "", "## Personas", "", "| Persona | Result | Failed checks | Turns |", "|---|---|---|---|"]
    for r in results:
        failed = "; ".join(f"{c['name']} {c['detail']}".strip() for c in r["checks"] if not c["passed"])
        mark = "⚠️ not run" if r.get("status") == "error" else "✅" if r["passed"] else "❌"
        lines.append(f"| {r['id']} | {mark} | {failed or r.get('error', '')} | {len(r['turns'])} |")
    path = REPORTS / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated persona ids")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--pace", type=float, default=2.0, help="seconds between turns (free-tier rate limits)")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    assert clinic_today() == date(2026, 11, 2), "evals expect SHIFA_TODAY=2026-11-02"

    personas = {p["id"]: p for p in yaml.safe_load(PERSONAS.read_text(encoding="utf-8"))}
    ids = args.only.split(",") if args.only else list(personas)
    results = []
    for pid in ids:
        console.print(f"[bold]▶ {pid}[/]")
        try:
            r = run_persona(personas[pid], args.max_turns, args.pace, not args.no_judge, args.label)
        except Exception as e:
            # Infra failure (quota, network): this persona was NOT evaluated; it must not count as an agent failure.
            r = {"id": pid, "category": personas[pid]["category"], "passed": False, "status": "error", "checks": [],
                 "turns": [], "escalated": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            console.print(traceback.format_exc(), style="red")
        for t in r["turns"]:
            console.print(f"  [cyan]caller:[/] {t['caller']}\n  [magenta]hina:[/] {t['agent']}  "
                          f"[dim]({t['route']}, {t['timings_ms']['total']}ms)[/]")
        console.print("  " + ("[green]PASS[/]" if r["passed"] else "[red]FAIL[/]") + " " + ", ".join(
            f"{c['name']}{'✓' if c['passed'] else '✗ ' + c['detail']}" for c in r["checks"]))
        results.append(r)

    summary = summarize(results, personas)
    path = write_reports(args.label, summary, results, name="smoke" if args.only else "latest")
    console.print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
