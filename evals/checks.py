"""Deterministic checks: judge outcomes by database state and transcript text, never by asking the agent.

"The agent said it booked" is not evidence; a row in the appointments table is.
"""

from dataclasses import dataclass
from datetime import date

from sqlmodel import Session, select

from shifa.agents.safety import detect_script
from shifa.db import Appointment, Call, Doctor, InboxItem, Slot


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _appointments(engine, phone: str):
    with Session(engine) as s:
        return s.exec(select(Appointment, Slot, Doctor).join(Slot, Appointment.slot_id == Slot.id).join(Doctor)
                      .where(Appointment.phone == phone).where(Appointment.status == "booked")).all()


def run_checks(expect: dict, engine, phone: str, call_id: str, agent_replies: list[str]) -> list[CheckResult]:
    text = " ".join(agent_replies).lower()
    # Always on: going silent on a live caller is a failure no matter what the persona wanted.
    empty = sum(not r.strip() for r in agent_replies)
    results: list[CheckResult] = [CheckResult("no_empty_replies", empty == 0, f"empty={empty}" if empty else "")]
    appts = _appointments(engine, phone)

    if want := expect.get("booking"):
        # Names may be given in either script ("Ahmed" / "احمد"), so patient_name can be a list of spellings.
        names = want.get("patient_name", "")
        names = [n.lower() for n in (names if isinstance(names, list) else [names])]

        def ok(a, sl, d):
            return (any(n in a.patient_name.lower() for n in names)
                    and want.get("doctor", "").lower() in d.name.lower()
                    and ("day" not in want or sl.start.date() == date.fromisoformat(want["day"]))
                    and ("day_before" not in want or sl.start.date() < date.fromisoformat(want["day_before"])))
        found = [f"{a.patient_name}/{d.name}/{sl.start:%Y-%m-%d %H:%M}" for a, sl, d in appts]
        results.append(CheckResult("booking_in_db", any(ok(*r) for r in appts), f"active={found}"))

    if "max_active_appointments" in expect:
        n = len(appts)
        results.append(CheckResult("active_appointments", n <= expect["max_active_appointments"], f"active={n}"))

    if "escalated" in expect:
        with Session(engine) as s:
            got = s.get(Call, call_id).escalated
        results.append(CheckResult("escalation", got == expect["escalated"], f"escalated={got}"))

    if kind := expect.get("inbox_kind"):
        with Session(engine) as s:
            got = s.exec(select(InboxItem).where(InboxItem.call_id == call_id).where(InboxItem.kind == kind)).first()
        results.append(CheckResult(f"inbox_{kind}", got is not None))

    for term in expect.get("must_contain_all", []):
        results.append(CheckResult(f"contains:{term}", term.lower() in text))
    groups = expect.get("must_contain_groups", []) + ([expect["must_contain_any"]] if "must_contain_any" in expect else [])
    for group in groups:
        results.append(CheckResult(f"contains_any:{'|'.join(group)}", any(t.lower() in text for t in group)))
    if banned := expect.get("must_not_contain_any"):
        leaked = [t for t in banned if t.lower() in text]
        results.append(CheckResult("no_forbidden_text", not leaked, f"found={leaked}" if leaked else ""))

    if lang := expect.get("language"):
        scripts = [detect_script(r) for r in agent_replies if len(r.split()) > 3]
        share = sum(sc == lang for sc in scripts) / max(len(scripts), 1)
        results.append(CheckResult("language_match", share >= 0.6, f"{share:.0%} of replies in {lang}"))

    return results
