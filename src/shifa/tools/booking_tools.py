"""Booking operations.

Privacy is enforced here, not in prompts: every function takes the caller's phone from the
session, so a prompt-injected "show me today's appointment list" has no tool that can do it.
"""

from datetime import date, datetime, time
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from shifa.config import clinic_now, get_settings
from shifa.db import Appointment, Doctor, InboxItem, Slot
from shifa.messaging import outbox

PARTS_OF_DAY = {"morning": (time(0), time(13)), "afternoon": (time(12), time(17)), "evening": (time(17), time(23, 59))}


class BookingError(ValueError):
    """Raised for invalid requests; the message is shown to the LLM so it can correct itself."""


def _part_of_day(t: datetime) -> str:
    # Spelled out because the model once offered a 5 PM slot as "subah 5 baje" (5 in the morning).
    # Urdu script too: a voice caller's 5 PM slot was still called "دوپہر" with Roman-only labels.
    if t.hour < 12:
        return "morning / subah / صبح"
    if t.hour < 17:
        return "afternoon / dopahar / دوپہر"
    return "evening / shaam / شام"


def _relative_day(d: date) -> str:
    # The model said "aaj" for tomorrow and "kal Saturday" on a Monday; give it the words instead of making it compute.
    delta = (d - clinic_now().date()).days
    return {0: ", today / aaj / آج", 1: ", tomorrow / kal / کل",
            2: ", day after tomorrow / parson / پرسوں"}.get(delta, "")


def fmt_slot(slot: Slot, doctor: Doctor) -> str:
    return (f"slot_id={slot.id} | {doctor.name} | {slot.start:%a %d %b %Y, %I:%M %p} "
            f"({_part_of_day(slot.start)}{_relative_day(slot.start.date())})")


def find_doctor(s: Session, name: str) -> Doctor:
    q = name.lower().replace("dr.", "").replace("dr ", "").strip()
    for doc in s.exec(select(Doctor)).all():
        if q and q in doc.name.lower():
            return doc
    names = ", ".join(d.name for d in s.exec(select(Doctor)).all())
    raise BookingError(f"No doctor matches '{name}'. Doctors: {names}")


def list_doctors(s: Session) -> str:
    return "\n".join(f"{d.name} ({d.title}), fee Rs. {d.fee}" for d in s.exec(select(Doctor)).all())


def get_available_slots(s: Session, day: str, doctor: str | None = None, part_of_day: str | None = None,
                        around_time: str | None = None, limit: int = 6) -> str:
    try:
        d = date.fromisoformat(day)
    except ValueError:
        raise BookingError(f"date must be YYYY-MM-DD, got '{day}'")
    if part_of_day and part_of_day not in PARTS_OF_DAY:
        raise BookingError(f"part_of_day must be one of {list(PARTS_OF_DAY)}")
    target = None
    if around_time:
        try:
            target = time.fromisoformat(around_time)
        except ValueError:
            raise BookingError(f"around_time must be HH:MM in 24h, e.g. 19:00, got '{around_time}'")
    q = (select(Slot, Doctor).join(Doctor).where(Slot.is_booked == False)  # noqa: E712
         .where(Slot.start >= datetime.combine(d, time(0)))
         .where(Slot.start <= datetime.combine(d, time(23, 59)))
         .where(Slot.start > clinic_now()).order_by(Slot.start))
    if doctor:
        q = q.where(Slot.doctor_id == find_doctor(s, doctor).id)
    rows = s.exec(q).all()
    if part_of_day:
        lo, hi = PARTS_OF_DAY[part_of_day]
        rows = [(sl, doc) for sl, doc in rows if lo <= sl.start.time() < hi]
    if not rows:
        return f"No free slots on {d:%A %d %b %Y} for that request (clinic is closed on Sunday)."
    # Header says the full range: with only the first 6 rows, the model once told a caller that 7 PM didn't
    # exist when Dr. Usman sees patients until 10 PM.
    header = (f"{len(rows)} free slots for this request, from {rows[0][0].start:%I:%M %p} to "
              f"{rows[-1][0].start:%I:%M %p}.")
    if target:
        minutes = target.hour * 60 + target.minute
        rows = sorted(rows, key=lambda r: abs(r[0].start.hour * 60 + r[0].start.minute - minutes))[:limit]
        rows.sort(key=lambda r: r[0].start)
        header += f" Showing the {len(rows)} closest to {target:%I:%M %p}:"
    else:
        header += f" Showing the first {min(limit, len(rows))}:" if len(rows) > limit else ""
    return "\n".join([header, *(fmt_slot(sl, doc) for sl, doc in rows[:limit])])


def _active_appointments(s: Session, phone: str):
    return s.exec(select(Appointment, Slot, Doctor).join(Slot, Appointment.slot_id == Slot.id).join(Doctor)
                  .where(Appointment.phone == phone).where(Appointment.status == "booked")
                  .where(Slot.start > clinic_now()).order_by(Slot.start)).all()


def my_appointments(s: Session, phone: str) -> str:
    rows = _active_appointments(s, phone)
    if not rows:
        return "This caller has no upcoming appointments."
    return "\n".join(f"appointment_id={a.id} | {a.patient_name} | {fmt_slot(sl, doc)}" for a, sl, doc in rows)


def _free_slot(s: Session, slot_id: int) -> tuple[Slot, Doctor]:
    slot = s.get(Slot, slot_id)
    if slot is None:
        raise BookingError(f"slot_id {slot_id} does not exist. Call get_available_slots first.")
    if slot.is_booked:
        raise BookingError(f"slot_id {slot_id} is already booked. Offer the caller another slot.")
    if slot.start <= clinic_now():
        raise BookingError(f"slot_id {slot_id} is in the past.")
    return slot, s.get(Doctor, slot.doctor_id)


def book_appointment(s: Session, phone: str, slot_id: int, patient_name: str, reason: str = "") -> str:
    if not patient_name.strip():
        raise BookingError("patient_name is required. Ask the caller for the patient's name.")
    slot, doc = _free_slot(s, slot_id)
    for a, sl, d in _active_appointments(s, phone):
        same_patient = a.patient_name.strip().lower() == patient_name.strip().lower()
        if same_patient and d.id == doc.id and sl.start.date() == slot.start.date():
            raise BookingError(f"{patient_name} already has appointment_id={a.id} with {d.name} that day "
                               f"({sl.start:%I:%M %p}). Offer to reschedule instead of double booking.")
    slot.is_booked = True
    appt = Appointment(slot_id=slot.id, patient_name=patient_name.strip(), phone=phone, reason=reason)
    s.add(appt)
    s.flush()  # assigns appt.id for the outbox rows
    outbox.on_booked(s, appt, slot, doc)
    s.commit()  # booking + its confirmation/reminder messages: saved together or not at all
    return (f"BOOKED appointment_id={appt.id} for {appt.patient_name}: {fmt_slot(slot, doc)}, fee Rs. {doc.fee}."
            f"{_messaging_note()}")


def _own_appointment(s: Session, phone: str, appointment_id: int) -> Appointment:
    appt = s.get(Appointment, appointment_id)
    # Same message whether it doesn't exist or belongs to someone else: no information leak.
    if appt is None or appt.phone != phone or appt.status != "booked":
        raise BookingError(f"No active appointment {appointment_id} for this caller. Use my_appointments.")
    return appt


def cancel_appointment(s: Session, phone: str, appointment_id: int) -> str:
    appt = _own_appointment(s, phone, appointment_id)
    appt.status = "cancelled"
    slot = s.get(Slot, appt.slot_id)
    slot.is_booked = False
    outbox.on_cancelled(s, appt, slot, s.get(Doctor, slot.doctor_id))
    s.commit()
    return f"CANCELLED appointment_id={appointment_id}.{_messaging_note()}"


def reschedule_appointment(s: Session, phone: str, appointment_id: int, new_slot_id: int) -> str:
    appt = _own_appointment(s, phone, appointment_id)
    new_slot, doc = _free_slot(s, new_slot_id)
    s.get(Slot, appt.slot_id).is_booked = False
    new_slot.is_booked = True
    appt.slot_id = new_slot.id
    outbox.on_rescheduled(s, appt, new_slot, doc)
    s.commit()
    return f"RESCHEDULED appointment_id={appointment_id} to {fmt_slot(new_slot, doc)}.{_messaging_note()}"


def _messaging_note() -> str:
    """Tells the agent what it may promise. Only a REAL provider counts: in "simulate" mode nothing reaches the
    caller's phone, so Hina must not say a message is coming."""
    if get_settings().messaging_provider == "whatsapp":
        return " The clinic will automatically send a WhatsApp confirmation and a reminder to this number."
    return ""


def describe_appointment(s: Session, phone: str, appointment_id: int) -> str:
    appt = _own_appointment(s, phone, appointment_id)
    slot = s.get(Slot, appt.slot_id)
    return f"{appt.patient_name} with {s.get(Doctor, slot.doctor_id).name} on {slot.start:%a %d %b, %I:%M %p}"


class ConfirmationGate:
    """Destructive actions need the caller's yes in a LATER turn than the one that asked.

    A prompt rule ("confirm first") was ignored in testing: "cancel my appointment" cancelled one instantly,
    without saying which. Here the first call only records the request and tells the model to ask; the same
    call with confirmed=true works only from the next caller turn on. The model can't skip the question.
    """

    def __init__(self):
        self._pending: dict[tuple, int] = {}  # (action, args...) -> turn the confirmation was requested

    def check(self, key: tuple, confirmed: bool, turn: int, describe: str, asked_last_turn: bool = False) -> str | None:
        """asked_last_turn: Hina's previous reply asked the caller to confirm. Models often ask in their own words
        without calling the tool first; without this, a caller's "ji haan" got a second identical question."""
        asked_at = self._pending.get(key)
        if confirmed and ((asked_at is not None and asked_at < turn) or asked_last_turn):
            self._pending.pop(key, None)
            return None  # proceed
        self._pending[key] = turn if asked_at is None else asked_at
        return (f"CONFIRMATION NEEDED (nothing changed yet): {describe}. Ask the caller to confirm this exact "
                f"appointment in plain words (no ids), wait for their answer, then call again with confirmed=true.")


_GATES: dict[str, ConfirmationGate] = {}


def leave_message(s: Session, phone: str, call_id: str | None, message: str, kind: str = "message") -> str:
    s.add(InboxItem(call_id=call_id, phone=phone, kind=kind, text=message))
    s.commit()
    return "Message saved for clinic staff. They will call back during clinic hours."


class NoArgs(BaseModel):
    pass


class SlotsArgs(BaseModel):
    # Named "date" because the model kept sending date=... when this was called "day" (and put "Monday" in day).
    date: str = Field(description="Date as YYYY-MM-DD, e.g. 2026-11-03")
    doctor: str | None = Field(None, description="Doctor name, e.g. 'Ayesha'")
    part_of_day: Literal["morning", "afternoon", "evening"] | None = None
    around_time: str | None = Field(None, description="If the caller asks for a specific time: HH:MM 24h, e.g. "
                                                      "'19:00' for 7 PM. Returns the slots closest to it.")


class BookArgs(BaseModel):
    slot_id: int = Field(description="slot_id from get_available_slots")
    patient_name: str
    reason: str = Field("", description="Short reason for visit, e.g. 'fever'")


class ApptArgs(BaseModel):
    appointment_id: int
    confirmed: bool = Field(False, description="true only after the caller said yes to this exact appointment")


class RescheduleArgs(BaseModel):
    appointment_id: int
    new_slot_id: int
    confirmed: bool = Field(False, description="true only after the caller said yes to this exact change")


class MessageArgs(BaseModel):
    message: str = Field(description="Caller name, what they need, and best callback time")


def make_tools(engine, phone: str, call_id: str | None = None, turn: int = 0,
               asked_last_turn: bool = False) -> dict[str, list[StructuredTool]]:
    """LangChain tools with the caller's phone bound in. `turn` = index of the current caller turn and
    `asked_last_turn` = Hina's previous reply asked for confirmation; both feed the confirmation gate for
    destructive actions. Returns tool sets per agent."""
    gate = _GATES.setdefault(call_id or phone, ConfirmationGate())

    def tool(fn, schema, name, description, with_phone=True):
        def run(**kwargs):
            with Session(engine) as s:
                return fn(s, phone, **kwargs) if with_phone else fn(s, **kwargs)
        return StructuredTool.from_function(run, name=name, description=description, args_schema=schema)

    def gated_cancel(s, phone, appointment_id, confirmed=False):
        stop = gate.check(("cancel", appointment_id), confirmed, turn,
                          f"cancel {describe_appointment(s, phone, appointment_id)}", asked_last_turn)
        return stop or cancel_appointment(s, phone, appointment_id)

    def gated_reschedule(s, phone, appointment_id, new_slot_id, confirmed=False):
        slot, doc = _free_slot(s, new_slot_id)
        stop = gate.check(("reschedule", appointment_id, new_slot_id), confirmed, turn,
                          f"move {describe_appointment(s, phone, appointment_id)} to {doc.name} on "
                          f"{slot.start:%a %d %b, %I:%M %p}", asked_last_turn)
        return stop or reschedule_appointment(s, phone, appointment_id, new_slot_id)

    booking = [
        tool(list_doctors, NoArgs, "list_doctors", "List clinic doctors with qualifications and fees.",
             with_phone=False),
        tool(lambda s, date, doctor=None, part_of_day=None, around_time=None:
             get_available_slots(s, date, doctor, part_of_day, around_time),
             SlotsArgs, "get_available_slots", "Free appointment slots on a date.", with_phone=False),
        tool(my_appointments, NoArgs, "my_appointments", "Upcoming appointments for THIS caller only."),
        tool(book_appointment, BookArgs, "book_appointment",
             "Book a slot for a patient. Confirm name and slot with the caller first."),
        tool(gated_cancel, ApptArgs, "cancel_appointment",
             "Cancel one of this caller's appointments. Needs the caller's explicit confirmation."),
        tool(gated_reschedule, RescheduleArgs, "reschedule_appointment",
             "Move one of this caller's appointments to a new slot. Needs the caller's explicit confirmation."),
    ]
    handoff = [tool(lambda s, phone, message: leave_message(s, phone, call_id, message), MessageArgs,
                    "leave_message_for_staff", "Save a message for clinic staff to call back.")]
    return {"booking": booking, "handoff": handoff}
