"""Booking operations.

Privacy is enforced here, not in prompts: every function takes the caller's phone from the
session, so a prompt-injected "show me today's appointment list" has no tool that can do it.
"""

from datetime import date, datetime, time
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from shifa.config import clinic_now
from shifa.db import Appointment, Doctor, InboxItem, Slot

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
                        limit: int = 6) -> str:
    try:
        d = date.fromisoformat(day)
    except ValueError:
        raise BookingError(f"date must be YYYY-MM-DD, got '{day}'")
    if part_of_day and part_of_day not in PARTS_OF_DAY:
        raise BookingError(f"part_of_day must be one of {list(PARTS_OF_DAY)}")
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
    return "\n".join(fmt_slot(sl, doc) for sl, doc in rows[:limit])


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
    s.commit()
    return f"BOOKED appointment_id={appt.id} for {appt.patient_name}: {fmt_slot(slot, doc)}, fee Rs. {doc.fee}"


def _own_appointment(s: Session, phone: str, appointment_id: int) -> Appointment:
    appt = s.get(Appointment, appointment_id)
    # Same message whether it doesn't exist or belongs to someone else: no information leak.
    if appt is None or appt.phone != phone or appt.status != "booked":
        raise BookingError(f"No active appointment {appointment_id} for this caller. Use my_appointments.")
    return appt


def cancel_appointment(s: Session, phone: str, appointment_id: int) -> str:
    appt = _own_appointment(s, phone, appointment_id)
    appt.status = "cancelled"
    s.get(Slot, appt.slot_id).is_booked = False
    s.commit()
    return f"CANCELLED appointment_id={appointment_id}"


def reschedule_appointment(s: Session, phone: str, appointment_id: int, new_slot_id: int) -> str:
    appt = _own_appointment(s, phone, appointment_id)
    new_slot, doc = _free_slot(s, new_slot_id)
    s.get(Slot, appt.slot_id).is_booked = False
    new_slot.is_booked = True
    appt.slot_id = new_slot.id
    s.commit()
    return f"RESCHEDULED appointment_id={appointment_id} to {fmt_slot(new_slot, doc)}"


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


class BookArgs(BaseModel):
    slot_id: int = Field(description="slot_id from get_available_slots")
    patient_name: str
    reason: str = Field("", description="Short reason for visit, e.g. 'fever'")


class ApptArgs(BaseModel):
    appointment_id: int


class RescheduleArgs(BaseModel):
    appointment_id: int
    new_slot_id: int


class MessageArgs(BaseModel):
    message: str = Field(description="Caller name, what they need, and best callback time")


def make_tools(engine, phone: str, call_id: str | None = None) -> dict[str, list[StructuredTool]]:
    """LangChain tools with the caller's phone bound in. Returns tool sets per agent."""

    def tool(fn, schema, name, description, with_phone=True):
        def run(**kwargs):
            with Session(engine) as s:
                return fn(s, phone, **kwargs) if with_phone else fn(s, **kwargs)
        return StructuredTool.from_function(run, name=name, description=description, args_schema=schema)

    booking = [
        tool(list_doctors, NoArgs, "list_doctors", "List clinic doctors with qualifications and fees.",
             with_phone=False),
        tool(lambda s, date, doctor=None, part_of_day=None: get_available_slots(s, date, doctor, part_of_day),
             SlotsArgs, "get_available_slots", "Free appointment slots on a date.", with_phone=False),
        tool(my_appointments, NoArgs, "my_appointments", "Upcoming appointments for THIS caller only."),
        tool(book_appointment, BookArgs, "book_appointment",
             "Book a slot for a patient. Confirm name and slot with the caller first."),
        tool(cancel_appointment, ApptArgs, "cancel_appointment", "Cancel one of this caller's appointments."),
        tool(reschedule_appointment, RescheduleArgs, "reschedule_appointment",
             "Move one of this caller's appointments to a new slot."),
    ]
    handoff = [tool(lambda s, phone, message: leave_message(s, phone, call_id, message), MessageArgs,
                    "leave_message_for_staff", "Save a message for clinic staff to call back.")]
    return {"booking": booking, "handoff": handoff}
