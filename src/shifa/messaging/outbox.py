"""Transactional outbox for patient messages (confirmations, reminders, changes).

These functions only ADD rows to the session; the caller's commit saves them together with the booking change.
Nothing here talks to WhatsApp: the dispatcher does that later, with retries.

Messages are bilingual (Urdu script, then English): in Pakistan the same number is often read by a parent, a
child or a grandparent, and both scripts are common.
"""

from datetime import datetime, time, timedelta

from sqlmodel import Session, select

from shifa.config import clinic_now
from shifa.db import Appointment, Doctor, OutboxMessage, Slot

REMINDER_EVENING_BEFORE = time(18, 0)
REMINDER_HOURS_BEFORE = 2
MIN_LEAD = timedelta(minutes=30)  # don't send a "reminder" for something about to start

DOCTOR_UR = {"Dr. Ayesha Malik": "ڈاکٹر عائشہ ملک", "Dr. Usman Tariq": "ڈاکٹر عثمان طارق",
             "Dr. Sana Iqbal": "ڈاکٹر ثنا اقبال"}
WEEKDAY_UR = ["پیر", "منگل", "بدھ", "جمعرات", "جمعہ", "ہفتہ", "اتوار"]


def _when_ur(t: datetime) -> str:
    part = "صبح" if t.hour < 12 else "دوپہر" if t.hour < 17 else "شام"
    hour = t.hour % 12 or 12
    clock = f"{hour}:{t.minute:02d}" if t.minute else f"{hour}"
    return f"{WEEKDAY_UR[t.weekday()]} {t.day} تاریخ، {part} {clock} بجے"


def _when_en(t: datetime) -> str:
    return f"{t:%A %d %b} at {t:%I:%M %p}".replace(" 0", " ")


def render(kind: str, patient: str, doctor: Doctor, start: datetime) -> str:
    d_ur, w_ur, w_en = DOCTOR_UR.get(doctor.name, doctor.name), _when_ur(start), _when_en(start)
    ur, en = {
        "confirmation": (f"{patient} کی اپائنٹمنٹ {d_ur} کے ساتھ {w_ur} کنفرم ہے۔ فیس {doctor.fee} روپے۔ "
                         f"براہِ کرم 10 منٹ پہلے تشریف لائیں۔",
                         f"{patient}'s appointment with {doctor.name} is confirmed for {w_en}. Fee Rs. {doctor.fee}. "
                         f"Please arrive 10 minutes early."),
        "reminder": (f"یاد دہانی: {patient} کی اپائنٹمنٹ {d_ur} کے ساتھ {w_ur} ہے۔",
                     f"Reminder: {patient}'s appointment with {doctor.name} is on {w_en}."),
        "rescheduled": (f"{patient} کی اپائنٹمنٹ تبدیل ہو گئی ہے: اب {d_ur} کے ساتھ {w_ur}۔",
                        f"{patient}'s appointment was moved: now with {doctor.name} on {w_en}."),
        "cancelled": (f"{patient} کی {d_ur} کے ساتھ {w_ur} والی اپائنٹمنٹ منسوخ کر دی گئی ہے۔",
                      f"{patient}'s appointment with {doctor.name} on {w_en} has been cancelled."),
    }[kind]
    return f"شفا فیملی کلینک\n{ur}\n\nShifa Family Clinic\n{en}\nCall/WhatsApp: 0300-7445566"


def reminder_time(start: datetime, now: datetime) -> datetime | None:
    """6 PM the evening before; if that's already past, 2 hours before; if even that is too close, none."""
    for candidate in (datetime.combine(start.date() - timedelta(days=1), REMINDER_EVENING_BEFORE),
                      start - timedelta(hours=REMINDER_HOURS_BEFORE)):
        if now + MIN_LEAD <= candidate < start:
            return candidate
    return None


def _queue(s: Session, appt: Appointment, kind: str, doctor: Doctor, start: datetime, send_after: datetime) -> None:
    s.add(OutboxMessage(appointment_id=appt.id, phone=appt.phone, kind=kind, send_after=send_after,
                        body=render(kind, appt.patient_name, doctor, start)))


def _cancel_pending(s: Session, appointment_id: int) -> None:
    """Drop not-yet-sent messages (e.g. the old reminder) when an appointment moves or is cancelled."""
    for msg in s.exec(select(OutboxMessage).where(OutboxMessage.appointment_id == appointment_id)
                      .where(OutboxMessage.status == "pending")).all():
        msg.status = "cancelled"
        s.add(msg)


def on_booked(s: Session, appt: Appointment, slot: Slot, doctor: Doctor) -> None:
    now = clinic_now()
    _queue(s, appt, "confirmation", doctor, slot.start, now)
    if (at := reminder_time(slot.start, now)) is not None:
        _queue(s, appt, "reminder", doctor, slot.start, at)


def on_rescheduled(s: Session, appt: Appointment, slot: Slot, doctor: Doctor) -> None:
    _cancel_pending(s, appt.id)
    now = clinic_now()
    _queue(s, appt, "rescheduled", doctor, slot.start, now)
    if (at := reminder_time(slot.start, now)) is not None:
        _queue(s, appt, "reminder", doctor, slot.start, at)


def on_cancelled(s: Session, appt: Appointment, slot: Slot, doctor: Doctor) -> None:
    _cancel_pending(s, appt.id)
    _queue(s, appt, "cancelled", doctor, slot.start, clinic_now())
