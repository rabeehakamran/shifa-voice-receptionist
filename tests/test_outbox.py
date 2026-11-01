"""Transactional outbox + dispatcher. 'Today' is pinned to Monday 2026-11-02, 08:00 (see conftest)."""

import re
from datetime import datetime

import pytest
from sqlmodel import Session, select

from shifa.db import OutboxMessage
from shifa.messaging import dispatch
from shifa.messaging.dispatch import SendResult, dispatch_due, to_e164
from shifa.tools import booking_tools as bt

ME = "03001112222"


def book(engine, day, doctor="ayesha", name="Ali Raza", index=0):
    with Session(engine) as s:
        slot_ids = re.findall(r"slot_id=(\d+)", bt.get_available_slots(s, day, doctor=doctor))
        out = bt.book_appointment(s, ME, int(slot_ids[index]), name)
    return int(re.search(r"appointment_id=(\d+)", out).group(1))


def outbox(engine, **where):
    with Session(engine) as s:
        rows = s.exec(select(OutboxMessage).order_by(OutboxMessage.id)).all()
    return [r for r in rows if all(getattr(r, k) == v for k, v in where.items())]


def test_booking_queues_confirmation_now_and_reminder_evening_before(engine):
    book(engine, "2026-11-04")  # Wednesday morning
    conf, rem = outbox(engine, kind="confirmation"), outbox(engine, kind="reminder")
    assert len(conf) == 1 and conf[0].send_after == datetime(2026, 11, 2, 8, 0)
    assert len(rem) == 1 and rem[0].send_after == datetime(2026, 11, 3, 18, 0)  # Tuesday 6 PM
    assert "ڈاکٹر عائشہ ملک" in conf[0].body and "Dr. Ayesha Malik" in conf[0].body  # bilingual


def test_same_day_evening_slot_gets_reminder_two_hours_before(engine):
    book(engine, "2026-11-02", doctor="usman")  # today 5:00 PM; "6 PM yesterday" is past
    assert outbox(engine, kind="reminder")[0].send_after == datetime(2026, 11, 2, 15, 0)


def test_slot_too_soon_gets_no_reminder(engine):
    book(engine, "2026-11-02", doctor="ayesha")  # today 9:00 AM, now is 8:00: a reminder would be pointless
    assert outbox(engine, kind="reminder") == []
    assert len(outbox(engine, kind="confirmation")) == 1


def test_failed_booking_queues_nothing(engine):
    book(engine, "2026-11-04", name="Ali Raza")
    before = len(outbox(engine))
    with pytest.raises(bt.BookingError):  # double booking: same patient, doctor and day
        book(engine, "2026-11-04", name="Ali Raza", index=1)
    assert len(outbox(engine)) == before  # same transaction: no booking, no message


def test_cancel_drops_pending_reminder_and_sends_cancellation(engine):
    appt = book(engine, "2026-11-04")
    with Session(engine) as s:
        bt.cancel_appointment(s, ME, appt)
    assert outbox(engine, kind="reminder")[0].status == "cancelled"
    assert len(outbox(engine, kind="cancelled", status="pending")) == 1


def test_reschedule_replaces_the_reminder(engine):
    appt = book(engine, "2026-11-04")
    with Session(engine) as s:
        new_slot = int(re.search(r"slot_id=(\d+)", bt.get_available_slots(s, "2026-11-06", doctor="usman")).group(1))
        bt.reschedule_appointment(s, ME, appt, new_slot)
    reminders = outbox(engine, kind="reminder")
    assert [r.status for r in reminders] == ["cancelled", "pending"]
    assert reminders[1].send_after == datetime(2026, 11, 5, 18, 0)  # evening before the NEW day
    assert len(outbox(engine, kind="rescheduled")) == 1


def test_dispatcher_sends_only_due_messages(engine):
    book(engine, "2026-11-04")
    counts = dispatch_due(engine, dispatch.SimulatedSender(), now=datetime(2026, 11, 2, 8, 0))
    assert counts["sent"] == 1  # the confirmation; the reminder is for Tuesday evening
    assert outbox(engine, kind="confirmation")[0].status == "simulated"
    assert outbox(engine, kind="reminder")[0].status == "pending"
    dispatch_due(engine, dispatch.SimulatedSender(), now=datetime(2026, 11, 3, 18, 1))
    assert outbox(engine, kind="reminder")[0].status == "simulated"


class Flaky:
    status_on_success = "sent"

    def __init__(self, failures, retryable=True):
        self.failures, self.retryable = failures, retryable

    def send(self, phone, body):
        if self.failures:
            self.failures -= 1
            return SendResult(ok=False, error="503 from provider", retryable=self.retryable)
        return SendResult(ok=True, external_id="wamid.123")


def test_failures_retry_with_exponential_backoff(engine):
    book(engine, "2026-11-04")
    sender, t = Flaky(failures=2), datetime(2026, 11, 2, 8, 0)
    dispatch_due(engine, sender, now=t)
    msg = outbox(engine, kind="confirmation")[0]
    assert msg.status == "pending" and msg.send_after == datetime(2026, 11, 2, 8, 2)  # retry in 2 min
    dispatch_due(engine, sender, now=datetime(2026, 11, 2, 8, 2))
    assert outbox(engine, kind="confirmation")[0].send_after == datetime(2026, 11, 2, 8, 6)  # then 4 min
    dispatch_due(engine, sender, now=datetime(2026, 11, 2, 8, 6))
    msg = outbox(engine, kind="confirmation")[0]
    assert msg.status == "sent" and msg.attempts == 3 and msg.external_id == "wamid.123"


def test_non_retryable_error_fails_immediately(engine):
    book(engine, "2026-11-04")
    dispatch_due(engine, Flaky(failures=1, retryable=False), now=datetime(2026, 11, 2, 8, 0))
    msg = outbox(engine, kind="confirmation")[0]
    assert msg.status == "failed" and msg.attempts == 1


def test_reminder_after_the_visit_expires_instead_of_sending(engine):
    book(engine, "2026-11-04")  # Wednesday 9 AM; pretend the server was off until Thursday
    dispatch_due(engine, dispatch.SimulatedSender(), now=datetime(2026, 11, 5, 9, 0))
    assert outbox(engine, kind="reminder")[0].status == "expired"


def test_pakistani_numbers_to_international_format():
    assert to_e164("03001234567") == "923001234567"
    assert to_e164("0300-1234567") == "923001234567"
    assert to_e164("+92 300 1234567") == "923001234567"
    assert to_e164("00923001234567") == "923001234567"


def test_agent_only_promises_messages_when_a_real_provider_is_set(engine, monkeypatch):
    from shifa.agents import prompts
    from shifa.config import get_settings

    sid = int(re.search(r"slot_id=(\d+)", bt.get_available_slots(Session(engine), "2026-11-05")).group(1))
    with Session(engine) as s:
        assert "WhatsApp" not in bt.book_appointment(s, ME, sid, "Ali")  # simulate mode: nothing is promised
    assert "CANNOT send" in prompts.messaging_rule()
    monkeypatch.setattr(get_settings(), "messaging_provider", "whatsapp")
    assert "AUTOMATICALLY sends a WhatsApp" in prompts.messaging_rule()
