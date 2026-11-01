"""Outbox dispatcher: sends due messages, retries failures with exponential backoff.

Delivery is AT-LEAST-ONCE: a message is marked sent only after the provider accepts it, so a crash between the
two can cause one duplicate on restart, but a message is never silently lost. (Exactly-once would need provider
idempotency keys, which WhatsApp's API doesn't offer.) One dispatcher process is assumed; with several, you'd
claim rows first (e.g. SELECT ... FOR UPDATE SKIP LOCKED on Postgres).
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from sqlmodel import Session, select

from shifa.config import clinic_now, get_settings
from shifa.db import Appointment, OutboxMessage, Slot

log = logging.getLogger("shifa.outbox")
MAX_ATTEMPTS = 5
DISPATCH_EVERY_S = 15


@dataclass
class SendResult:
    ok: bool
    external_id: str = ""
    error: str = ""
    retryable: bool = True  # e.g. a network blip is retryable; an invalid phone number is not


class Sender(Protocol):
    status_on_success: str

    def send(self, phone: str, body: str) -> SendResult: ...


class SimulatedSender:
    """Development mode: nothing leaves the laptop; messages show up as "simulated" on the dashboard."""

    status_on_success = "simulated"

    def send(self, phone: str, body: str) -> SendResult:
        log.info("[simulated message to %s]\n%s", phone, body)
        return SendResult(ok=True, external_id=f"sim-{uuid.uuid4().hex[:8]}")


def to_e164(phone: str) -> str:
    """Pakistani mobile '0300-1234567' -> '923001234567' (international format, no '+')."""
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("0092"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = "92" + digits[1:]
    return digits


def get_sender() -> Sender:
    provider = get_settings().messaging_provider
    if provider == "simulate":
        return SimulatedSender()
    if provider == "whatsapp":
        from shifa.messaging.whatsapp import WhatsAppSender

        return WhatsAppSender()
    raise ValueError(f"Unknown SHIFA_MESSAGING_PROVIDER '{provider}' (use simulate or whatsapp)")


def _appointment_start(s: Session, appointment_id: int | None) -> datetime | None:
    if appointment_id is None:
        return None
    appt = s.get(Appointment, appointment_id)
    return s.get(Slot, appt.slot_id).start if appt else None


def dispatch_due(engine, sender: Sender, now: datetime | None = None, batch: int = 20) -> dict[str, int]:
    """Send every pending message whose send_after has passed. Commits after each message, so a crash mid-batch
    doesn't resend the ones already delivered."""
    now = now or clinic_now()
    counts = {"sent": 0, "failed": 0, "retry": 0, "expired": 0}
    with Session(engine) as s:
        due = s.exec(select(OutboxMessage).where(OutboxMessage.status == "pending")
                     .where(OutboxMessage.send_after <= now).order_by(OutboxMessage.send_after).limit(batch)).all()
        for msg in due:
            start = _appointment_start(s, msg.appointment_id)
            if msg.kind == "reminder" and start is not None and start <= now:
                msg.status = "expired"  # e.g. the server was off: a reminder after the visit is just noise
                counts["expired"] += 1
            else:
                try:
                    result = sender.send(msg.phone, msg.body)
                except Exception as e:  # noqa: BLE001 - a sender bug must not kill the dispatcher loop
                    result = SendResult(ok=False, error=f"{type(e).__name__}: {e}")
                msg.attempts += 1
                if result.ok:
                    msg.status, msg.sent_at, msg.external_id = sender.status_on_success, now, result.external_id
                    counts["sent"] += 1
                elif not result.retryable or msg.attempts >= MAX_ATTEMPTS:
                    msg.status, msg.last_error = "failed", result.error
                    counts["failed"] += 1
                else:
                    # Exponential backoff: 2, 4, 8, 16 minutes. Gives a WhatsApp outage time to recover.
                    msg.send_after, msg.last_error = now + timedelta(minutes=2 ** msg.attempts), result.error
                    counts["retry"] += 1
            s.add(msg)
            s.commit()
    return counts


async def run_dispatcher(engine, interval_s: float = DISPATCH_EVERY_S) -> None:
    sender = get_sender()
    log.info("outbox dispatcher started (provider=%s)", get_settings().messaging_provider)
    while True:
        try:
            counts = await asyncio.to_thread(dispatch_due, engine, sender)
            if any(counts.values()):
                log.info("outbox: %s", counts)
        except Exception:  # noqa: BLE001
            log.exception("outbox dispatch failed; retrying next tick")
        await asyncio.sleep(interval_s)
