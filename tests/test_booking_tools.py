import re

import pytest
from sqlmodel import Session

from shifa.tools import booking_tools as bt

ME, OTHER = "03001112222", "03339998888"


def first_slot_id(text: str) -> int:
    return int(re.search(r"slot_id=(\d+)", text).group(1))


def test_sunday_closed(engine):
    with Session(engine) as s:
        assert "No free slots" in bt.get_available_slots(s, "2026-11-08")


def test_part_of_day_and_doctor_filter(engine):
    with Session(engine) as s:
        out = bt.get_available_slots(s, "2026-11-02", doctor="usman", part_of_day="evening")
        assert "Dr. Usman Tariq" in out and "Ayesha" not in out
        assert "05:00 PM" in out


def test_bad_date_is_actionable_error(engine):
    with Session(engine) as s, pytest.raises(bt.BookingError, match="YYYY-MM-DD"):
        bt.get_available_slots(s, "kal")


def test_book_then_slot_disappears(engine):
    with Session(engine) as s:
        sid = first_slot_id(bt.get_available_slots(s, "2026-11-02", doctor="ayesha"))
        assert "BOOKED" in bt.book_appointment(s, ME, sid, "Ali Raza", "fever")
        assert f"slot_id={sid} " not in bt.get_available_slots(s, "2026-11-02", doctor="ayesha")
        with pytest.raises(bt.BookingError, match="already booked"):
            bt.book_appointment(s, OTHER, sid, "Someone")


def test_double_booking_same_patient_same_day_rejected(engine):
    with Session(engine) as s:
        slots = re.findall(r"slot_id=(\d+)", bt.get_available_slots(s, "2026-11-02", doctor="ayesha"))
        bt.book_appointment(s, ME, int(slots[0]), "Ali Raza")
        with pytest.raises(bt.BookingError, match="reschedule"):
            bt.book_appointment(s, ME, int(slots[1]), "ali raza")
        # A different family member on the same phone is fine.
        assert "BOOKED" in bt.book_appointment(s, ME, int(slots[1]), "Fatima Raza")


def test_privacy_other_callers_appointments_invisible(engine):
    with Session(engine) as s:
        sid = first_slot_id(bt.get_available_slots(s, "2026-11-03"))
        booked = bt.book_appointment(s, OTHER, sid, "Secret Patient")
        appt_id = int(re.search(r"appointment_id=(\d+)", booked).group(1))
        assert "Secret Patient" not in bt.my_appointments(s, ME)
        with pytest.raises(bt.BookingError, match="No active appointment"):
            bt.cancel_appointment(s, ME, appt_id)


def test_reschedule_frees_old_slot(engine):
    with Session(engine) as s:
        slots = [int(x) for x in re.findall(r"slot_id=(\d+)", bt.get_available_slots(s, "2026-11-04"))]
        appt_id = int(re.search(r"appointment_id=(\d+)", bt.book_appointment(s, ME, slots[0], "Ali")).group(1))
        assert "RESCHEDULED" in bt.reschedule_appointment(s, ME, appt_id, slots[1])
        assert f"slot_id={slots[0]} " in bt.get_available_slots(s, "2026-11-04")


def test_tools_bind_phone(engine):
    tools = {t.name: t for t in bt.make_tools(engine, ME)["booking"]}
    assert "phone" not in tools["book_appointment"].args
    sid = first_slot_id(tools["get_available_slots"].invoke({"date": "2026-11-05"}))
    tools["book_appointment"].invoke({"slot_id": sid, "patient_name": "Ali"})
    assert "Ali" in tools["my_appointments"].invoke({})
