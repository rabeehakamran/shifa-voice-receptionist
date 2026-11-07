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
    tools["book_appointment"].invoke({"slot_id": sid, "time_hhmm": _hhmm(engine, sid), "patient_name": "Ali"})
    assert "Ali" in tools["my_appointments"].invoke({})


def _hhmm(engine, sid: int) -> str:
    from shifa.db import Slot

    with Session(engine) as s:
        return f"{s.get(Slot, sid).start:%H:%M}"


def test_book_tool_rejects_time_that_does_not_match_slot(engine):
    """Arena: caller chose 9:00, the model sent the 10:00 slot_id."""
    book = {t.name: t for t in bt.make_tools(engine, ME)["booking"]}["book_appointment"]
    sid = first_slot_id(bt.get_available_slots(Session(engine), "2026-11-05"))
    with pytest.raises(bt.BookingError, match="not 23:00"):
        book.invoke({"slot_id": sid, "time_hhmm": "23:00", "patient_name": "Ali"})
    assert "BOOKED" in book.invoke({"slot_id": sid, "time_hhmm": _hhmm(engine, sid), "patient_name": "Ali"})


def test_placeholder_patient_name_rejected(engine):
    with Session(engine) as s:
        sid = first_slot_id(bt.get_available_slots(s, "2026-11-05"))
        with pytest.raises(bt.BookingError, match="patient's name"):
            bt.book_appointment(s, ME, sid, "UNKNOWN")


def test_book_with_same_doctor_points_to_reschedule(engine):
    """Arena "reschedule": 'shift my appointment to Thursday' booked a second one and kept the old one."""
    book = {t.name: t for t in bt.make_tools(engine, ME)["booking"]}["book_appointment"]
    with Session(engine) as s:
        bt.book_appointment(s, ME, first_slot_id(bt.get_available_slots(s, "2026-11-03", "Ayesha")), "Sadia")
        thursday = first_slot_id(bt.get_available_slots(s, "2026-11-05", "Ayesha"))
    args = {"slot_id": thursday, "time_hhmm": _hhmm(engine, thursday), "patient_name": "Sadia"}
    with pytest.raises(bt.BookingError, match="reschedule_appointment"):
        book.invoke(args)
    assert "BOOKED" in book.invoke({**args, "keep_existing": True})


def test_cancel_needs_confirmation_in_a_later_turn(engine):
    """Regression: 'cancel my appointment' once cancelled instantly without saying which one."""
    with Session(engine) as s:
        sid = first_slot_id(bt.get_available_slots(s, "2026-11-04"))
        appt_id = int(re.search(r"appointment_id=(\d+)", bt.book_appointment(s, ME, sid, "Ali")).group(1))
    turn1 = {t.name: t for t in bt.make_tools(engine, ME, call_id="c1", turn=1)["booking"]}
    out = turn1["cancel_appointment"].invoke({"appointment_id": appt_id, "confirmed": True})
    assert out.startswith("CONFIRMATION NEEDED") and "Ali" in out  # same-turn "confirmed" is refused
    assert "Ali" in turn1["my_appointments"].invoke({})  # nothing changed
    turn2 = {t.name: t for t in bt.make_tools(engine, ME, call_id="c1", turn=2)["booking"]}
    assert turn2["cancel_appointment"].invoke({"appointment_id": appt_id, "confirmed": True}).startswith("CANCELLED")


def test_around_time_finds_late_evening_slots(engine):
    """Regression: only the first 6 slots were shown, so 7 PM looked unavailable."""
    with Session(engine) as s:
        out = bt.get_available_slots(s, "2026-11-07", doctor="usman", around_time="19:00")
    assert "07:00 PM" in out and "to 09:40 PM" in out


def test_confirmation_accepted_when_hina_asked_in_her_own_words(engine):
    """Regression: the model asked "kya aap cancel karna chahenge?" without calling the tool, then the caller's
    "ji haan" got asked again."""
    with Session(engine) as s:
        sid = first_slot_id(bt.get_available_slots(s, "2026-11-04"))
        appt_id = int(re.search(r"appointment_id=(\d+)", bt.book_appointment(s, ME, sid, "Ali")).group(1))
    tools = {t.name: t for t in bt.make_tools(engine, ME, call_id="c2", turn=3, asked_last_turn=True)["booking"]}
    assert tools["cancel_appointment"].invoke({"appointment_id": appt_id, "confirmed": True}).startswith("CANCELLED")
