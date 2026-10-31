"""Create doctors and two weeks of 20-minute slots. Run: python -m shifa.seed"""

from datetime import date, datetime, time, timedelta

from sqlmodel import Session, SQLModel, delete

from shifa.config import clinic_today
from shifa.db import Appointment, Doctor, Slot, get_engine

SLOT_MINUTES = 20

# name, title, fee, weekdays (Mon=0), shift start, shift end
DOCTORS = [
    ("Dr. Ayesha Malik", "MBBS, FCPS (Family Medicine)", 2000, range(0, 6), time(9), time(13)),
    ("Dr. Usman Tariq", "MBBS, MCPS", 1500, range(0, 6), time(17), time(22)),
    ("Dr. Sana Iqbal", "MBBS, FCPS (Paediatrics), visiting", 2500, [5], time(10), time(14)),
]


def seed(engine=None, start: date | None = None, days: int = 14) -> None:
    engine = engine or get_engine()
    start = start or clinic_today()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        for model in (Appointment, Slot, Doctor):
            s.exec(delete(model))
        for name, title, fee, weekdays, t0, t1 in DOCTORS:
            doc = Doctor(name=name, title=title, fee=fee)
            s.add(doc)
            s.flush()
            for d in range(days):
                day = start + timedelta(days=d)
                if day.weekday() not in weekdays:
                    continue
                t = datetime.combine(day, t0)
                while t < datetime.combine(day, t1):
                    s.add(Slot(doctor_id=doc.id, start=t))
                    t += timedelta(minutes=SLOT_MINUTES)
        s.commit()


if __name__ == "__main__":
    seed()
    print(f"Seeded doctors and 14 days of slots from {clinic_today()}")
