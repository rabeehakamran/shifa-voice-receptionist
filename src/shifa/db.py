from datetime import datetime
from functools import lru_cache

# Naive datetimes on purpose: one clinic, all times are local Pakistan time.
from pydantic import NaiveDatetime
from sqlalchemy import JSON, Column
from sqlmodel import Field, Session, SQLModel, create_engine

from shifa.config import get_settings


class Doctor(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    title: str
    fee: int


class Slot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    doctor_id: int = Field(foreign_key="doctor.id", index=True)
    start: NaiveDatetime = Field(index=True)
    is_booked: bool = False


class Appointment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    slot_id: int = Field(foreign_key="slot.id")
    patient_name: str
    phone: str = Field(index=True)
    reason: str = ""
    status: str = "booked"  # booked | cancelled
    created_at: NaiveDatetime = Field(default_factory=datetime.now)


class Call(SQLModel, table=True):
    id: str = Field(primary_key=True)
    phone: str = Field(index=True)
    channel: str = "text"  # text | voice | sim
    started_at: NaiveDatetime = Field(default_factory=datetime.now)
    ended_at: NaiveDatetime | None = None
    escalated: bool = False
    summary: str = ""
    sentiment: str = ""
    outcome: str = ""
    tags: list = Field(default_factory=list, sa_column=Column(JSON))


class Turn(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    call_id: str = Field(foreign_key="call.id", index=True)
    role: str  # caller | agent
    text: str
    route: str = ""
    timings_ms: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: NaiveDatetime = Field(default_factory=datetime.now)


class InboxItem(SQLModel, table=True):
    """Things the AI couldn't handle: messages for staff, unanswered questions, emergency alerts."""

    id: int | None = Field(default=None, primary_key=True)
    call_id: str | None = None
    phone: str = ""
    kind: str  # message | unanswered | emergency
    text: str
    answer: str = ""
    status: str = "open"  # open | resolved
    created_at: NaiveDatetime = Field(default_factory=datetime.now)


@lru_cache
def get_engine(url: str | None = None):
    if url is None:
        path = get_settings().db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def get_session() -> Session:
    return Session(get_engine())
