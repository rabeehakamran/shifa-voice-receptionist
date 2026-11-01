import os
from datetime import date

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

os.environ["SHIFA_TODAY"] = "2026-11-02"  # a Monday; must be set before settings are first read
os.environ["LANGSMITH_TRACING"] = "false"  # unit tests never send traces, even if .env enables tracing
os.environ["SHIFA_MESSAGING_PROVIDER"] = "simulate"  # ...and never message anyone

from shifa.config import get_settings  # noqa: E402
from shifa.seed import seed  # noqa: E402

get_settings.cache_clear()
MONDAY = date(2026, 11, 2)


@pytest.fixture(autouse=True)
def no_network_retrieval(monkeypatch):
    """The router retrieves speculatively; unit tests must never hit the embedding API."""
    from shifa.rag import retriever

    monkeypatch.setattr(retriever, "search", lambda q, **k: [])


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    seed(eng, start=MONDAY)
    return eng
