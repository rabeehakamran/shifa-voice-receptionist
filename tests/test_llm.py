"""Provider chain + circuit breaker, with fake models (no API calls)."""

from langchain_core.runnables import RunnableLambda

from shifa import llm


class RateLimitError(Exception):
    pass


def test_breaker_parses_provider_retry_hints():
    b = llm.CircuitBreaker()
    # Shapes of real Groq / Gemini messages.
    assert b.trip("groq", RateLimitError("... tokens per day (TPD). Please try again in 7m12.5s."), now=0) == 432.5
    assert b.trip("gemini", RateLimitError("429 RESOURCE_EXHAUSTED. Please retry in 43.03s."), now=0) == 43.03
    assert b.trip("x", RateLimitError("Please try again in 1h2m3s"), now=0) == 3723
    assert b.trip("y", RateLimitError("429 slow down"), now=0) == llm.CircuitBreaker.DEFAULT_COOLDOWN_S


def test_rate_limited_provider_is_skipped_on_later_calls(monkeypatch):
    monkeypatch.setattr(llm, "BREAKER", llm.CircuitBreaker())
    calls = []

    def groq(_):
        calls.append("groq")
        raise RateLimitError("Rate limit reached (TPD). Please try again in 10m0s.")

    def gemini(_):
        calls.append("gemini")
        return "ok"

    chain = llm._chain([RunnableLambda(groq), RunnableLambda(gemini)], ["groq", "gemini"])
    assert chain.invoke("q") == "ok" and calls == ["groq", "gemini"]
    calls.clear()
    assert chain.invoke("q") == "ok" and calls == ["gemini"]  # no wasted round trip to the exhausted provider


def test_non_rate_limit_errors_do_not_open_the_breaker(monkeypatch):
    monkeypatch.setattr(llm, "BREAKER", llm.CircuitBreaker())
    calls = []

    def flaky(_):
        calls.append("flaky")
        raise TimeoutError("read timeout")

    chain = llm._chain([RunnableLambda(flaky), RunnableLambda(lambda _: "ok")], ["flaky", "backup"])
    chain.invoke("q")
    chain.invoke("q")
    assert calls == ["flaky", "flaky"]  # a timeout is transient: keep trying the preferred provider


def test_all_open_still_tries_the_soonest(monkeypatch):
    b = llm.CircuitBreaker()
    monkeypatch.setattr(llm, "BREAKER", b)
    b.open_until.update({"a": 1e12, "b": 1e11})
    chain = llm._chain([RunnableLambda(lambda _: "A"), RunnableLambda(lambda _: "B")], ["a", "b"])
    assert chain.invoke("q") == "B"
