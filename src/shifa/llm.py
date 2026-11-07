"""Provider-agnostic chat models with fallbacks: Groq -> Gemini -> (optional) local Ollama.

Free tiers have rate limits; when one provider 429s or errors, the next one answers.
Structured output and tool binding must be applied per model *before* chaining fallbacks,
so we expose helpers instead of a single pre-built model.

Fallbacks are logged and counted on purpose: a silent fallback once hid a 404 from us
(wrong model id) while Gemini quietly answered everything. `FALLBACKS` feeds the eval report.
"""

import logging
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from shifa.config import get_settings

Tier = Literal["fast", "smart"]
log = logging.getLogger("shifa.llm")
FALLBACKS: Counter = Counter()  # "provider_index:ErrorType" -> count


@lru_cache
def _models(tier: Tier, temperature: float) -> tuple[BaseChatModel, ...]:
    """Cached: building a client per call meant a fresh TLS handshake to the US on every LLM call.
    Measured from Lahore: router 2262 ms -> 592 ms just by reusing clients."""
    s = get_settings()
    models: list[BaseChatModel] = []
    if s.groq_api_key:
        from langchain_groq import ChatGroq

        name = s.fast_model if tier == "fast" else s.smart_model
        # gpt-oss models reason before answering; on a phone call latency beats depth.
        extra = {"reasoning_effort": "low"} if "gpt-oss" in name else {}
        models.append(ChatGroq(model=name, api_key=s.groq_api_key, temperature=temperature,
                               max_retries=0, timeout=20, **extra))
        if tier == "smart" and s.fast_model != name:
            # Groq rate limits are per model (8K tokens/min each), so the 20B model is a fast second chance
            # before the much slower fall to Gemini. Seen in testing: 120B 429 -> Gemini turns of 11-33 s.
            models.append(ChatGroq(model=s.fast_model, api_key=s.groq_api_key, temperature=temperature,
                                   max_retries=0, timeout=20, reasoning_effort="low"))
    if s.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        models.append(ChatGoogleGenerativeAI(model=s.gemini_model, google_api_key=s.google_api_key,
                                             max_retries=1, timeout=30))
    if s.use_ollama:
        from langchain_ollama import ChatOllama

        models.append(ChatOllama(model=s.ollama_model, temperature=temperature))
    if not models:
        raise RuntimeError("No LLM configured. Put GROQ_API_KEY and/or GOOGLE_API_KEY in .env")
    return tuple(models)


class CircuitBreaker:
    """Skip a rate-limited provider until it says it's available again.

    Without this, once Groq's DAILY quota ran out, every single LLM call still asked Groq first, waited for the
    429, tried the next model, and only then reached Gemini: voice turns of 8-19 s. Now the first 429 opens the
    breaker for the time the provider asks for ("try again in 7m12s"), and calls go straight to a working one.
    """

    DEFAULT_COOLDOWN_S = 60.0
    # Groq: "Please try again in 7m12.5s". Gemini: "Please retry in 43.03s".
    _RETRY = re.compile(r"(?:try again|retry) in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.I)

    def __init__(self):
        self.open_until: dict[str, float] = {}

    def is_open(self, key: str, now: float | None = None) -> bool:
        return self.open_until.get(key, 0.0) > (now if now is not None else time.monotonic())

    def trip(self, key: str, error: Exception, now: float | None = None) -> float:
        m = self._RETRY.search(str(error))
        h, mins, secs = (m.groups() if m and any(m.groups()) else (None, None, None))
        cooldown = (int(h or 0) * 3600 + int(mins or 0) * 60 + float(secs or 0)) or self.DEFAULT_COOLDOWN_S
        self.open_until[key] = (now if now is not None else time.monotonic()) + cooldown
        return cooldown


BREAKER = CircuitBreaker()


def _is_rate_limit(e: Exception) -> bool:
    text = f"{type(e).__name__} {e}"
    return "RateLimit" in text or "429" in text or "RESOURCE_EXHAUSTED" in text


def _key(model: BaseChatModel) -> str:
    return f"{type(model).__name__}:{getattr(model, 'model_name', None) or getattr(model, 'model', '')}"


def _chain(runnables: list[Runnable], keys: list[str] | None = None) -> Runnable:
    if len(runnables) == 1:
        return runnables[0]
    keys = keys or [str(i) for i in range(len(runnables))]

    def invoke(inp, config=None):
        err = None
        order = [i for i in range(len(runnables)) if not BREAKER.is_open(keys[i])]
        if not order:  # everything is cooling down: try the one that reopens soonest rather than failing
            order = [min(range(len(runnables)), key=lambda i: BREAKER.open_until.get(keys[i], 0.0))]
        for i in order:
            try:
                return runnables[i].invoke(inp, config)
            except Exception as e:  # noqa: BLE001 - any provider failure should fall through
                FALLBACKS[f"{i}:{type(e).__name__}"] += 1
                if _is_rate_limit(e):
                    cooldown = BREAKER.trip(keys[i], e)
                    log.warning("LLM provider %s rate-limited; skipping it for %.0fs", keys[i], cooldown)
                else:
                    log.warning("LLM provider %s failed (%s: %s); trying next", keys[i], type(e).__name__,
                                str(e)[:200])
                err = e
        raise err

    return RunnableLambda(invoke)


def chat(tier: Tier = "smart", temperature: float = 0.2) -> Runnable:
    # Same temperature as with_tools on purpose: one cached client (one warm connection) per tier.
    models = _models(tier, temperature)
    return _chain(list(models), [_key(m) for m in models])


@lru_cache
def structured(schema, tier: Tier = "fast", temperature: float = 0.0) -> Runnable:
    models = _models(tier, temperature)
    return _chain([m.with_structured_output(schema) for m in models], [_key(m) for m in models])


def with_tools(tools, tier: Tier = "smart", temperature: float = 0.2) -> Runnable:
    # Not cached: tools are bound to the caller's phone. bind_tools is cheap; it reuses the cached clients.
    models = _models(tier, temperature)
    return _chain([m.bind_tools(tools) for m in models], [_key(m) for m in models])


def warmup() -> None:
    """Open TLS connections before the first caller: a cold first turn measured 6.8 s vs ~2 s warm.
    Runs in a background thread at app start; failures are harmless."""
    import threading

    def run():
        from shifa.rag import retriever

        for fn in (lambda: _models("fast", 0.0)[0].invoke("hi", max_tokens=1),
                   lambda: _models("smart", 0.2)[0].invoke("hi", max_tokens=1),
                   lambda: retriever.embed(["warmup"], "query")):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.info("warmup step failed: %s", e)

    threading.Thread(target=run, daemon=True).start()


@lru_cache
def gemini(model: str | None = None) -> BaseChatModel:
    """A different model family from the agent: used by the eval simulator and judge to reduce self-bias."""
    s = get_settings()
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model or s.gemini_model, google_api_key=s.google_api_key, max_retries=0)


_RETRY_IN = re.compile(r"(?:retry|try again) in (?:(\d+)m)?([\d.]+)s", re.I)


_TRANSIENT = re.compile(r"(?:500|502|503|504)|UNAVAILABLE|DEADLINE_EXCEEDED|INTERNAL")


def invoke_with_backoff(runnable: Runnable, inp, max_wait_s: float = 90, attempts: int = 4):
    """For eval-only callers: on a 429 with a short 'retry in Ns', wait and retry; long waits (daily quota) raise.
    Transient server errors (503 high demand, 504 deadline) are retried too: they cost a v5 Arena run 6 personas."""
    for attempt in range(attempts):
        try:
            return runnable.invoke(inp)
        except Exception as e:  # noqa: BLE001
            m = _RETRY_IN.search(str(e))
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else None
            if wait is None and _TRANSIENT.search(str(e)):
                wait = 15.0 * (attempt + 1)
            if attempt == attempts - 1 or wait is None or wait > max_wait_s:
                raise
            log.warning("rate limited; waiting %.0fs", wait)
            time.sleep(wait)
