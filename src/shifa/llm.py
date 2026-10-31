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
                               max_retries=1, timeout=20, **extra))
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


def _chain(runnables: list[Runnable]) -> Runnable:
    if len(runnables) == 1:
        return runnables[0]

    def invoke(inp, config=None):
        err = None
        for i, r in enumerate(runnables):
            try:
                return r.invoke(inp, config)
            except Exception as e:  # noqa: BLE001 - any provider failure should fall through
                FALLBACKS[f"{i}:{type(e).__name__}"] += 1
                log.warning("LLM provider %d failed (%s: %s); trying next", i, type(e).__name__, str(e)[:200])
                err = e
        raise err

    return RunnableLambda(invoke)


def chat(tier: Tier = "smart", temperature: float = 0.2) -> Runnable:
    # Same temperature as with_tools on purpose: one cached client (one warm connection) per tier.
    return _chain(list(_models(tier, temperature)))


@lru_cache
def structured(schema, tier: Tier = "fast", temperature: float = 0.0) -> Runnable:
    return _chain([m.with_structured_output(schema) for m in _models(tier, temperature)])


def with_tools(tools, tier: Tier = "smart", temperature: float = 0.2) -> Runnable:
    # Not cached: tools are bound to the caller's phone. bind_tools is cheap; it reuses the cached clients.
    return _chain([m.bind_tools(tools) for m in _models(tier, temperature)])


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


def invoke_with_backoff(runnable: Runnable, inp, max_wait_s: float = 90, attempts: int = 4):
    """For eval-only callers: on a 429 with a short 'retry in Ns', wait and retry; long waits (daily quota) raise."""
    for attempt in range(attempts):
        try:
            return runnable.invoke(inp)
        except Exception as e:  # noqa: BLE001
            m = _RETRY_IN.search(str(e))
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else None
            if attempt == attempts - 1 or wait is None or wait > max_wait_s:
                raise
            log.warning("rate limited; waiting %.0fs", wait)
            time.sleep(wait)
