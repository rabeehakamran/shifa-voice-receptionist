# Shifa Desk: architecture and tech choices

The full system on one page: every component, how a caller turn flows through it, and why each piece was chosen (with what it cost). For results and the bugs that shaped these choices, see the [README](../README.md).

## System diagram

```
                         ┌───────────────────────── CALLER ─────────────────────────┐
                         │  Browser /call page (voice)          Text CLI / chat      │
                         │  mic → Silero VAD + noise gate       (typed turns)        │
                         └──────┬──────────────────────────────────────┬─────────────┘
                     WAV over   │                                      │
                     WebSocket  ▼                                      │
                ┌──────────────────────────────┐                       │
                │ STT: Groq Whisper-large-v3   │                       │
                │ · Hindi-script → retry "ur"  │                       │
                │ · prompt-echo filter         │                       │
                └──────────────┬───────────────┘                       │
                               └───────────────┬───────────────────────┘
                                               ▼
╔══════════════════════ LangGraph: one graph invocation = one caller turn ══════════════════════╗
║                                                                                              ║
║   ┌────────────────────────────┐  emergency   ┌──────────────────────────────────────┐       ║
║   │ 1. SAFETY RULES (no LLM)   │─────────────►│ ESCALATE (no LLM)                    │       ║
║   │ regex lexicon EN/Roman/UR  │              │ fixed "call Rescue 1122" message     │       ║
║   │ + language detection       │              │ + emergency alert in staff inbox     │       ║
║   └─────────────┬──────────────┘              └──────────────────▲───────────────────┘       ║
║                 ▼                                                │ risk = emergency          ║
║   ┌────────────────────────────┐   parallel   ┌──────────────────┴───────────┐               ║
║   │ 2. ROUTER (gpt-oss-20b)    │─ ─ ─ ─ ─ ─ ─►│ Speculative RAG retrieval    │               ║
║   │ route + risk + EN rewrite  │              │ Gemini embeddings → Chroma   │               ║
║   │ + caller-name memory       │              │ (retry with rewrite if weak) │               ║
║   └──┬──────────┬──────────┬───┘              └──────────────┬───────────────┘               ║
║      │ faq      │ booking  │ handoff                         │ hits                          ║
║      ▼          ▼          ▼                                 │                               ║
║  ┌────────┐ ┌──────────────────────┐ ┌──────────────┐        │                               ║
║  │  FAQ   │◄┼──────────────────────┼─┼──────────────┼────────┘                               ║
║  │ 120b   │ │ BOOKING tool loop    │ │ HANDOFF      │                                        ║
║  │ cites  │ │ (gpt-oss-120b)       │ │ leave_message│                                        ║
║  │ [ids]  │ │ phone-bound tools:   │ └──────┬───────┘                                        ║
║  │ NO_ANS │ │  slots · book ·      │        │                                                ║
║  │ →inbox │ │  cancel* · move*     │        │   * = confirmation gate:                       ║
║  └───┬────┘ │ tool-side checks:    │        │     needs caller's "yes" in a later turn       ║
║      │      │  time==slot, real    │        │                                                ║
║      │      │  name, no 2nd booking│        │                                                ║
║      │      └──────────┬───────────┘        │                                                ║
║      ▼                 ▼                    ▼                                                ║
║   ┌────────────────────────────────────────────────────────┐                                 ║
║   │ 3. OUTPUT GUARDRAILS (code, every spoken reply)        │                                 ║
║   │ · strip leaks/artifacts · 2-sentence cap (keeps "?")   │                                 ║
║   │ · right language/script · no ids · no gendered address │                                 ║
║   │ · "booked/cancelled" claim ⇒ must have tool result     │                                 ║
║   │ · staff-callback promise ⇒ must create inbox item      │                                 ║
║   │ fail → regenerate once with reason → safe fallback     │                                 ║
║   └────────────────────────────┬───────────────────────────┘                                 ║
╚════════════════════════════════╪═════════════════════════════════════════════════════════════╝
                                 ▼
             ┌────────────────────────────────────────┐        ┌───────────────────────────┐
             │ TTS: Edge-TTS (ur-PK / en voices)      │        │ LLM layer + circuit       │
             │ streamed sentence by sentence, filler  │        │ breaker:                  │
             │ phrase after 0.8 s so line isn't silent│        │ Groq → Gemini → Ollama    │
             └────────────────────┬───────────────────┘        │ every fallback logged     │
                     MP3 over WS  ▼                            └───────────────────────────┘
                               caller

   ┌──────────────── STATE & SIDE EFFECTS ───────────────┐     ┌──────── OBSERVABILITY ────────┐
   │ SQLite (SQLModel): doctors · slots · appointments · │     │ LangSmith: 1 trace per turn,  │
   │ calls · turns · inbox                               │     │ per-node latency, tags; phone │
   │        │                                            │     │ number hashed, never sent     │
   │        ├─► Transactional OUTBOX (same commit as the │     └───────────────────────────────┘
   │        │   booking) ─► dispatcher ─► WhatsApp Cloud │
   │        │   API (or "simulate")                      │     ┌──────── OWNER DASHBOARD ──────┐
   │        └─► Inbox ─► owner answers ─► 90_learned.md  │────►│ calls · outcomes · latency ·  │
   │             ─► re-ingest (self-improving KB)        │     │ inbox · "teach Hina"          │
   └─────────────────────────────────────────────────────┘     └───────────────────────────────┘

══════════════════════════════════ OFFLINE: CALLER SIMULATOR ARENA ══════════════════════════════════
  personas.yaml (30) ─► Gemini plays caller ─► real graph on fresh in-memory DB per persona
        ─► deterministic checks (DB rows, escalation, language, forbidden text)
        ─► Gemini judge (grounded / helpful / tone + unsupported claims)
        ─► reports/latest.md + history.jsonl ─► pytest gate: emergency recall & injection = 100%
```

## Tech choices (and trade-offs)

| Choice | Why this one | What we gave up |
|---|---|---|
| **LangGraph** for the agents | A turn is a small state machine (safety → router → specialist). The graph makes the routing explicit and testable, times each node, and keeps state across turns. | More setup than one big prompt, but each agent stays small and testable on its own. |
| **Safety rules before any LLM** | An emergency must never wait on, or depend on, a model that can be down, slow or wrong. Regex plus a fixed message gives 0 ms and 0 hallucination risk. The router's `risk` field is a second layer for paraphrases. | The word list needs care. Too broad and every headache gets told "call 1122", which happened and was fixed. |
| **One router call doing 3 jobs** (route + risk + English rewrite) | Saves one LLM call per turn, and the English rewrite fixes Roman-Urdu retrieval. | The router's job is more complex, which is why the roadmap includes distilling it into a small classifier. |
| **Speculative retrieval in parallel** | Hides about 700 ms of embedding time behind the router call. | Wasted embedding calls on booking turns, which is cheap. |
| **Groq `gpt-oss-20b` / `120b`** | Free, very fast, good at tool use. The small model routes and the big one answers. | 8K tokens/minute and 1K requests/day per model, so rate limits shape the design. |
| **Groq → Gemini → Ollama fallback chain with a circuit breaker** | Free tiers fail often. A different provider keeps calls alive, and Ollama works offline. Every fallback is logged, because a silent fallback once hid a 404. | The fallback model's behaviour and quality differ from the main one. |
| **Gemini as the simulated caller and judge** | A different model family, so the agent isn't graded by its own family. | Gemini Flash-Lite's 503/504 errors cost personas, so they're reported as "not run", never as passes. |
| **Rules in code, not prompts** (tool-side checks, confirmation gate, phone-bound tools, claim verifier) | Every rule that was only in a prompt got broken in testing: instant cancels, wrong slot IDs, fake "booked". Code can't be talked out of a rule. Privacy too: there's simply no tool that lists other patients. | More code. The prompts stay short. |
| **Judge success by the DB, not by the reply** | "The agent said it booked" isn't evidence; a row in `appointments` is. That's how the Arena caught the fake bookings. | Checks have to be written per persona. |
| **Output guardrails on every reply** | Every reply is spoken aloud by TTS, so a leaked chain of thought or the wrong script would reach the caller's ear. | One regeneration adds latency in the rare bad case. |
| **Chroma + Gemini embeddings, small markdown KB, key facts pinned in the prompt** | Free, local, enough for a clinic. Fees and doctors are always in the prompt because retrieval once returned payment methods for "I can't afford Rs 2000". | Not built for a large knowledge base, which a clinic doesn't need. |
| **SQLite + SQLModel** | No server, a single file, enough for one clinic. It's easy to reset for evals (an in-memory DB per persona). | Swapping in Postgres later is easy because SQLModel works with either. |
| **Transactional outbox → WhatsApp Cloud API** | The booking and its confirmation message commit together or not at all, so no "booked but never told" and no "told but not booked". Sending runs separately and retries. | An extra dispatcher process. "Simulate" mode means Hina never promises a message that won't arrive. |
| **Browser voice over WebSocket, no telephony** | $0, and a demo works anywhere. Silero VAD runs in the browser, so silence never goes over the network. | It isn't a real phone line. Adding one (Twilio/SIP) later only changes the input and output ends. |
| **Groq Whisper-large-v3 with an Urdu guard** | Fast, free and good with Urdu. The guard fixes Urdu coming back in Hindi script, and a filter catches Whisper repeating its own prompt on silence. | It's a cloud dependency. |
| **Edge-TTS, streamed sentence by sentence, with a filler phrase** | A free, natural ur-PK voice. Streaming plus the filler means the line is never silent. | 1.6–3.8 s to first audio is the current bottleneck, versus about 1 s for paid voice agents. |
| **LangSmith with privacy by design** | One trace per turn with the latency of each node. Phone numbers are hashed before anything leaves the laptop. | Another hosted service, on its free tier. |
| **Caller Simulator Arena + a pytest gate** | Manual testing doesn't scale. Real numbers (77.8% → 96.4%) beat "it works on my demo". Every bug becomes a regression persona, and the gate fails if emergency recall or injection resistance drops below 100%. | Uses free quota; each full run is about 300 Gemini calls. |

