from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# pydantic-settings reads .env into Settings only; libraries like LangSmith read os.environ. Without this,
# LANGSMITH_TRACING=true in .env silently traced nothing. override=False: real env vars (and tests) win.
load_dotenv(ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="SHIFA_", extra="ignore")

    groq_api_key: str = ""
    google_api_key: str = ""

    # Two tiers: "fast" for routing/classification, "smart" for agents that talk and call tools.
    # Check what your key can use with Groq().models.list(); the public docs list more than accounts get.
    fast_model: str = "openai/gpt-oss-20b"
    smart_model: str = "openai/gpt-oss-120b"
    # Free-tier Gemini quotas are per model (3.6-flash: 5 RPM / 20 per day), so spread the load:
    gemini_model: str = "gemini-3.1-flash-lite"  # agent fallback (rarely used)
    sim_model: str = "gemini-3.5-flash-lite"  # eval caller simulator
    judge_model: str = "gemini-3.1-flash-lite"  # eval judge
    use_ollama: bool = False
    ollama_model: str = "qwen2.5:3b"

    db_path: Path = DATA_DIR / "shifa.db"
    checkpoint_path: Path = DATA_DIR / "checkpoints.db"
    chroma_path: Path = DATA_DIR / "chroma"
    kb_dir: Path = DATA_DIR / "clinic_kb"
    embedding_model: str = "gemini-embedding-2"

    # Pinning "today" makes evals reproducible; empty means the real date.
    today: date | None = None

    # Patient messaging. "simulate": the outbox fills and the dispatcher marks messages "simulated" (visible on the
    # dashboard, nothing leaves the laptop). "whatsapp": sent for real via the WhatsApp Cloud API.
    # Hina only tells callers about messages when a real provider is configured.
    messaging_provider: str = "simulate"
    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_api_version: str = "v23.0"
    whatsapp_mode: str = "text"  # text (inside the 24 h window, for testing) | template (approved template)
    whatsapp_template_name: str = "appointment_update"
    whatsapp_template_lang: str = "en"


class Keys(BaseSettings):
    """API keys use their standard un-prefixed env names."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    groq_api_key: str = ""
    google_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    keys = Keys()
    s.groq_api_key = s.groq_api_key or keys.groq_api_key
    s.google_api_key = s.google_api_key or keys.google_api_key
    return s


def clinic_today() -> date:
    return get_settings().today or date.today()


def clinic_now() -> datetime:
    """With a pinned date, pretend it's 8 AM (before opening) so every slot that day is bookable."""
    pinned = get_settings().today
    return datetime.combine(pinned, time(8)) if pinned else datetime.now()
