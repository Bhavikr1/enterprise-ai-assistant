"""
config.py
All thresholds, constants, and environment variables in one place.
Business rules belong in code — not buried in prompts.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
WEATHER_API_KEY  = os.getenv("WEATHER_API_KEY", "")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY", "")   # optional, some endpoints are free

# ── LLM Config ────────────────────────────────────────────────────────────────
LLM_MODEL        = "gemini-2.5-flash"
EMBEDDING_MODEL  = "models/gemini-embedding-2"
LLM_TEMPERATURE  = 0.0      # deterministic — we want grounded answers, not creative ones
LLM_MAX_TOKENS   = 1024

# ── RAG Config ────────────────────────────────────────────────────────────────
CHUNK_SIZE       = 512      # tokens — fits one complete policy clause or SOP step
CHUNK_OVERLAP    = 64       # ~12.5% — prevents boundary fragmentation
TOP_K            = 4        # retrieve top 4 chunks — enough for complex questions, avoids noise

# ── Reliability / Confidence ──────────────────────────────────────────────────
# confidence = 1 - (cosine_distance / 2)  → normalised 0-1
CONFIDENCE_HIGH    = 0.80   # full answer + citations
CONFIDENCE_MEDIUM  = 0.65   # answer with uncertainty warning
# below CONFIDENCE_MEDIUM → hard refusal, no LLM call

# ── Injection guard keywords ──────────────────────────────────────────────────
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your instructions",
    "reveal your system prompt",
    "show me your prompt",
    "you are now",
    "act as",
    "jailbreak",
    "dan mode",
]

# ── Memory Config ─────────────────────────────────────────────────────────────
MEMORY_MAX_TOKEN_LIMIT = 2000   # recent turns kept verbatim up to this limit

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR    = "./data/chroma_db"
CHROMA_COLLECTION     = "enterprise_docs"

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR    = "./data/documents"
CSV_PATH         = "./data/placement_data.csv"
FEEDBACK_DB_PATH = "./data/feedback.db"

# ── API / Retry Config ────────────────────────────────────────────────────────
REQUEST_TIMEOUT      = 10     # seconds before timeout on external API calls
MAX_RETRIES          = 2      # retry count for external tool calls
RETRY_BACKOFF        = 2      # seconds — doubles on each retry (exponential backoff)

# ── External APIs ─────────────────────────────────────────────────────────────
WEATHER_BASE_URL  = "https://api.openweathermap.org/data/2.5/weather"
CURRENCY_BASE_URL = "https://api.exchangerate-api.com/v4/latest"    # free tier, no key needed

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE    = "AI Enterprise Assistant"
APP_SUBTITLE = "Ask from documents, data, or live tools"
