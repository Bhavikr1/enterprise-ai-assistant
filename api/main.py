"""
api/main.py
FastAPI backend — clean separation between AI logic and HTTP layer.
The agent lives in core/. This file only handles HTTP concerns.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from core.agent import get_assistant
from core.reliability import check_injection, get_injection_response, sanitise_input
from api.feedback import store_feedback, get_feedback_stats, get_recent_feedback
from config import API_HOST, API_PORT

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan handler — replaces deprecated @app.on_event("startup").
    Runs blocking initialisation in a thread-pool so the event loop stays free.
    """
    logger.info("Server starting — initialising assistant...")
    assistant = get_assistant()
    if not assistant._initialised:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, assistant.initialise)
    logger.info("Assistant ready — accepting requests.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="AI Enterprise Assistant",
    description="ReAct agent with RAG, structured data, tool calling, and feedback loop.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    answer: str
    success: bool
    injection_detected: bool = False

class FeedbackRequest(BaseModel):
    question: str
    response: str
    feedback: str                     # 'helpful' or 'not_helpful'
    confidence_score: Optional[float] = None
    tool_used: Optional[str] = None

class FeedbackResponse(BaseModel):
    stored: bool
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — used by Streamlit to verify the service is up."""
    assistant = get_assistant()
    return {
        "status": "ok",
        "assistant_ready": assistant._initialised,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Injection check → sanitise → agent (in thread pool) → return answer.
    Agent.run() is synchronous/blocking; run_in_threadpool keeps the event loop free.
    """
    # Step 1: Injection guard (deterministic business rule — not in prompt)
    if check_injection(request.question):
        logger.warning("Injection attempt detected: %s", request.question[:80])
        return ChatResponse(
            answer=get_injection_response(),
            success=False,
            injection_detected=True,
        )

    # Step 2: Sanitise input
    clean_input = sanitise_input(request.question)

    # Step 3: Guard — assistant must be ready
    assistant = get_assistant()
    if not assistant._initialised:
        raise HTTPException(status_code=503, detail="Assistant is initialising. Please try again shortly.")

    # Step 4: Run through agent — offload blocking call to thread pool
    logger.info("Processing question: %s", clean_input[:80])
    result = await run_in_threadpool(assistant.run, clean_input)

    if not result["success"]:
        logger.warning("Agent returned error: %s", result.get("error"))

    return ChatResponse(
        answer=result["answer"],
        success=result["success"],
        injection_detected=False,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """
    Store user feedback for a question-answer pair.
    Preference logging — not RLHF (no model weights modified here).
    """
    if request.feedback not in ("helpful", "not_helpful"):
        raise HTTPException(status_code=400, detail="feedback must be 'helpful' or 'not_helpful'")

    stored = await run_in_threadpool(
        store_feedback,
        request.question,
        request.response,
        request.feedback,
        request.confidence_score,
        request.tool_used,
    )

    return FeedbackResponse(
        stored=stored,
        message="Feedback recorded." if stored else "Failed to store feedback.",
    )


@app.get("/feedback/stats")
async def feedback_stats():
    """Return aggregate feedback statistics."""
    return await run_in_threadpool(get_feedback_stats)


@app.get("/feedback/recent")
async def recent_feedback(limit: int = 10):
    """Return recent feedback records."""
    return await run_in_threadpool(get_recent_feedback, limit)


@app.post("/clear_memory")
async def clear_memory():
    """Reset conversation memory — called when user starts a new session."""
    assistant = get_assistant()
    if assistant._initialised:
        await run_in_threadpool(assistant.clear_memory)
    return {"cleared": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
