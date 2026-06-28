"""
core/reliability.py
Business rules for reliability — confidence gating, injection detection, refusal.
These are deterministic rules in code — NOT in the LLM prompt.
Business rules belong here. AI reasoning belongs in prompts.
"""
import re
from config import (
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
    INJECTION_PATTERNS
)
from core.prompts import (
    REFUSAL_LOW_CONFIDENCE, REFUSAL_INJECTION, UNCERTAINTY_WARNING
)


def compute_confidence(distance: float) -> float:
    """
    Convert ChromaDB cosine distance to a confidence score (0-1).
    ChromaDB cosine distance ranges 0 (identical) to 2 (maximally different).
    confidence = 1 - (distance / 2) normalises to 0-1.
    """
    return round(max(0.0, min(1.0, 1.0 - (distance / 2.0))), 4)


def get_confidence_label(score: float) -> str:
    """Human-readable confidence label for UI display."""
    if score >= CONFIDENCE_HIGH:
        return f"High ({score:.0%})"
    elif score >= CONFIDENCE_MEDIUM:
        return f"Moderate ({score:.0%})"
    else:
        return f"Low ({score:.0%})"


def should_refuse(best_confidence: float) -> bool:
    """
    Hard refusal gate — if best chunk confidence is below threshold,
    do NOT call the LLM. Return True to trigger refusal.
    """
    return best_confidence < CONFIDENCE_MEDIUM


def get_refusal_message(confidence: float) -> str:
    """Return the appropriate refusal or warning message."""
    return REFUSAL_LOW_CONFIDENCE


def maybe_add_uncertainty_warning(answer: str, confidence: float) -> str:
    """
    If confidence is moderate (between thresholds), append a warning.
    Only called when we DID answer (confidence above refusal threshold).
    """
    if CONFIDENCE_MEDIUM <= confidence < CONFIDENCE_HIGH:
        return answer + UNCERTAINTY_WARNING
    return answer


def check_injection(user_input: str) -> bool:
    """
    Scan user input for known prompt injection patterns.
    Returns True if injection attempt is detected.
    Case-insensitive matching.
    """
    text = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern.lower() in text:
            return True
    return False


def get_injection_response() -> str:
    """Return the injection detection response."""
    return REFUSAL_INJECTION


def sanitise_input(user_input: str) -> str:
    """
    Light sanitisation — strip excessive whitespace.
    We do NOT strip content aggressively; the injection guard handles bad intent.
    """
    return re.sub(r'\s+', ' ', user_input.strip())
