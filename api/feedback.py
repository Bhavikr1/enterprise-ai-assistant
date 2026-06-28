"""
api/feedback.py
SQLite feedback store — question, response, feedback, confidence, timestamp.
This is preference logging — NOT RLHF.
RLHF requires a reward model + PPO training loop that modifies model weights.
This collects the signal that COULD feed into RLHF later.
"""
import sqlite3
import json
from datetime import datetime
from config import FEEDBACK_DB_PATH


def init_db():
    """Create feedback table if it doesn't exist."""
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            question        TEXT    NOT NULL,
            response        TEXT    NOT NULL,
            confidence_score REAL   DEFAULT NULL,
            tool_used       TEXT    DEFAULT NULL,
            feedback        TEXT    NOT NULL,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def store_feedback(
    question: str,
    response: str,
    feedback: str,           # 'helpful' or 'not_helpful'
    confidence_score: float = None,
    tool_used: str = None,
) -> bool:
    """
    Store a feedback record in SQLite.
    Returns True on success, False on failure.
    """
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (question, response, confidence_score, tool_used, feedback, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            question[:2000],        # truncate long questions
            response[:4000],        # truncate long responses
            confidence_score,
            tool_used,
            feedback,
            datetime.utcnow().isoformat(),
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Feedback storage error: {e}")
        return False


def get_feedback_stats() -> dict:
    """
    Return aggregate feedback statistics.
    Useful for demonstrating the feedback loop during the walkthrough.
    """
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback = 'helpful'")
        helpful = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback = 'not_helpful'")
        not_helpful = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence_score) FROM feedback WHERE feedback = 'helpful'")
        avg_conf_helpful = cursor.fetchone()[0]

        cursor.execute("SELECT tool_used, COUNT(*) FROM feedback GROUP BY tool_used")
        tool_breakdown = dict(cursor.fetchall())

        conn.close()

        return {
            "total": total,
            "helpful": helpful,
            "not_helpful": not_helpful,
            "helpful_rate": f"{(helpful/total*100):.1f}%" if total > 0 else "N/A",
            "avg_confidence_on_helpful": round(avg_conf_helpful or 0, 3),
            "tool_breakdown": tool_breakdown,
        }
    except Exception as e:
        return {"error": str(e)}


def get_recent_feedback(limit: int = 10) -> list:
    """Return the most recent N feedback records."""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT question, feedback, confidence_score, tool_used, timestamp
            FROM feedback
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "question": r[0],
                "feedback": r[1],
                "confidence": r[2],
                "tool": r[3],
                "timestamp": r[4],
            }
            for r in rows
        ]
    except Exception as e:
        return []


# Initialise DB on import
init_db()
