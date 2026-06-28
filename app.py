"""
app.py
Streamlit frontend — chat interface with feedback buttons and confidence display.
Communicates with FastAPI backend over HTTP.
"""
import streamlit as st
import requests
from config import APP_TITLE, APP_SUBTITLE, API_PORT

API_BASE = f"http://localhost:{API_PORT}"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### What can I help with?")
    st.markdown("- Company policies and leave rules")
    st.markdown("- Procurement and approval processes")
    st.markdown("- Candidate screening procedures")
    st.markdown("- Branch placement analytics")
    st.markdown("- Live weather updates")
    st.markdown("- Currency exchange rates")
    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_question = None
        st.session_state.last_answer = None
        st.session_state.feedback_given = False
        try:
            requests.post(f"{API_BASE}/clear_memory", timeout=5)
        except Exception:
            pass
        st.rerun()

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Check API health
try:
    health = requests.get(f"{API_BASE}/health", timeout=3)
    if not (health.status_code == 200 and health.json().get("assistant_ready")):
        st.warning("Assistant is initialising — please wait a moment and refresh.")
        st.stop()
except Exception:
    st.error("Backend not reachable. Start it with: `uvicorn api.main:app --reload --port 8000`")
    st.stop()

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about policies, placement data, weather, or exchange rates..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": prompt},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]

                    if data.get("injection_detected"):
                        answer = "⚠️ " + answer

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_question = prompt
                    st.session_state.last_answer = answer
                    st.session_state.feedback_given = False

                else:
                    err = f"API error: HTTP {response.status_code}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

            except requests.Timeout:
                msg = "⏱ Request timed out. The agent may be processing a complex query. Please try again."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except Exception as e:
                msg = f"❌ Could not reach the backend API. Error: {type(e).__name__}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

def _send_feedback(fb_type: str):
    """Send feedback to the API and update state."""
    try:
        requests.post(
            f"{API_BASE}/feedback",
            json={
                "question": st.session_state.last_question,
                "response": st.session_state.last_answer,
                "feedback": fb_type,
            },
            timeout=5,
        )
        st.session_state.feedback_given = True
        st.success("Thanks for your feedback!" if fb_type == "helpful" else "Feedback noted — we'll use this to improve.")
        st.rerun()
    except Exception:
        st.warning("Could not store feedback right now.")


# ── Feedback buttons ──────────────────────────────────────────────────────────
if st.session_state.last_answer and not st.session_state.feedback_given:
    st.divider()
    st.caption("Was this response helpful?")
    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if st.button("👍 Helpful"):
            _send_feedback("helpful")

    with col2:
        if st.button("👎 Not helpful"):
            _send_feedback("not_helpful")
