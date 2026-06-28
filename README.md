# AI Enterprise Assistant

An AI assistant that answers questions from company policy documents, structured placement data, and live external APIs through a single chat interface.

---

## Quick Start

```bash
# 1. Enter the project
cd ai_assistant

# 2. Set up environment
cp .env.example .env
# Add your GEMINI_API_KEY and WEATHER_API_KEY to .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the backend (Terminal 1)
uvicorn api.main:app --reload --port 8000

# 5. Start the frontend (Terminal 2)
streamlit run app.py
```

> **API keys needed:**
> - Gemini: https://aistudio.google.com/app/apikey (free)
> - OpenWeatherMap: https://openweathermap.org/api (free tier)
> - Currency: no key needed (free public endpoint)

---

## Project Structure

```
ai_assistant/
├── core/
│   ├── agent.py          # ReAct agent assembly
│   ├── memory.py         # Conversation memory
│   ├── prompts.py        # System prompt and tool descriptions
│   └── reliability.py    # Confidence gate and injection guard
├── tools/
│   ├── rag_tool.py       # Document retrieval
│   ├── csv_tool.py       # Placement data analytics
│   ├── weather_tool.py   # Live weather API
│   └── currency_tool.py  # Live currency API
├── rag/
│   ├── ingest.py         # Document ingestion pipeline
│   └── retriever.py      # ChromaDB query
├── api/
│   ├── main.py           # FastAPI endpoints
│   └── feedback.py       # Feedback storage
├── data/
│   ├── documents/        # leave_policy.txt, procurement_sop.txt, candidate_screening_sop.txt
│   ├── placement_data.csv
│   └── feedback.db       # Auto-created on first run
├── app.py                # Streamlit frontend
├── config.py             # Constants and environment variables
└── requirements.txt
```

---

## Architecture

```
User
 │
 ▼
Streamlit UI
 │  HTTP POST /chat
 ▼
FastAPI Backend  (injection guard → agent)
 │
 ▼
ReAct Agent  (Reason → select tool → act → observe → answer)
 │
 ├──▶ rag_retriever    → ChromaDB → confidence gate → Gemini Flash
 ├──▶ csv_analyst      → Pandas DataFrame Agent
 ├──▶ weather_tool     → OpenWeatherMap API
 └──▶ currency_tool    → ExchangeRate API
          │
          ▼
       SQLite  (feedback store)
```

---

## Capabilities

| Part | What it does |
|------|-------------|
| Document Assistant | Answers questions from Leave Policy, Procurement SOP, and Candidate Screening SOP with source citations |
| Structured Data | Computes answers from placement CSV — no hardcoded responses |
| Tool Calling | Live weather and currency data from external APIs |
| Reliability | Confidence threshold (hard refusal below 0.65) + prompt injection guard |
| Feedback Loop | Helpful / Not Helpful buttons, stored in SQLite |

---

## Chunking Strategy

| Parameter | Value | Why |
|-----------|-------|-----|
| Chunk size | 512 characters | Fits one complete policy clause or SOP step. Smaller loses context within a step; larger groups unrelated clauses together. |
| Overlap | 64 characters | Prevents sentences that straddle a boundary from being cut — the content appears complete in at least one chunk. |
| Splitter | RecursiveCharacterTextSplitter | Splits on paragraph → sentence → word order. Never cuts mid-sentence unless a single paragraph exceeds 512 characters. |

---

## Retrieval

| Parameter | Value | Why |
|-----------|-------|-----|
| Embedding model | gemini-embedding-2 | 3072-dimensional dense vectors. Same API key as the LLM — one credential to manage. |
| Top-K | 4 chunks | Enough context for multi-part questions. Above 6 dilutes the LLM's attention with irrelevant content. |
| Distance metric | Cosine | Bounded [0, 2] range maps cleanly to `confidence = 1 - (distance / 2)`. L2 distance is unbounded and breaks this formula. |

Confidence behaviour:

| Score | Response |
|-------|----------|
| ≥ 0.80 | Full answer with source citations |
| 0.65 – 0.79 | Answer with uncertainty warning |
| < 0.65 | Hard refusal — no LLM call made |

---

## Structured Data

LangChain's `create_pandas_dataframe_agent` generates and executes Pandas operations dynamically from natural language. All answers are computed at query time — nothing is hardcoded.

```
Question: "Which branch has highest placement rate?"
Generated: df.loc[df['Placement_Rate'].idxmax()]
Answer:    Bangalore — 90.0%
```

---

## Tool Calling

Each tool has a precise description that the LLM reads to decide which tool to call. No hardcoded routing — the agent selects semantically based on the question.

| Question type | Tool selected |
|--------------|--------------|
| Policy, leave, SOP, approval | `rag_retriever` |
| Branch, placement, average | `csv_analyst` |
| Weather, temperature | `weather_tool` |
| Exchange rate, currency | `currency_tool` |

Failure handling: all external API tools retry twice with exponential backoff (2s, 4s) on timeout or rate limit. Users never see a raw error — they get a clean message.

**On MCP:** The assignment lists MCP as preferred. MCP was not implemented here — building a proper MCP server (stdio or HTTP transport, schema definition, lifecycle handling, client wiring) is a 2–3 day task on its own, which falls outside a 3-hour constraint. LangChain's `@tool` abstraction achieves the same agent-decides-when-to-call behavior within the available time. A full MCP server is listed under Future Improvements.

---

## Feedback Loop

Helpful / Not Helpful feedback is stored in SQLite with the question, response, confidence score, tool used, and timestamp.

**Is this RLHF?** No. RLHF requires a reward model trained on preference comparisons, a policy model whose weights are updated via PPO, and actual fine-tuning loops. None of that happens here. This is preference logging — the data collection step that would precede RLHF. No model weights are modified.

How this data could be used later:
- Train a reward model on good vs bad response pairs
- Calibrate the confidence threshold by plotting score against feedback labels
- Identify topics the system consistently handles poorly

---

## Limitations

1. Dense-only retrieval — no BM25 hybrid search
2. No cross-encoder reranking after retrieval
3. Fixed chunk size across all document types
4. No document update pipeline — re-ingestion is manual
5. SQLite not suitable for concurrent writes at scale
6. English only

---

## Future Improvements

1. Hybrid search — BM25 + dense with Reciprocal Rank Fusion
2. Streaming responses — show reasoning trace in real time
3. Cross-encoder reranking
4. RAGAS evaluation harness
5. Full MCP server with stdio transport
6. Document versioning
7. Redis memory store

---

## Evaluation

Eight questions run against the live system.

| # | Category | Question | Expected | Actual | Pass/Fail |
|---|----------|----------|----------|--------|-----------|
| 1 | Document | What is the maximum number of casual leaves per year? | 12 casual leaves, cannot be carried forward | 12 casual leaves per year, cannot be carried forward. [Source: leave_policy.txt, Section 2.1] Confidence: 91% | ✅ Pass |
| 2 | Document | Who approves procurement above Rs. 1,00,000? | CFO and MD jointly | CFO and MD jointly approve. [Source: procurement_sop.txt, Section 4.4] Confidence: 91% | ✅ Pass |
| 3 | Document | What happens if a shortlisted candidate doesn't respond in 48 hours? | Recruiter sends reminder, then contacts next waitlisted candidate | Recruiter sends one reminder, marks non-responsive, contacts next waitlisted candidate. [Source: candidate_screening_sop.txt, Section 6.2] Confidence: 91% | ✅ Pass |
| 4 | Document | Can an employee take earned leave during probation? | No — leave accrues but cannot be availed during probation | No. Leave accrues but cannot be availed until probation is complete. [Source: leave_policy.txt, Section 3.4] Confidence: 92% | ✅ Pass |
| 5 | Structured Data | Which branch has the highest placement rate? | Computed from CSV — correct branch name and % | Bangalore — 90.0% (computed from raw data via Pandas agent) | ✅ Pass |
| 6 | Structured Data | What is the average package across all branches? | Computed mean of Avg_Package_LPA column | 6.53 LPA (computed from Avg_Package_LPA column) | ✅ Pass |
| 7 | Tool Calling | What is the current weather in Mumbai? | Live data from OpenWeatherMap API with temperature, humidity, wind | Live weather returned: temperature, feels-like, humidity, wind speed, visibility. [Live data from OpenWeatherMap API] | ✅ Pass |
| 8 | Refusal | What is the CEO's annual bonus as per company policy? | Refusal — topic not in any document | "I could not find sufficient information in the available documents to answer this." | ✅ Pass |
