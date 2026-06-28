"""
core/agent.py
ReAct agent assembly — binds tools, memory, system prompt into AgentExecutor.
The LLM reads tool descriptions and reasons about which tool to invoke.
No hardcoded routing logic.
"""
import logging
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from core.memory import create_memory
from core.prompts import SYSTEM_PROMPT
from tools.rag_tool import create_rag_tool, set_vectorstore
from tools.csv_tool import create_csv_tool
from tools.weather_tool import create_weather_tool
from tools.currency_tool import create_currency_tool
from rag.ingest import run_ingestion
from config import GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


# ReAct agent requires a specific prompt format.
# {tools} and {tool_names} are filled by LangChain from the tool list.
# {agent_scratchpad} holds the Thought/Action/Observation trace.
# {chat_history} receives the formatted string from ConversationSummaryBufferMemory
# (return_messages=False ensures this is a plain string, not a list of BaseMessage).
REACT_PROMPT_TEMPLATE = SYSTEM_PROMPT + """

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: reason about what you need to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important:
- Always start with a Thought.
- Always use a tool before giving the Final Answer (except for out-of-scope questions).
- Never skip directly to Final Answer without reasoning.
- If a tool returns that information was not found or is unavailable, immediately give Final Answer with that message. Do not call another tool.

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


class AIAssistant:
    """
    Main assistant class — initialises everything and exposes a run() method.
    Handles ingestion, tool creation, agent assembly, and memory.

    Thread-safety note: this is a singleton used across a single server process.
    FastAPI with a thread-pool executor means concurrent requests may call run()
    simultaneously. AgentExecutor is not thread-safe out of the box; each
    invocation creates its own chain state, but shared memory is not isolated.
    For production multi-tenant use, create one AIAssistant per session.
    """

    def __init__(self):
        self.vectorstore = None
        self.agent_executor = None
        self.memory = None
        self._initialised = False

    def initialise(self, force_reingest: bool = False) -> None:
        """
        Initialise the full pipeline:
        1. Run document ingestion (or load existing ChromaDB index)
        2. Create all four tools
        3. Assemble ReAct agent with memory
        """
        logger.info("Initialising AI Enterprise Assistant...")

        logger.info("Step 1/3 — Document ingestion...")
        self.vectorstore = run_ingestion(force=force_reingest)
        set_vectorstore(self.vectorstore)

        logger.info("Step 2/3 — Creating tools...")
        tools = [
            create_rag_tool(),
            create_csv_tool(),
            create_weather_tool(),
            create_currency_tool(),
        ]

        logger.info("Step 3/3 — Assembling ReAct agent...")
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE,
            convert_system_message_to_human=True,
        )

        self.memory = create_memory()

        prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            early_stopping_method="force",
        )

        self._initialised = True
        logger.info("Assistant ready.")

    def run(self, user_input: str) -> dict:
        """
        Process a user query through the full pipeline.
        Returns a dict with answer, success flag, and error context.
        """
        if not self._initialised:
            raise RuntimeError("Assistant not initialised. Call initialise() first.")

        try:
            result = self.agent_executor.invoke({"input": user_input})
            answer = result.get("output", "I could not generate a response.")
            return {
                "answer": answer,
                "success": True,
                "error": None,
            }

        except Exception as exc:
            logger.exception("Agent execution failed for input: %s", user_input[:80])
            return {
                "answer": "I encountered an error processing your request. Please try again.",
                "success": False,
                "error": str(exc),
            }

    def clear_memory(self) -> None:
        """Reset conversation memory — called when user starts a new session."""
        if self.memory:
            self.memory.clear()
            logger.debug("Conversation memory cleared.")


# ── Singleton ─────────────────────────────────────────────────────────────────
# One instance per server process. For concurrent multi-user deployments,
# move to a per-session factory instead.
_assistant_instance: "AIAssistant | None" = None


def get_assistant() -> AIAssistant:
    """Return the singleton assistant instance (uninitialised until initialise() is called)."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = AIAssistant()
    return _assistant_instance
