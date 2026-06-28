"""
core/memory.py
ConversationSummaryBufferMemory setup.
Keeps recent turns verbatim + rolling LLM summary of older turns.
Token-efficient and context-preserving — production realistic behaviour.
"""
import logging
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY, LLM_MODEL, MEMORY_MAX_TOKEN_LIMIT

logger = logging.getLogger(__name__)


def create_memory() -> ConversationSummaryBufferMemory:
    """
    Create a ConversationSummaryBufferMemory instance.

    Why SummaryBufferMemory over BufferMemory:
    - BufferMemory keeps full history verbatim → token window blows up on long sessions.
    - SummaryMemory compresses everything → loses precision of recent context.
    - SummaryBufferMemory: keeps last N tokens verbatim + rolling summary of older turns.
    - Best of both worlds. Production-realistic for enterprise sessions of 20-30 turns.

    Why return_messages=False:
    - The ReAct agent uses a string-based PromptTemplate with a {chat_history} placeholder.
    - return_messages=True would inject a list of BaseMessage objects, which Python would
      serialise as their repr (e.g. "[HumanMessage(content='...')]") — unreadable in the prompt.
    - return_messages=False formats history as a clean "Human: ... AI: ..." string.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0,
        convert_system_message_to_human=True,
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=MEMORY_MAX_TOKEN_LIMIT,
        memory_key="chat_history",
        return_messages=False,   # string format — required for ReAct PromptTemplate
        output_key="output",
    )
    logger.debug("ConversationSummaryBufferMemory created (max_token_limit=%d)", MEMORY_MAX_TOKEN_LIMIT)
    return memory
