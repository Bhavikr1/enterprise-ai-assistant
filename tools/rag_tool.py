"""
tools/rag_tool.py
LangChain tool wrapping the RAG retrieval pipeline.
The agent calls this tool for all document/policy/SOP questions.
"""
import logging
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from core.prompts import RAG_ANSWER_TEMPLATE, RAG_TOOL_DESCRIPTION
from core.reliability import should_refuse, maybe_add_uncertainty_warning, get_refusal_message
from rag.retriever import retrieve
from config import GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# vectorstore is injected at startup — see core/agent.py
_vectorstore = None


def set_vectorstore(vs) -> None:
    """Inject the ChromaDB vectorstore after ingestion completes."""
    global _vectorstore
    _vectorstore = vs


def create_rag_tool():
    """
    Factory function that creates the RAG tool with access to the vectorstore.
    Returns a LangChain tool the agent can call.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=LLM_TEMPERATURE,
        convert_system_message_to_human=True,
    )

    def rag_retriever(query: str) -> str:
        if _vectorstore is None:
            logger.error("rag_retriever called before vectorstore was injected.")
            return "Document store is not initialised. Please restart the application."

        # Step 1: Retrieve from ChromaDB
        retrieval = retrieve(_vectorstore, query)

        # Step 2: Confidence gate — hard refusal below threshold (no LLM call)
        if should_refuse(retrieval["best_confidence"]):
            logger.info(
                "Confidence gate triggered: %.4f < threshold for query: %s",
                retrieval["best_confidence"], query[:60],
            )
            return get_refusal_message(retrieval["best_confidence"])

        # Step 3: Build prompt and generate grounded answer
        prompt = RAG_ANSWER_TEMPLATE.format(
            context=retrieval["context_text"],
            question=query,
            confidence_label=retrieval["confidence_label"],
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

            # If the LLM couldn't find relevant content, return the refusal cleanly.
            # Appending "Confidence: High" to a not-found response is contradictory
            # and causes the ReAct agent to loop looking for more actions.
            _not_found_phrases = ("could not find", "not found", "not available", "no information")
            if any(phrase in answer.lower() for phrase in _not_found_phrases):
                return "I could not find sufficient information in the available documents to answer this question."

            # Step 4: Append uncertainty warning for moderate confidence
            answer = maybe_add_uncertainty_warning(answer, retrieval["best_confidence"])

            # Step 5: Append citations and confidence
            citation_str = "\n".join(retrieval["citations"])
            answer += f"\n\n📚 Sources: {citation_str}"
            answer += f"\n🎯 Confidence: {retrieval['confidence_label']}"

            return answer

        except Exception:
            logger.exception("LLM call failed in rag_retriever for query: %s", query[:60])
            return "I encountered an error generating the answer. Please try again."

    # LangChain reads __doc__ as the tool description for agent tool-selection.
    # String-concat as a docstring is a syntax no-op — it evaluates to an expression
    # that Python discards without storing in __doc__. Explicit assignment is correct.
    rag_retriever.__doc__ = RAG_TOOL_DESCRIPTION
    return tool(rag_retriever)
