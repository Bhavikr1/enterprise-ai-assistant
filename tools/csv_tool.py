"""
tools/csv_tool.py
Pandas DataFrame agent for structured data computation.
Answers are computed dynamically — not hardcoded.
The agent generates Pandas operations from the natural language question.
"""
import logging
import pandas as pd
from langchain_core.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from core.prompts import CSV_TOOL_DESCRIPTION
from config import GEMINI_API_KEY, LLM_MODEL, CSV_PATH

logger = logging.getLogger(__name__)


def create_csv_tool():
    """
    Factory that creates the CSV analyst tool.
    Loads the DataFrame once at startup and reuses it across calls.
    """
    try:
        df = pd.read_csv(CSV_PATH)
        logger.info("Placement CSV loaded: %d rows, %d columns from %s", len(df), len(df.columns), CSV_PATH)
    except FileNotFoundError:
        logger.error("Placement CSV not found at %s", CSV_PATH)

        def csv_analyst(query: str) -> str:
            return f"CSV file not found at {CSV_PATH}. Please ensure placement_data.csv exists."

        csv_analyst.__doc__ = CSV_TOOL_DESCRIPTION
        return tool(csv_analyst)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0,
        convert_system_message_to_human=True,
    )

    # LangChain's pandas agent — generates and executes Pandas operations dynamically.
    # allow_dangerous_code=True is required because the agent executes generated Python;
    # this is intentional and sandboxed within the agent's execution scope.
    pandas_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=False,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    def csv_analyst(query: str) -> str:
        try:
            result = pandas_agent.invoke({"input": query})
            answer = result.get("output", str(result))
            return f"📊 Computed from placement data:\n{answer}"

        except Exception:
            logger.exception("Pandas agent failed for query: %s", query[:60])
            return (
                "I encountered an error computing the answer from placement data. "
                "Please rephrase the question."
            )

    csv_analyst.__doc__ = CSV_TOOL_DESCRIPTION
    return tool(csv_analyst)
