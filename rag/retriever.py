"""
rag/retriever.py
Query ChromaDB, compute confidence scores, build citations.
Separate from ingest.py — retrieval runs on every query, ingestion runs once.
"""
import logging
from langchain_chroma import Chroma
from core.reliability import compute_confidence, get_confidence_label
from config import TOP_K

logger = logging.getLogger(__name__)


def retrieve(vectorstore: Chroma, query: str, k: int = TOP_K) -> dict:
    """
    Retrieve top-K chunks from ChromaDB with confidence scores.

    Distance semantics:
    - similarity_search_with_score returns the raw ChromaDB distance.
    - Collections created with hnsw:space=cosine return cosine distance in [0, 2]:
        0 = identical vectors, 2 = maximally dissimilar.
    - compute_confidence maps this to [0, 1]: confidence = 1 - (distance / 2).
    - Results are ordered by ascending distance (most similar first).

    Returns a dict with:
    - chunks: list of (document, confidence_score) tuples
    - best_confidence: highest confidence score (first result, lowest distance)
    - context_text: formatted context string for LLM prompt
    - citations: list of source citation strings
    - confidence_label: human-readable label for the best confidence
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    if not results:
        logger.warning("No results returned from ChromaDB for query: %s", query[:60])
        return {
            "chunks": [],
            "best_confidence": 0.0,
            "context_text": "",
            "citations": [],
            "confidence_label": "Low (0%)",
        }

    chunks_with_scores = []
    citations = []
    context_parts = []

    for i, (doc, distance) in enumerate(results):
        confidence = compute_confidence(distance)
        chunks_with_scores.append((doc, confidence))

        source = doc.metadata.get("source_file", "Unknown document")
        page   = doc.metadata.get("page", doc.metadata.get("page_number", "—"))
        citation = f"[Source: {source}, Page: {page}]"
        citations.append(citation)

        context_parts.append(
            f"--- Excerpt {i+1} {citation} ---\n{doc.page_content}"
        )

    # results are ordered ascending by distance → first entry is the best match
    best_confidence = chunks_with_scores[0][1]

    logger.debug(
        "Retrieved %d chunks for query '%s...' — best confidence: %.4f",
        len(results), query[:40], best_confidence,
    )

    return {
        "chunks": chunks_with_scores,
        "best_confidence": best_confidence,
        "context_text": "\n\n".join(context_parts),
        "citations": citations,
        "confidence_label": get_confidence_label(best_confidence),
    }


def format_retrieval_summary(retrieval_result: dict) -> str:
    """
    Format retrieval metadata as a summary string for UI display.
    Shows what was retrieved and how confident the system is.
    """
    if not retrieval_result["chunks"]:
        return "No relevant content found."

    lines = [f"📄 Retrieved {len(retrieval_result['chunks'])} chunks"]
    lines.append(f"🎯 Confidence: {retrieval_result['confidence_label']}")
    for citation in retrieval_result["citations"]:
        lines.append(f"  {citation}")
    return "\n".join(lines)
