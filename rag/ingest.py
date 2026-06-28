"""
rag/ingest.py
Document ingestion pipeline — runs once at startup (or when documents change).
Load → Chunk → Embed → Store in ChromaDB.
Separate from retriever.py because ingestion and retrieval have different lifecycles.
"""
import os
import glob
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    DOCUMENTS_DIR
)

logger = logging.getLogger(__name__)

# Cosine distance normalises to [0, 2] (identical=0, opposite=2) and maps cleanly
# to the confidence formula: confidence = 1 - (distance / 2).
# L2 (default) has an unbounded upper range and breaks the formula for large vectors.
_CHROMA_DISTANCE_METRIC = "cosine"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return Gemini embedding model instance."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )


def load_documents(docs_dir: str) -> list:
    """
    Load all .txt and .pdf files from the documents directory.
    Returns a flat list of LangChain Document objects.
    """
    documents = []
    for txt_path in glob.glob(os.path.join(docs_dir, "*.txt")):
        loader = TextLoader(txt_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            filename = os.path.basename(txt_path)
            doc.metadata["source_file"] = filename
            doc.metadata["document_type"] = _infer_document_type(filename)
        documents.extend(docs)
        logger.info("Loaded: %s (%d pages)", txt_path, len(docs))

    for pdf_path in glob.glob(os.path.join(docs_dir, "*.pdf")):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            filename = os.path.basename(pdf_path)
            doc.metadata["source_file"] = filename
            doc.metadata["document_type"] = _infer_document_type(filename)
        documents.extend(docs)
        logger.info("Loaded: %s (%d pages)", pdf_path, len(docs))

    return documents


def _infer_document_type(filename: str) -> str:
    """Tag document type from filename for metadata filtering."""
    name = filename.lower()
    if "policy" in name:
        return "policy"
    elif "sop" in name or "procedure" in name:
        return "sop"
    elif "screening" in name or "candidate" in name:
        return "operational"
    return "general"


def chunk_documents(documents: list) -> list:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.

    Why RecursiveCharacterTextSplitter:
    - Splits on paragraph → sentence → word boundary in priority order.
    - Preserves semantic coherence — never cuts mid-sentence unless the
      paragraph itself exceeds chunk_size.
    - Simple CharacterTextSplitter cuts on character count alone = mid-word cuts.

    Why 512 tokens:
    - Fits one complete policy clause or SOP step with context.
    - 256 loses intra-step references. 1024 groups unrelated clauses → noisy retrieval.

    Why 64 overlap:
    - Prevents boundary fragmentation.
    - A sentence straddling two chunks appears complete in at least one chunk.
    - Critical for SOPs where steps reference each other.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def build_vector_store(chunks: list) -> Chroma:
    """
    Embed chunks and store in ChromaDB with explicit cosine distance.

    Why cosine distance (not the default L2):
    - The confidence formula confidence = 1 - (distance / 2) requires a
      bounded [0, 2] distance range, which cosine guarantees.
    - L2 distance is unbounded; for high-magnitude embedding vectors it can
      exceed 2, making confidence scores negative before clamping.
    - Cosine similarity is semantically appropriate for comparing dense embeddings.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": _CHROMA_DISTANCE_METRIC},
    )
    logger.info("ChromaDB collection built with %s distance metric.", _CHROMA_DISTANCE_METRIC)
    return vectorstore


def load_existing_store() -> Chroma:
    """Load existing ChromaDB collection from disk (no re-ingestion)."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def is_store_populated() -> bool:
    """Check if ChromaDB already has documents — avoid re-ingestion on restart."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return False
    try:
        store = load_existing_store()
        count = store._collection.count()
        if count == 0:
            return False
        # Warn if the existing collection uses a different distance metric than expected.
        # This does NOT trigger re-ingestion (existing data is used as-is to avoid
        # unnecessary API calls on restart). Re-ingest manually with force=True to
        # rebuild the collection with the correct cosine metric.
        hnsw_cfg = store._collection.configuration_json.get("hnsw", {})
        stored_metric = hnsw_cfg.get("space")
        if stored_metric != _CHROMA_DISTANCE_METRIC:
            logger.warning(
                "ChromaDB collection uses '%s' distance, expected '%s'. "
                "Confidence scores may be approximate. "
                "Re-ingest with force=True to rebuild with correct metric.",
                stored_metric, _CHROMA_DISTANCE_METRIC,
            )
        return True
    except Exception:
        logger.exception("Failed to inspect ChromaDB collection.")
        return False


def run_ingestion(force: bool = False) -> Chroma:
    """
    Main ingestion entry point.
    If ChromaDB is already populated with the correct metric and force=False, reuse index.
    If force=True or the metric is wrong, re-ingest all documents.
    """
    if not force and is_store_populated():
        logger.info("ChromaDB already populated with correct metric — loading existing index.")
        return load_existing_store()

    logger.info("Starting document ingestion...")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        raise FileNotFoundError(f"No documents found in {DOCUMENTS_DIR}")

    logger.info("Total raw documents loaded: %d", len(documents))
    chunks = chunk_documents(documents)
    logger.info("Total chunks after splitting: %d", len(chunks))

    logger.info("Embedding and storing in ChromaDB...")
    store = build_vector_store(chunks)
    logger.info("Ingestion complete — %d chunks stored.", len(chunks))
    return store
