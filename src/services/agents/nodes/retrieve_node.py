import logging
from typing import Dict

from src.services.embeddings.client import get_embedding_model
from src.services.search.hybrid_search import hybrid_search
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)
MAX_RETRIEVAL_ATTEMPTS = 2


def retrieve_node(state: AgentState) -> Dict:
    """Retrieve relevant chunks using hybrid search."""
    logger.info("NODE: retrieve")

    # use rewritten query if available
    query = state.get("rewritten_query") or state.get("original_query", "")
    attempts = state.get("retrieval_attempts", 0) + 1

    model = get_embedding_model()
    embedding = model.encode(query).tolist()
    chunks = hybrid_search(query, embedding, top_k=5)

    # build context string
    context = ""
    for i, hit in enumerate(chunks):
        src = hit.get("_source", {})
        context += (
            f"Document {i} (section: {src.get('section_title', 'N/A')}, "
            f"page: {src.get('page_number', 'N/A')}):\n"
            f"{src.get('text', '')}\n\n"
        )

    logger.info(f"Retrieved {len(chunks)} chunks (attempt {attempts})")
    return {
        "retrieved_chunks": chunks,
        "context": context,
        "retrieval_attempts": attempts,
    }
