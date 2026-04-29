import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import ollama

from src.constants import ASYMMETRIC_EMBEDDING, OLLAMA_MODEL_NAME
from src.services.embeddings.client import get_embedding_model
from src.services.llm.prompts import build_prompt
from src.services.search.hybrid_search import hybrid_search
from src.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def ensure_model_pulled(model: str) -> bool:
    """Check model is available locally, pull if not. Runs once per model name."""
    try:
        available = ollama.list()
        if model not in str(available):
            logger.info(f"Pulling model {model}...")
            ollama.pull(model)
        logger.info(f"Model {model} ready.")
    except ollama.ResponseError as e:
        logger.error(f"Ollama error: {e.error}")
        return False
    return True


def run_llm_streaming(
    prompt: str,
    temperature: float,
) -> Optional[Iterable]:
    """Send prompt to Ollama and return a streaming generator."""
    try:
        return ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature},
        )
    except ollama.ResponseError as e:
        logger.error(f"Streaming error: {e.error}")
        return None


def generate_response_streaming(
    query: str,
    use_hybrid_search: bool = True,
    num_results: int = 3,
    temperature: float = 0.3,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Iterable]:
    """
    Full RAG pipeline:
    1. embed query
    2. hybrid search → retrieve context
    3. build prompt with context + history
    4. stream response from Ollama
    """
    ensure_model_pulled(OLLAMA_MODEL_NAME)
    chat_history = chat_history or []
    history = chat_history[-10:]  # keep last 10 messages
    context = ""

    if use_hybrid_search:
        prefix = "passage: " if ASYMMETRIC_EMBEDDING else ""
        query_embedding = get_embedding_model().encode(prefix + query).tolist()
        results = hybrid_search(query, query_embedding, top_k=num_results)
        for i, hit in enumerate(results):
            context += f"Document {i}:\n{hit['_source']['text']}\n\n"
        logger.info(f"Retrieved {len(results)} chunks for query.")

    prompt = build_prompt(query, context, history)
    return run_llm_streaming(prompt, temperature)
