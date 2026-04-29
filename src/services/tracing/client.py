import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


class RAGTracer:
    """
    Langfuse tracer for RAG pipeline observability.
    Tracks: query, retrieval, LLM generation, latency, cache hits.
    Gracefully disabled if Langfuse is not configured.
    """

    def __init__(self):
        self.client = None
        settings = get_settings()

        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info(f"Langfuse tracing initialized at {settings.langfuse_host}")
            except Exception as e:
                logger.warning(f"Langfuse init failed: {e} — tracing disabled")
        else:
            logger.info("Langfuse keys not set — tracing disabled")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def trace_rag_query(
        self,
        query: str,
        session_id: str,
        response: str,
        retrieved_chunks: list,
        latency_ms: float,
        cached: bool = False,
        cache_type: Optional[str] = None,
        similarity: Optional[float] = None,
    ) -> Optional[str]:
        """
        Create a full trace for one RAG query covering:
        - Input query
        - Retrieved chunks (retrieval span)
        - LLM response (generation span)
        - Latency and cache metadata
        """
        if not self.enabled:
            return None

        try:
            trace = self.client.trace(
                name="rag_query",
                input=query,
                output=response,
                session_id=session_id,
                metadata={
                    "cached": cached,
                    "cache_type": cache_type,
                    "similarity": similarity,
                    "latency_ms": round(latency_ms, 2),
                    "num_chunks": len(retrieved_chunks),
                },
            )

            # retrieval span
            trace.span(
                name="hybrid_search",
                input={"query": query},
                output={
                    "chunks": [
                        {
                            "section": c.get("_source", {}).get("section_title"),
                            "page": c.get("_source", {}).get("page_number"),
                            "score": c.get("_score"),
                        }
                        for c in retrieved_chunks
                    ]
                },
                metadata={"num_results": len(retrieved_chunks)},
            )

            # generation span
            if not cached:
                settings = get_settings()
                trace.generation(
                    name="ollama_generation",
                    model=settings.ollama_model_name,
                    input=query,
                    output=response,
                    metadata={"latency_ms": round(latency_ms, 2)},
                )

            self.client.flush()
            logger.info(f"Traced RAG query — trace_id={trace.id}")
            return trace.id

        except Exception as e:
            logger.warning(f"Langfuse trace failed: {e}")
            return None

    def flush(self):
        if self.client:
            self.client.flush()


# singleton
_tracer: Optional[RAGTracer] = None


def get_tracer() -> RAGTracer:
    global _tracer
    if _tracer is None:
        _tracer = RAGTracer()
    return _tracer
