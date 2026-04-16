import hashlib
import json
import logging
from typing import Optional

import numpy as np
import redis

from src.config import get_settings

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours
SIMILARITY_THRESHOLD = 0.85        # cosine similarity — tune this


def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
    )


def _exact_key(query: str, num_results: int, use_hybrid_search: bool) -> str:
    key_data = json.dumps({
        "query": query.strip().lower(),
        "num_results": num_results,
        "use_hybrid_search": use_hybrid_search,
    }, sort_keys=True)
    return f"rag_exact:{hashlib.sha256(key_data.encode()).hexdigest()[:16]}"


def _cosine_similarity(a: list, b: list) -> float:
    va = np.array(a)
    vb = np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def get_cached_response(
    query: str,
    num_results: int,
    use_hybrid_search: bool,
    query_embedding: Optional[list] = None,
) -> Optional[dict]:
    """
    Look up cache. Two strategies:
    1. Exact match (fast, O(1))
    2. Semantic match (slower, scans all cached embeddings)

    Returns dict with keys: response, match_type, similarity
    """
    try:
        client = get_redis_client()

        # --- 1. Exact match first ---
        exact_key = _exact_key(query, num_results, use_hybrid_search)
        exact_hit = client.get(exact_key)
        if exact_hit:
            logger.info(f"Exact cache hit: '{query[:50]}'")
            return {"response": exact_hit, "match_type": "exact", "similarity": 1.0}

        # --- 2. Semantic match ---
        if query_embedding is None:
            return None

        # scan all semantic cache entries
        semantic_keys = client.keys("rag_semantic:*")
        best_similarity = 0.0
        best_response = None

        for key in semantic_keys:
            entry_json = client.get(key)
            if not entry_json:
                continue
            try:
                entry = json.loads(entry_json)
                # only compare if same search params
                if entry.get("num_results") != num_results:
                    continue
                if entry.get("use_hybrid_search") != use_hybrid_search:
                    continue
                sim = _cosine_similarity(query_embedding, entry["embedding"])
                if sim > best_similarity:
                    best_similarity = sim
                    best_response = entry["response"]
            except Exception:
                continue

        if best_similarity >= SIMILARITY_THRESHOLD and best_response:
            logger.info(
                f"Semantic cache hit: '{query[:50]}' "
                f"(similarity={best_similarity:.4f})"
            )
            return {
                "response": best_response,
                "match_type": "semantic",
                "similarity": round(best_similarity, 4),
            }

        logger.info(f"Cache miss: '{query[:50]}' (best_sim={best_similarity:.4f})")
        return None

    except Exception as e:
        logger.warning(f"Redis error (get): {e} — proceeding without cache")
        return None


def store_cached_response(
    query: str,
    num_results: int,
    use_hybrid_search: bool,
    response: str,
    query_embedding: Optional[list] = None,
) -> None:
    """
    Store response in two ways:
    1. Exact key — for fast exact match lookup
    2. Semantic key — embedding + response for semantic lookup
    """
    try:
        client = get_redis_client()

        # exact cache
        exact_key = _exact_key(query, num_results, use_hybrid_search)
        client.set(exact_key, response, ex=CACHE_TTL_SECONDS)

        # semantic cache
        if query_embedding is not None:
            sem_key = f"rag_semantic:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
            entry = json.dumps({
                "query": query.strip().lower(),
                "embedding": query_embedding,
                "response": response,
                "num_results": num_results,
                "use_hybrid_search": use_hybrid_search,
            })
            client.set(sem_key, entry, ex=CACHE_TTL_SECONDS)
            logger.info(f"Stored semantic cache: '{query[:50]}'")

    except Exception as e:
        logger.warning(f"Redis error (set): {e} — cache not stored")
