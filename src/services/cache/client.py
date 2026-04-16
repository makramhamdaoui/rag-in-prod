import hashlib
import json
import logging
from typing import Optional

import redis

from src.config import get_settings

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
    )


def _cache_key(query: str, num_results: int, use_hybrid_search: bool) -> str:
    """Generate a deterministic cache key from query parameters."""
    key_data = json.dumps({
        "query": query.strip().lower(),
        "num_results": num_results,
        "use_hybrid_search": use_hybrid_search,
    }, sort_keys=True)
    hash_str = hashlib.sha256(key_data.encode()).hexdigest()[:16]
    return f"rag_cache:{hash_str}"


def get_cached_response(
    query: str,
    num_results: int,
    use_hybrid_search: bool,
) -> Optional[str]:
    """Return cached response string or None if not found."""
    try:
        client = get_redis_client()
        key = _cache_key(query, num_results, use_hybrid_search)
        value = client.get(key)
        if value:
            logger.info(f"Cache hit for query: '{query[:50]}'")
            return value
        logger.info(f"Cache miss for query: '{query[:50]}'")
        return None
    except Exception as e:
        logger.warning(f"Redis error (get): {e} — proceeding without cache")
        return None


def store_cached_response(
    query: str,
    num_results: int,
    use_hybrid_search: bool,
    response: str,
) -> None:
    """Store response in Redis with TTL."""
    try:
        client = get_redis_client()
        key = _cache_key(query, num_results, use_hybrid_search)
        client.set(key, response, ex=CACHE_TTL_SECONDS)
        logger.info(f"Cached response for query: '{query[:50]}'")
    except Exception as e:
        logger.warning(f"Redis error (set): {e} — cache not stored")
