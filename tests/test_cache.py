"""Redis semantic cache tests."""
import pytest

from src.services.cache.client import (
    _exact_key,
    _cosine_similarity,
    get_redis_client,
    store_cached_response,
    get_cached_response,
)


def test_cosine_similarity_identical():
    a = [1.0, 0.0, 0.0]
    assert _cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_exact_key_deterministic():
    k1 = _exact_key("hello", 3, True)
    k2 = _exact_key("hello", 3, True)
    assert k1 == k2


def test_exact_key_different_for_different_queries():
    k1 = _exact_key("hello", 3, True)
    k2 = _exact_key("world", 3, True)
    assert k1 != k2


def test_exact_key_normalizes_case_and_whitespace():
    k1 = _exact_key("Hello World", 3, True)
    k2 = _exact_key("  hello world  ", 3, True)
    assert k1 == k2


def test_cache_store_and_retrieve():
    """Integration test — requires running Redis."""
    try:
        client = get_redis_client()
        client.ping()
    except Exception:
        pytest.skip("Redis not available")

    query = "test cache query unique 12345"
    embedding = [0.1] * 768

    store_cached_response(
        query=query,
        num_results=3,
        use_hybrid_search=True,
        response="cached response",
        query_embedding=embedding,
    )

    result = get_cached_response(
        query=query,
        num_results=3,
        use_hybrid_search=True,
        query_embedding=embedding,
    )

    assert result is not None
    assert result["response"] == "cached response"
    assert result["match_type"] == "exact"
