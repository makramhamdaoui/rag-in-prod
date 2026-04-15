import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from opensearchpy import OpenSearch, helpers

from src.constants import ASYMMETRIC_EMBEDDING, EMBEDDING_DIMENSION, OPENSEARCH_INDEX
from src.opensearch import get_opensearch_client
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_index_config() -> Dict[str, Any]:
    with open("src/index_config.json", "r") as f:
        config = json.load(f)
    config["mappings"]["properties"]["embedding"]["dimension"] = EMBEDDING_DIMENSION
    logger.info("Index config loaded.")
    return config


def create_index(client: OpenSearch) -> None:
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        client.indices.create(index=OPENSEARCH_INDEX, body=load_index_config())
        logger.info(f"Index '{OPENSEARCH_INDEX}' created.")
    else:
        logger.info(f"Index '{OPENSEARCH_INDEX}' already exists.")


def delete_index(client: OpenSearch) -> None:
    if client.indices.exists(index=OPENSEARCH_INDEX):
        client.indices.delete(index=OPENSEARCH_INDEX)
        logger.info(f"Index '{OPENSEARCH_INDEX}' deleted.")


def create_search_pipeline(client: OpenSearch) -> None:
    """Create the nlp-search-pipeline for hybrid search score normalization."""
    pipeline_body = {
        "description": "Hybrid search score normalization",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.3, 0.7]},
                    },
                }
            }
        ],
    }
    client.transport.perform_request(
        "PUT",
        "/_search/pipeline/nlp-search-pipeline",
        body=pipeline_body,
    )
    logger.info("Search pipeline created.")


def bulk_index_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[Any],
) -> Tuple[int, List[Any]]:
    """
    Index chunks with embeddings and full metadata into OpenSearch.

    Args:
        chunks     : list of chunk dicts from chunk_document()
        embeddings : list of numpy arrays from generate_embeddings()

    Returns:
        (success_count, errors)
    """
    client = get_opensearch_client()
    actions = []
    indexed_at = datetime.now(timezone.utc).isoformat()

    for chunk, embedding in zip(chunks, embeddings):
        text = f"passage: {chunk['text']}" if ASYMMETRIC_EMBEDDING else chunk["text"]
        actions.append({
            "_index": OPENSEARCH_INDEX,
            "_id": str(uuid.uuid4()),
            "_source": {
                "text":               text,
                "embedding":          embedding.tolist(),
                "document_id":        chunk["document_id"],
                "document_name":      chunk["document_name"],
                "section_title":      chunk["section_title"],
                "page_number":        chunk["page_number"],
                "chunk_index":        chunk["chunk_index"],
                "global_chunk_index": chunk["global_chunk_index"],
                "total_chunks":       chunk["total_chunks"],
                "section_index":      chunk["section_index"],
                "total_sections":     chunk["total_sections"],
                "char_count":         chunk["char_count"],
                "indexed_at":         indexed_at,
            },
        })

    success, errors = helpers.bulk(client, actions)
    logger.info(f"Indexed {success} chunks, {len(errors)} errors.")
    return success, errors


def delete_chunks_by_document_id(document_id: str) -> Dict[str, Any]:
    """Delete all chunks for a document from OpenSearch."""
    client = get_opensearch_client()
    response = client.delete_by_query(
        index=OPENSEARCH_INDEX,
        body={"query": {"term": {"document_id": document_id}}},
    )
    logger.info(f"Deleted chunks for document_id='{document_id}'.")
    return response
