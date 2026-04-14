import json
import logging
import uuid
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


def bulk_index_documents(
    documents: List[Dict[str, Any]],
) -> Tuple[int, List[Any]]:
    client = get_opensearch_client()
    actions = []
    for doc in documents:
        text = f"passage: {doc['text']}" if ASYMMETRIC_EMBEDDING else doc["text"]
        actions.append({
            "_index": OPENSEARCH_INDEX,
            "_id": doc.get("doc_id", str(uuid.uuid4())),
            "_source": {
                "text": text,
                "embedding": doc["embedding"].tolist(),
                "document_name": doc["document_name"],
            },
        })
    success, errors = helpers.bulk(client, actions)
    logger.info(f"Indexed {success} docs, {len(errors)} errors.")
    return success, errors


def delete_documents_by_name(document_name: str) -> Dict[str, Any]:
    client = get_opensearch_client()
    response = client.delete_by_query(
        index=OPENSEARCH_INDEX,
        body={"query": {"term": {"document_name": document_name}}},
    )
    logger.info(f"Deleted docs for '{document_name}'.")
    return response
