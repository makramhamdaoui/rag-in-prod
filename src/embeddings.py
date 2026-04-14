import logging
from functools import lru_cache
from typing import Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.constants import EMBEDDING_MODEL_PATH
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model (runs only once)."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_PATH}")
    return SentenceTransformer(EMBEDDING_MODEL_PATH)


def generate_embeddings(chunks: List[str]) -> List[np.ndarray]:
    """Generate normalized embeddings for a list of text chunks."""
    model = get_embedding_model()
    embeddings = [np.array(model.encode(chunk)) for chunk in chunks]
    logger.info(f"Generated {len(embeddings)} embeddings.")
    return embeddings
