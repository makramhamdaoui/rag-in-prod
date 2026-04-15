from src.config import get_settings

_s = get_settings()

# Embedding
EMBEDDING_MODEL_PATH  = _s.embedding_model_path
ASYMMETRIC_EMBEDDING  = _s.asymmetric_embedding
EMBEDDING_DIMENSION   = _s.embedding_dimension

# Chunking
TEXT_CHUNK_SIZE    = _s.text_chunk_size
TEXT_CHUNK_OVERLAP = _s.text_chunk_overlap

# Ollama
OLLAMA_MODEL_NAME = _s.ollama_model_name

# Logging
LOG_FILE_PATH = "logs/app.log"

# OpenSearch
OPENSEARCH_HOST  = _s.opensearch_host
OPENSEARCH_PORT  = _s.opensearch_port
OPENSEARCH_INDEX = _s.opensearch_index

# PostgreSQL
POSTGRES_URL = _s.postgres_url
