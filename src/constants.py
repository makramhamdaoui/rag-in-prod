from pathlib import Path
import yaml

_config_path = Path(__file__).parent.parent / "config.yaml"

with open(_config_path, "r") as f:
    _cfg = yaml.safe_load(f)

# Embedding
EMBEDDING_MODEL_PATH = _cfg["embedding"]["model_path"]
ASYMMETRIC_EMBEDDING  = _cfg["embedding"]["asymmetric"]
EMBEDDING_DIMENSION   = _cfg["embedding"]["dimension"]

# Chunking
TEXT_CHUNK_SIZE    = _cfg["chunking"]["chunk_size"]
TEXT_CHUNK_OVERLAP = _cfg["chunking"]["overlap"]

# Ollama
OLLAMA_MODEL_NAME = _cfg["ollama"]["model_name"]

# Logging
LOG_FILE_PATH = _cfg["logging"]["file_path"]

# OpenSearch
OPENSEARCH_HOST  = _cfg["opensearch"]["host"]
OPENSEARCH_PORT  = _cfg["opensearch"]["port"]
OPENSEARCH_INDEX = _cfg["opensearch"]["index"]
