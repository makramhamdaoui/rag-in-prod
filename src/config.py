"""
Hybrid configuration:
- config.yaml → non-secret settings (models, ports, thresholds)
- .env        → secrets (passwords, API keys)
"""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"


def _load_yaml_config() -> Dict[str, Any]:
    """Load the YAML config file."""
    if not CONFIG_YAML_PATH.exists():
        return {}
    with open(CONFIG_YAML_PATH, "r") as f:
        return yaml.safe_load(f) or {}


class Secrets(BaseSettings):
    """Secrets from .env — never committed to git."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


class Settings:
    """
    Unified settings combining yaml config + env secrets.
    Access flat attributes: settings.postgres_url, settings.ollama_model_name, etc.
    """

    def __init__(self):
        yaml_cfg = _load_yaml_config()
        secrets = Secrets()

        # --- embedding ---
        emb = yaml_cfg.get("embedding", {})
        self.embedding_model_path: str = emb.get("model_path", "sentence-transformers/all-mpnet-base-v2")
        self.embedding_dimension: int = emb.get("dimension", 768)
        self.asymmetric_embedding: bool = emb.get("asymmetric", False)

        # --- chunking ---
        chunk = yaml_cfg.get("chunking", {})
        self.text_chunk_size: int = chunk.get("chunk_size", 300)
        self.text_chunk_overlap: int = chunk.get("overlap", 100)

        # --- ollama ---
        ollama = yaml_cfg.get("ollama", {})
        self.ollama_host: str = ollama.get("host", "http://localhost:11434")
        self.ollama_model_name: str = ollama.get("model_name", "qwen3:8b")

        # --- logging ---
        log = yaml_cfg.get("logging", {})
        self.log_file_path: str = log.get("file_path", "logs/app.log")
        self.log_level: str = log.get("level", "INFO")

        # --- opensearch ---
        os_cfg = yaml_cfg.get("opensearch", {})
        self.opensearch_host: str = os_cfg.get("host", "localhost")
        self.opensearch_port: int = os_cfg.get("port", 9202)
        self.opensearch_index: str = os_cfg.get("index", "documents")

        # --- postgres ---
        pg = yaml_cfg.get("postgres", {})
        self.postgres_host: str = pg.get("host", "localhost")
        self.postgres_port: int = pg.get("port", 5433)
        self.postgres_db: str = pg.get("database", "rag_db")
        self.postgres_user: str = secrets.postgres_user
        self.postgres_password: str = secrets.postgres_password

        # --- redis ---
        redis_cfg = yaml_cfg.get("redis", {})
        self.redis_host: str = redis_cfg.get("host", "localhost")
        self.redis_port: int = redis_cfg.get("port", 6380)

        # --- langfuse ---
        lf = yaml_cfg.get("langfuse", {})
        self.langfuse_host: str = lf.get("host", "http://localhost:3000")
        self.langfuse_public_key: str = secrets.langfuse_public_key
        self.langfuse_secret_key: str = secrets.langfuse_secret_key

        # --- agents ---
        agents = yaml_cfg.get("agents", {})
        self.guardrail_threshold: int = agents.get("guardrail_threshold", 75)
        self.max_retrieval_attempts: int = agents.get("max_retrieval_attempts", 2)

        # --- cache ---
        cache = yaml_cfg.get("cache", {})
        self.cache_ttl_seconds: int = cache.get("ttl_seconds", 86400)
        self.cache_similarity_threshold: float = cache.get("similarity_threshold", 0.85)

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
