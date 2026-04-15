from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL
    postgres_db: str = "rag_db"
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"
    postgres_host: str = "localhost"
    postgres_port: int = 5433

    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # OpenSearch
    opensearch_host: str = "localhost"
    opensearch_port: int = 9202
    opensearch_index: str = "documents"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model_name: str = "qwen3:8b"

    # Embedding
    embedding_model_path: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: int = 768
    asymmetric_embedding: bool = False

    # Chunking
    text_chunk_size: int = 300
    text_chunk_overlap: int = 100

    # Langfuse (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
