from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Framework"
    docs_dir: Path = Field(default=Path("data/docs"), validation_alias="DOCS_DIR")
    index_dir: Path = Field(default=Path("data/index"), validation_alias="INDEX_DIR")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="llama3.1:8b", validation_alias="LLM_MODEL")
    llm_base_url: str = Field(default="http://localhost:11434", validation_alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")
    chunk_size: int = Field(default=900, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, validation_alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, validation_alias="TOP_K")
    relevance_threshold: float = Field(default=0.28, validation_alias="RELEVANCE_THRESHOLD")
    rewrite_intent_threshold: float = Field(
        default=0.65,
        validation_alias="REWRITE_INTENT_THRESHOLD",
    )
    rewrite_evidence_margin: float = Field(
        default=0.05,
        validation_alias="REWRITE_EVIDENCE_MARGIN",
    )
    reranker_evidence_threshold: float = Field(
        default=0.0,
        validation_alias="RERANKER_EVIDENCE_THRESHOLD",
    )
    enable_reranker: bool = Field(default=False, validation_alias="ENABLE_RERANKER")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="RERANKER_MODEL",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
