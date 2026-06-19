from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "meta-llama/llama-3.1-8b-instruct"   
    openai_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    qdrant_url: str = "http://localhost:6333"
    chunks_collection: str = "docs_chunks"
    cache_collection: str = "semantic_cache"
    redis_url: str = "redis://localhost:6379/0"
    cost_db_path: str = "data/costs.db"
    docs_dir: str = "data/docs"
    top_k: int = 3
    min_relevance_score: float = 0.40   # if top-1 cosine is below this, context is irrelevant -> abstain
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    cache_threshold: float = 0.90   # cosine-similarity threshold; >0.9 catches paraphrases
    cache_ttl_seconds: int = 3600
    max_input_chars: int = 4000
    max_concurrent_llm: int = 20
    llm_timeout_seconds: float = 15.0
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30   # dense top-N -> CrossEncoder rerank -> top_k
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()