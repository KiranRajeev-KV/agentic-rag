from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")
    app_db_path: Path = Field(default=Path("./data/app.sqlite"), alias="APP_DB_PATH")
    app_runs_dir: Path = Field(default=Path("./runs"), alias="APP_RUNS_DIR")
    app_pdf_dir: Path = Field(default=Path("./data/raw_pdfs"), alias="APP_PDF_DIR")
    app_log_jsonl: Path = Field(default=Path("./runs/logs/app.jsonl"), alias="APP_LOG_JSONL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="arxiv_child_chunks", alias="QDRANT_COLLECTION")
    bge_model_name: str = Field(default="BAAI/bge-m3", alias="BGE_MODEL_NAME")
    bge_device: str = Field(default="auto", alias="BGE_DEVICE")
    llm_provider: str = Field(default="", alias="LLM_PROVIDER")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    arxiv_user_agent: str = Field(default="agentic-rag-assignment/0.1", alias="ARXIV_USER_AGENT")

    def ensure_runtime_dirs(self) -> None:
        self.app_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.app_runs_dir.mkdir(parents=True, exist_ok=True)
        self.app_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.app_log_jsonl.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
