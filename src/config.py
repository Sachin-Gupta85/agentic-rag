"""Environment-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k: int = int(os.getenv("TOP_K", "4"))
    max_query_rewrites: int = int(os.getenv("MAX_QUERY_REWRITES", "2"))
    collection_name: str = "agentic_kb"

    def validate_llm(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(or point OPENAI_BASE_URL at a local OpenAI-compatible server)."
            )


settings = Settings()
