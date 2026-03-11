"""
config.py — loads and validates environment configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Anthropic
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CONTEXT_MODEL: str = os.getenv("CONTEXT_MODEL", "claude-haiku-4-5-20251001")

    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "engineering_rag")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    # Storage
    TABLE_STORE_DIR: Path = Path(os.getenv("TABLE_STORE_DIR", "./table_store"))

    # PDF Splitting (for large concatenated tender PDFs)
    SPLIT_THRESHOLD: int = int(os.getenv("SPLIT_THRESHOLD", "100"))
    """PDFs with this many pages or more are automatically split before ingestion."""
    SPLIT_SCORE_THRESHOLD: float = float(os.getenv("SPLIT_SCORE_THRESHOLD", "3.0"))
    """Confidence threshold for the splitter's signal fusion (higher = fewer splits)."""
    SPLIT_MIN_DOC_PAGES: int = int(os.getenv("SPLIT_MIN_DOC_PAGES", "4"))
    """Minimum pages per split segment (prevents tiny fragments)."""

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        cls.TABLE_STORE_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
