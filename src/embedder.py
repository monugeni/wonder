"""
embedder.py — local sentence-transformer embeddings

Uses a locally downloaded model so no data leaves the server.
Model is loaded once and cached for the lifetime of the process.

Recommended models (set in .env):
  BAAI/bge-large-en-v1.5   — best quality for technical English (1.3GB)
  BAAI/bge-base-en-v1.5    — good balance (430MB) ← default
  all-MiniLM-L6-v2         — fastest, smallest (90MB)

BGE models from BAAI are particularly strong on domain-specific technical text.
They also support a query prefix for asymmetric retrieval (docs vs queries),
which is important: queries are short, documents are long.
"""

from functools import lru_cache
from typing import Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import config

# BGE models need a query prefix for asymmetric retrieval.
# Documents are embedded as-is; queries get this prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_MODEL_PREFIXES = {"BAAI/bge", "bge-"}


def _is_bge_model(model_name: str) -> bool:
    return any(model_name.startswith(p) or f"/{p.rstrip('-')}" in model_name
               for p in BGE_MODEL_PREFIXES)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the embedding model (called once at startup)."""
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    logger.info(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def get_embedding_dimension() -> int:
    return _load_model().get_sentence_embedding_dimension()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts (no prefix applied).
    These are the enriched chunk texts (context sentence + chunk text).
    """
    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 20,
        normalize_embeddings=True,  # Cosine similarity works better with normalized vectors
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a user query.
    BGE models require a prefix for queries in asymmetric retrieval.
    """
    model = _load_model()
    is_bge = _is_bge_model(config.EMBEDDING_MODEL)
    text = f"{BGE_QUERY_PREFIX}{query}" if is_bge else query

    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()
