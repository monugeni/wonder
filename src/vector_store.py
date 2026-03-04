"""
vector_store.py — Qdrant operations with per-folder collection isolation

Each project/tender folder has its own Qdrant collection.
This gives hard query isolation — a search against Project 269 will never
return results from Project 245, even if they share terminology.

Collection lifecycle:
  create_collection(collection_name)  — on folder creation
  drop_collection(collection_name)    — on folder deletion
  upsert_chunks(collection_name, ...) — on document ingestion
  search(collection_name, ...)        — on query
  list_documents(collection_name)     — per folder
  delete_document(collection_name, source_file) — per document

Schema per point (unchanged):
  id          — UUID (chunk_id)
  vector      — float list
  payload:
    text, context, enriched_text, source_file, doc_type,
    headings, page_numbers, is_table, table_id, chunk_id, ocr_applied
"""

import uuid
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import config
from embedder import get_embedding_dimension
from parser import ParsedChunk


def _get_client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


# ── Collection lifecycle ──────────────────────────────────────────────────────

def create_collection(collection_name: str):
    """Create a Qdrant collection for a folder. No-op if it already exists."""
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        logger.info(f"Collection '{collection_name}' already exists — skipping create")
        return

    dim = get_embedding_dimension()
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

    # Create payload indexes for filtered search performance
    client.create_payload_index(
        collection_name=collection_name,
        field_name="source_file",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_type",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="headings",
        field_schema=qmodels.PayloadSchemaType.TEXT,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="is_table",
        field_schema=qmodels.PayloadSchemaType.BOOL,
    )

    logger.info(f"Created Qdrant collection '{collection_name}' (dim={dim}) with payload indexes")


def drop_collection(collection_name: str) -> bool:
    """
    Delete a Qdrant collection and all its vectors.
    Returns True if deleted, False if it did not exist.
    """
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}

    if collection_name not in existing:
        logger.warning(f"Collection '{collection_name}' does not exist — nothing to drop")
        return False

    client.delete_collection(collection_name)
    logger.info(f"Dropped Qdrant collection '{collection_name}'")
    return True


def collection_exists(collection_name: str) -> bool:
    client = _get_client()
    return collection_name in {c.name for c in client.get_collections().collections}


# ── Data operations ───────────────────────────────────────────────────────────

def upsert_chunks(
    collection_name: str,
    chunks: list[ParsedChunk],
    enriched_texts: list[str],
    embeddings: list[list[float]],
):
    """Upsert chunks into the specified folder collection."""
    client = _get_client()
    points = []

    for chunk, enriched_text, embedding in zip(chunks, enriched_texts, embeddings):
        parts = enriched_text.split("\n\n", 1)
        context_sentence = parts[0] if len(parts) == 2 else ""

        payload = {
            "text": chunk.text,
            "context": context_sentence,
            "enriched_text": enriched_text,
            "source_file": chunk.source_file,
            "doc_type": chunk.doc_type,
            "headings": chunk.headings,
            "page_numbers": chunk.page_numbers,
            "is_table": chunk.is_table,
            "table_id": chunk.table_id,
            "chunk_id": chunk.chunk_id,
            "ocr_applied": chunk.ocr_applied,
        }

        points.append(
            qmodels.PointStruct(
                id=str(uuid.UUID(chunk.chunk_id)),
                vector=embedding,
                payload=payload,
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i:i + batch_size],
        )

    logger.info(f"Upserted {len(points)} chunks into '{collection_name}'")


def search(
    collection_name: str,
    query_vector: list[float],
    top_k: int = 8,
    source_file: Optional[str] = None,
    doc_type: Optional[str] = None,
    headings_contain: Optional[str] = None,
) -> list[dict]:
    """
    Semantic search within a single folder collection.
    All results are guaranteed to come from this collection only.
    """
    client = _get_client()

    must_conditions = []

    if source_file:
        must_conditions.append(
            qmodels.FieldCondition(
                key="source_file",
                match=qmodels.MatchValue(value=source_file),
            )
        )

    if doc_type:
        must_conditions.append(
            qmodels.FieldCondition(
                key="doc_type",
                match=qmodels.MatchValue(value=doc_type),
            )
        )

    if headings_contain:
        must_conditions.append(
            qmodels.FieldCondition(
                key="headings",
                match=qmodels.MatchText(text=headings_contain),
            )
        )

    query_filter = (
        qmodels.Filter(must=must_conditions) if must_conditions else None
    )

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score": round(hit.score, 4),
            "text": hit.payload.get("text", ""),
            "context": hit.payload.get("context", ""),
            "source_file": hit.payload.get("source_file", ""),
            "doc_type": hit.payload.get("doc_type", ""),
            "headings": hit.payload.get("headings", []),
            "page_numbers": hit.payload.get("page_numbers", []),
            "is_table": hit.payload.get("is_table", False),
            "table_id": hit.payload.get("table_id"),
            "chunk_id": hit.payload.get("chunk_id", ""),
            "ocr_applied": hit.payload.get("ocr_applied", False),
        }
        for hit in results
    ]


def list_documents(collection_name: str) -> list[dict]:
    """Return deduplicated documents in a folder collection with basic stats."""
    client = _get_client()

    if not collection_exists(collection_name):
        return []

    seen = {}
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=500,
            offset=offset,
            with_payload=["source_file", "doc_type", "is_table"],
            with_vectors=False,
        )

        for record in records:
            sf = record.payload.get("source_file", "unknown")
            if sf not in seen:
                seen[sf] = {
                    "source_file": sf,
                    "doc_type": record.payload.get("doc_type", "unknown"),
                    "chunk_count": 0,
                    "table_count": 0,
                }
            seen[sf]["chunk_count"] += 1
            if record.payload.get("is_table"):
                seen[sf]["table_count"] += 1

        if next_offset is None:
            break
        offset = next_offset

    return list(seen.values())


def get_table_ids(collection_name: str, source_file: str) -> list[str]:
    """Return all table_ids for a given document (for cleanup on delete)."""
    client = _get_client()
    table_ids: set[str] = set()
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_file",
                        match=qmodels.MatchValue(value=source_file),
                    ),
                    qmodels.FieldCondition(
                        key="is_table",
                        match=qmodels.MatchValue(value=True),
                    ),
                ]
            ),
            limit=500,
            offset=offset,
            with_payload=["table_id"],
            with_vectors=False,
        )
        for record in records:
            tid = record.payload.get("table_id")
            if tid:
                table_ids.add(tid)
        if next_offset is None:
            break
        offset = next_offset

    return list(table_ids)


def delete_document(collection_name: str, source_file: str) -> int:
    """Delete all chunks for a document. Returns points deleted."""
    client = _get_client()

    count_before = client.count(
        collection_name=collection_name,
        count_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="source_file",
                    match=qmodels.MatchValue(value=source_file),
                )
            ]
        ),
    ).count

    client.delete(
        collection_name=collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_file",
                        match=qmodels.MatchValue(value=source_file),
                    )
                ]
            )
        ),
    )

    logger.info(
        f"Deleted {count_before} chunks for '{source_file}' "
        f"from '{collection_name}'"
    )
    return count_before
