"""
vector_store.py — Qdrant operations with per-folder collection isolation

Each project/tender folder has its own Qdrant collection.
This gives hard query isolation — a search against Project 269 will never
return results from Project 245, even if they share terminology.

Collection lifecycle:
  create_collection(collection_name)  — on folder creation
  drop_collection(collection_name)    — on folder deletion
  upsert_chunks(collection_name, ...) — on document ingestion
  search(collection_name, ...)        — on query (hybrid: vector + BM25)
  list_documents(collection_name)     — per folder
  delete_document(collection_name, source_file) — per document

Schema per point (unchanged):
  id          — UUID (chunk_id)
  vector      — float list
  payload:
    text, context, enriched_text, source_file, doc_type,
    headings, page_numbers, is_table, table_id, chunk_id, ocr_applied
"""

import math
import re
import uuid
from collections import Counter
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import config
from embedder import get_embedding_dimension
from parser import ParsedChunk


# ── BM25 scoring for hybrid search ──────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """
    Simplified BM25 score (without IDF — uses query term frequency only).
    Good enough for re-ranking a small candidate set from vector search.
    """
    dl = len(doc_tokens)
    if dl == 0 or avg_dl == 0:
        return 0.0
    tf_map = Counter(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf > 0:
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
            score += numerator / denominator
    return score


def _hybrid_rerank(query: str, results: list[dict], top_k: int,
                   rrf_k: int = 60) -> list[dict]:
    """
    Re-rank vector search results using Reciprocal Rank Fusion (RRF)
    of vector score ranking and BM25 keyword ranking.

    RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)
    """
    if not results:
        return results

    query_tokens = _tokenize(query)
    if not query_tokens:
        return results[:top_k]

    # Tokenize all result texts (use enriched_text for full content)
    doc_tokens_list = []
    for r in results:
        text = r.get("enriched_text", r.get("text", ""))
        headings = " ".join(r.get("headings", []))
        doc_tokens_list.append(_tokenize(headings + " " + text))

    avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(len(doc_tokens_list), 1)

    # Compute BM25 scores
    bm25_scores = []
    for dt in doc_tokens_list:
        bm25_scores.append(_bm25_score(query_tokens, dt, avg_dl))

    # Build rankings
    vector_rank = {i: rank for rank, i in enumerate(range(len(results)))}

    bm25_order = sorted(range(len(results)), key=lambda i: -bm25_scores[i])
    bm25_rank = {i: rank for rank, i in enumerate(bm25_order)}

    # RRF fusion
    rrf_scores = []
    for i in range(len(results)):
        rrf = 1.0 / (rrf_k + vector_rank[i]) + 1.0 / (rrf_k + bm25_rank[i])
        rrf_scores.append((i, rrf))

    rrf_scores.sort(key=lambda x: -x[1])

    reranked = []
    for i, rrf_score in rrf_scores[:top_k]:
        r = dict(results[i])
        r["rrf_score"] = round(rrf_score, 6)
        r["bm25_rank"] = bm25_rank[i] + 1
        r["vector_rank"] = vector_rank[i] + 1
        reranked.append(r)

    return reranked


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
    query_text: str = "",
) -> list[dict]:
    """
    Hybrid search within a single folder collection.
    Uses vector similarity for semantic matching, then re-ranks with BM25
    keyword scoring via Reciprocal Rank Fusion (RRF).
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

    # Fetch a broad candidate set for hybrid re-ranking
    fetch_limit = max(top_k * 5, 30)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=fetch_limit,
        with_payload=True,
    )

    # Deduplicate: skip chunks whose text is ≥90% identical to an already-seen chunk
    seen_texts: list[str] = []
    deduped: list[dict] = []

    for hit in results.points:
        text = hit.payload.get("text", "")
        # Check for near-duplicate (simple substring overlap)
        is_dup = False
        for seen in seen_texts:
            shorter, longer = (text, seen) if len(text) <= len(seen) else (seen, text)
            if shorter and shorter in longer:
                is_dup = True
                break
        if is_dup:
            continue

        seen_texts.append(text)
        deduped.append({
            "score": round(hit.score, 4),
            "text": text,
            "context": hit.payload.get("context", ""),
            "enriched_text": hit.payload.get("enriched_text", ""),
            "source_file": hit.payload.get("source_file", ""),
            "doc_type": hit.payload.get("doc_type", ""),
            "headings": hit.payload.get("headings", []),
            "page_numbers": hit.payload.get("page_numbers", []),
            "is_table": hit.payload.get("is_table", False),
            "table_id": hit.payload.get("table_id"),
            "chunk_id": hit.payload.get("chunk_id", ""),
            "ocr_applied": hit.payload.get("ocr_applied", False),
        })

    # Apply BM25 hybrid re-ranking if query text is provided
    if query_text and deduped:
        deduped = _hybrid_rerank(query_text, deduped, top_k)
    else:
        deduped = deduped[:top_k]

    # Remove enriched_text from output (only needed for BM25 scoring)
    for r in deduped:
        r.pop("enriched_text", None)

    return deduped


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
