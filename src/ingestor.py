"""
ingestor.py — orchestrates the full ingestion pipeline into a specific folder

Flow:
  1. Resolve folder → get Qdrant collection name
  2. Parse document with Docling (returns both chunks AND the DoclingDocument)
  3. Build section tree from DoclingDocument + generate summaries via Claude haiku
  4. Enrich each chunk with context sentence (Contextual Retrieval)
  5. Embed enriched texts (sentence-transformers, local)
  6. Upsert into folder's Qdrant collection

The tree (step 3) is saved to disk and used at deep_query time for
section-level reasoning. The vector index (steps 4-6) is used at regular
query time for fast coarse retrieval across the corpus.

Both pipelines run on the same Docling parse, so the document is only
converted once.
"""

import time
from pathlib import Path

from loguru import logger

from config import config
from contextualizer import enrich_chunks_with_context
from embedder import embed_documents
from folder_manager import get_folder, update_folder_doc_count
from parser import parse_document_with_doc
from tree_builder import build_tree
from tree_store import save_tree, delete_tree
from vector_store import delete_document, get_table_ids, upsert_chunks


def ingest(file_path: str | Path, folder_id: str) -> dict:
    """
    Ingest a single document into a specific project/tender folder.

    Args:
        file_path: Path to a PDF or DOCX file
        folder_id: The folder ID to ingest into

    Returns:
        Summary dict with counts, timing, and folder info
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() not in {".pdf", ".docx", ".doc"}:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            "Supported: .pdf, .docx, .doc"
        )

    folder = get_folder(folder_id)
    collection_name = folder["collection_name"]

    start = time.time()
    logger.info(
        f"=== Ingesting: {file_path.name} → "
        f"folder '{folder['name']}' ({collection_name}) ==="
    )

    # Step 1: Parse document — returns chunks AND DoclingDocument
    t0 = time.time()
    chunks, docling_doc = parse_document_with_doc(file_path)
    parse_time = time.time() - t0
    logger.info(f"Step 1 (parse): {len(chunks)} chunks in {parse_time:.1f}s")

    if not chunks:
        logger.warning("No chunks extracted — document may be empty or unreadable")
        return {"status": "warning", "message": "No content extracted", "chunks": 0}

    # Step 1b: Remove existing data for this document (handles re-ingest)
    # Collect table IDs before deleting chunks from Qdrant
    old_table_ids = get_table_ids(collection_name, file_path.name)
    existing_count = delete_document(collection_name, file_path.name)
    if existing_count > 0:
        logger.info(f"  Re-ingest: removed {existing_count} existing chunks for {file_path.name}")
        for tid in old_table_ids:
            artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
            if artifact_path.exists():
                artifact_path.unlink()
        delete_tree(folder_id, file_path.name)
        update_folder_doc_count(folder_id, delta=-1)

    # Step 2: Build section tree from the DoclingDocument
    t0 = time.time()
    try:
        tree = build_tree(docling_doc, file_path.name, show_progress=True)
        save_tree(folder_id, tree)
        tree_time = time.time() - t0
        tree_nodes = sum(1 for _ in _iter_nodes(tree["nodes"]))
        logger.info(f"Step 2 (tree):  {tree_nodes} nodes in {tree_time:.1f}s")
    except Exception as e:
        logger.warning(f"Step 2 (tree): failed — {e} — continuing without tree")
        tree_time = 0.0
        tree_nodes = 0

    # Step 3: Contextual Retrieval enrichment
    t0 = time.time()
    enriched_pairs = enrich_chunks_with_context(chunks, show_progress=True)
    context_time = time.time() - t0
    logger.info(f"Step 3 (contextualise): {len(enriched_pairs)} chunks in {context_time:.1f}s")

    enriched_texts = [text for _, text in enriched_pairs]

    # Step 4: Embed
    t0 = time.time()
    embeddings = embed_documents(enriched_texts)
    embed_time = time.time() - t0
    logger.info(f"Step 4 (embed): {len(embeddings)} vectors in {embed_time:.1f}s")

    # Step 5: Upsert into folder collection
    t0 = time.time()
    upsert_chunks(collection_name, chunks, enriched_texts, embeddings)
    upsert_time = time.time() - t0
    logger.info(f"Step 5 (upsert): done in {upsert_time:.1f}s")

    update_folder_doc_count(folder_id, delta=1)

    total_time = time.time() - start
    table_count = sum(1 for c in chunks if c.is_table)
    ocr_count = sum(1 for c in chunks if c.ocr_applied)

    summary = {
        "status": "success",
        "folder_id": folder["folder_id"],
        "folder_name": folder["name"],
        "source_file": file_path.name,
        "doc_type": chunks[0].doc_type if chunks else "unknown",
        "total_chunks": len(chunks),
        "table_chunks": table_count,
        "text_chunks": len(chunks) - table_count,
        "ocr_chunks": ocr_count,
        "tree_nodes": tree_nodes,
        "total_time_seconds": round(total_time, 1),
        "breakdown": {
            "parse_seconds": round(parse_time, 1),
            "tree_seconds": round(tree_time, 1),
            "context_seconds": round(context_time, 1),
            "embed_seconds": round(embed_time, 1),
            "upsert_seconds": round(upsert_time, 1),
        },
    }

    logger.info(
        f"=== Done: {file_path.name} → '{folder['name']}' — "
        f"{len(chunks)} chunks, {tree_nodes} tree nodes in {total_time:.1f}s ==="
    )
    return summary


def _iter_nodes(nodes):
    for n in nodes:
        yield n
        yield from _iter_nodes(n.get("children", []))
