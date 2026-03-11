"""
ingestor.py — orchestrates the full ingestion pipeline into a specific folder

Flow:
  1. Resolve folder -> get Qdrant collection name
  2. Parse document with Docling (returns both chunks AND the DoclingDocument)
  3. Build section tree from DoclingDocument + generate summaries via Claude haiku
  4. Enrich each chunk with context sentence (Contextual Retrieval)
  5. Embed enriched texts (sentence-transformers, local)
  6. Upsert into folder's Qdrant collection

For large PDFs (>SPLIT_THRESHOLD pages), ingest_with_split() automatically:
  - Runs the EPC Tender PDF Splitter (10-engine heuristic, no LLM)
  - Ingests each sub-document individually through the full pipeline
  - Tracks the parent->parts relationship via a manifest for re-ingest/delete

Progress tracking:
  When a job_id is provided, emits real-time progress events via the
  ProgressTracker. The admin SSE endpoint streams these to the GUI.
"""

import time
from pathlib import Path

from loguru import logger

from config import config
from contextualizer import enrich_chunks_with_context
from embedder import embed_documents
from folder_manager import get_folder, update_folder_doc_count
from parser import parse_document_with_doc
from pdf_split import (
    should_split,
    split_pdf,
    save_manifest,
    load_manifest,
    delete_manifest,
    cleanup_split_dir,
    get_page_count,
)
from progress import tracker
from tree_builder import build_tree
from tree_store import save_tree, delete_tree
from vector_store import delete_document, get_table_ids, upsert_chunks


# Step weights for progress calculation (sum to 1.0)
_STEP_WEIGHTS = {
    "parse": 0.10,
    "tree": 0.15,
    "contextualise": 0.45,
    "embed": 0.20,
    "upsert": 0.10,
}

# Cumulative progress at the START of each step
_STEP_STARTS = {}
_cumulative = 0.0
for _step, _weight in _STEP_WEIGHTS.items():
    _STEP_STARTS[_step] = _cumulative
    _cumulative += _weight


class IngestionCancelledError(Exception):
    """Raised when an ingestion job is cancelled by the user."""
    pass


def _emit(job_id, event_type, **kwargs):
    """Emit a progress event if job_id is set."""
    if job_id:
        tracker.emit(job_id, event_type, **kwargs)


def _check_cancel(job_id):
    """Raise IngestionCancelledError if job has been cancelled."""
    if job_id and tracker.is_cancelled(job_id):
        raise IngestionCancelledError(f"Job {job_id} cancelled")


def ingest(
    file_path: str | Path,
    folder_id: str,
    job_id: str | None = None,
    _progress_base: float = 0.0,
    _progress_scale: float = 1.0,
) -> dict:
    """
    Ingest a single document into a specific project/tender folder.

    Args:
        file_path: Path to a PDF or DOCX file
        folder_id: The folder ID to ingest into
        job_id: Optional job ID for progress tracking
        _progress_base: Internal — base progress offset (for split parts)
        _progress_scale: Internal — progress fraction for this part (for split parts)

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
        f"=== Ingesting: {file_path.name} -> "
        f"folder '{folder['name']}' ({collection_name}) ==="
    )

    def _step_progress(step_name, fraction=0.0):
        """Compute overall progress for a step within this part."""
        step_start = _STEP_STARTS.get(step_name, 0.0)
        step_weight = _STEP_WEIGHTS.get(step_name, 0.0)
        local = step_start + step_weight * fraction
        return _progress_base + local * _progress_scale

    # ── Step 1: Parse ────────────────────────────────────────────────────
    _check_cancel(job_id)
    _emit(job_id, "step", phase="parse",
          message=f"Parsing {file_path.name}...",
          progress=_step_progress("parse"))

    t0 = time.time()
    chunks, docling_doc = parse_document_with_doc(file_path)
    parse_time = time.time() - t0
    logger.info(f"Step 1 (parse): {len(chunks)} chunks in {parse_time:.1f}s")

    if not chunks:
        logger.warning("No chunks extracted — document may be empty or unreadable")
        _emit(job_id, "step_done", phase="parse",
              message="No content extracted", detail="0 chunks",
              step_seconds=round(parse_time, 1))
        return {"status": "warning", "message": "No content extracted", "chunks": 0}

    table_count = sum(1 for c in chunks if c.is_table)
    _emit(job_id, "step_done", phase="parse",
          message=f"Parsed: {len(chunks)} chunks ({table_count} tables)",
          detail=f"{len(chunks)} chunks",
          step_seconds=round(parse_time, 1),
          progress=_step_progress("tree"))

    # Step 1b: Remove existing data for this document (handles re-ingest)
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

    # ── Step 2: Build section tree ───────────────────────────────────────
    _check_cancel(job_id)
    _emit(job_id, "step", phase="tree",
          message="Building section tree...",
          progress=_step_progress("tree"))

    t0 = time.time()
    tree_nodes = 0
    try:
        def _tree_progress(done, total):
            frac = done / total if total else 0
            _emit(job_id, "step_progress", phase="tree",
                  message=f"Summarising sections {done}/{total}...",
                  step_done=done, step_total=total,
                  progress=_step_progress("tree", frac))

        tree = build_tree(docling_doc, file_path.name, show_progress=True,
                          on_progress=_tree_progress,
                          cancel_check=lambda: _check_cancel(job_id))
        save_tree(folder_id, tree)
        tree_time = time.time() - t0
        tree_nodes = sum(1 for _ in _iter_nodes(tree["nodes"]))
        logger.info(f"Step 2 (tree):  {tree_nodes} nodes in {tree_time:.1f}s")
    except Exception as e:
        logger.warning(f"Step 2 (tree): failed — {e} — continuing without tree")
        tree_time = 0.0

    _emit(job_id, "step_done", phase="tree",
          message=f"Tree: {tree_nodes} nodes",
          detail=f"{tree_nodes} nodes",
          step_seconds=round(tree_time, 1),
          progress=_step_progress("contextualise"))

    # Step 2b: Use tree to assign better section headings to chunks
    try:
        _enrich_headings_from_tree(chunks, tree)
    except Exception as e:
        logger.warning(f"Step 2b (tree headings): failed — {e}")

    # ── Step 3: Contextual Retrieval enrichment ──────────────────────────
    _check_cancel(job_id)
    _emit(job_id, "step", phase="contextualise",
          message=f"Enriching {len(chunks)} chunks with context...",
          progress=_step_progress("contextualise"))

    t0 = time.time()

    def _context_progress(done, total):
        frac = done / total if total else 0
        _emit(job_id, "step_progress", phase="contextualise",
              message=f"Contextualising {done}/{total} chunks...",
              step_done=done, step_total=total,
              progress=_step_progress("contextualise", frac))

    enriched_pairs = enrich_chunks_with_context(
        chunks, show_progress=True, on_progress=_context_progress,
        cancel_check=lambda: _check_cancel(job_id))
    context_time = time.time() - t0
    logger.info(f"Step 3 (contextualise): {len(enriched_pairs)} chunks in {context_time:.1f}s")

    _emit(job_id, "step_done", phase="contextualise",
          message=f"Contextualised {len(enriched_pairs)} chunks",
          detail=f"{len(enriched_pairs)} chunks",
          step_seconds=round(context_time, 1),
          progress=_step_progress("embed"))

    enriched_texts = [text for _, text in enriched_pairs]

    # ── Step 4: Embed ────────────────────────────────────────────────────
    _check_cancel(job_id)
    _emit(job_id, "step", phase="embed",
          message=f"Embedding {len(enriched_texts)} chunks...",
          progress=_step_progress("embed"))

    t0 = time.time()
    embeddings = embed_documents(enriched_texts)
    embed_time = time.time() - t0
    logger.info(f"Step 4 (embed): {len(embeddings)} vectors in {embed_time:.1f}s")

    _emit(job_id, "step_done", phase="embed",
          message=f"Embedded {len(embeddings)} vectors",
          detail=f"{len(embeddings)} vectors",
          step_seconds=round(embed_time, 1),
          progress=_step_progress("upsert"))

    # ── Step 5: Upsert ──────────────────────────────────────────────────
    _check_cancel(job_id)
    _emit(job_id, "step", phase="upsert",
          message="Upserting into vector store...",
          progress=_step_progress("upsert"))

    t0 = time.time()
    upsert_chunks(collection_name, chunks, enriched_texts, embeddings)
    upsert_time = time.time() - t0
    logger.info(f"Step 5 (upsert): done in {upsert_time:.1f}s")

    _emit(job_id, "step_done", phase="upsert",
          message="Upserted into vector store",
          detail="done",
          step_seconds=round(upsert_time, 1),
          progress=_progress_base + _progress_scale)

    update_folder_doc_count(folder_id, delta=1)

    total_time = time.time() - start
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
        f"=== Done: {file_path.name} -> '{folder['name']}' — "
        f"{len(chunks)} chunks, {tree_nodes} tree nodes in {total_time:.1f}s ==="
    )
    return summary


def _iter_nodes(nodes):
    for n in nodes:
        yield n
        yield from _iter_nodes(n.get("children", []))


def _enrich_headings_from_tree(chunks, tree):
    """
    Use the section tree to give chunks better heading breadcrumbs.

    The tree walker sees headings that HybridChunker may have missed (e.g.
    'TECHNICAL CRITERIA:-') because the tree walks the full Docling body.
    For each chunk we find the best-matching tree node by page overlap or
    content overlap (for pageless tables) and prepend any ancestor section
    titles that the chunk is missing.
    """
    if not tree or not tree.get("nodes"):
        return

    # Build a list of (breadcrumb, page_range, full_text) from the tree
    tree_sections = []

    def _collect(nodes, ancestors=None):
        if ancestors is None:
            ancestors = []
        for node in nodes:
            ps = node.get("page_start")
            pe = node.get("page_end")
            breadcrumb = ancestors + [node["title"]]
            tree_sections.append({
                "breadcrumb": breadcrumb,
                "page_start": ps,
                "page_end": pe,
                "full_text": node.get("full_text", ""),
            })
            _collect(node.get("children", []), breadcrumb)

    _collect(tree["nodes"])

    for chunk in chunks:
        best_match = None
        best_score = 0

        for sec in tree_sections:
            score = 0
            # Page overlap scoring
            if chunk.page_numbers and sec["page_start"] and sec["page_end"]:
                page_set = set(range(sec["page_start"], sec["page_end"] + 1))
                overlap = len(set(chunk.page_numbers) & page_set)
                if overlap > 0:
                    score += overlap * 10

                    # Bonus for matching heading text
                    if chunk.headings:
                        chunk_hdg = chunk.headings[-1].lower().strip("*: ")
                        sec_title = sec["breadcrumb"][-1].lower().strip("*: ")
                        if chunk_hdg == sec_title or chunk_hdg in sec_title:
                            score += 50

            # Content overlap for pageless chunks (e.g. orphan tables)
            elif not chunk.page_numbers and sec["full_text"]:
                # Check if chunk text appears in tree node content
                chunk_snippet = chunk.text[:200]
                if chunk_snippet and chunk_snippet[:80] in sec["full_text"]:
                    score += 100  # Strong match

            # Prefer deeper (more specific) breadcrumbs
            score += len(sec["breadcrumb"])

            if score > best_score:
                best_score = score
                best_match = sec

        if best_match and best_score > 5:
            # Build a clean breadcrumb from the tree match
            tree_bc = list(best_match["breadcrumb"])

            # For content-matched chunks (pageless tables), also collect
            # nearby tree sections on the same pages to capture parent
            # headings like "TECHNICAL CRITERIA" that precede the match
            if best_score >= 100 and best_match["page_start"]:
                matched_idx = tree_sections.index(best_match)
                for j in range(matched_idx - 1, max(matched_idx - 5, -1), -1):
                    nearby = tree_sections[j]
                    if (nearby["page_start"] and nearby["page_end"]
                            and nearby["page_start"] <= best_match["page_start"]
                            and nearby["page_end"] >= best_match["page_start"]):
                        for h in nearby["breadcrumb"]:
                            if h not in tree_bc:
                                tree_bc.insert(0, h)

            # Filter out generic/noisy headings (CRFQ numbers, tender titles)
            filtered = [h for h in tree_bc
                        if not h.lower().startswith(("crfq", "tender for"))]
            if not filtered:
                filtered = tree_bc

            # For strong tree matches (content or page+heading), REPLACE
            # chunk headings entirely with tree breadcrumb (max 3 entries)
            if best_score >= 50:
                chunk.headings = filtered[-3:]
            else:
                # Weak match: only prepend missing tree headings
                existing = set(h.lower().strip("*: ") for h in chunk.headings)
                new_headings = [h for h in filtered
                                if h.lower().strip("*: ") not in existing]
                if new_headings:
                    chunk.headings = new_headings[-2:] + chunk.headings

            logger.debug(
                f"  Tree enriched: {chunk.chunk_id[:8]} -> {chunk.headings}"
            )


def _delete_split_parts(folder_id: str, collection_name: str, manifest: dict) -> int:
    """
    Delete all split parts of a previously-ingested large PDF.
    Returns the total number of chunks removed.
    """
    total_removed = 0
    for part in manifest.get("parts", []):
        part_file = part["filename"]
        # Collect table artifact IDs before deleting
        old_table_ids = get_table_ids(collection_name, part_file)
        count = delete_document(collection_name, part_file)
        if count > 0:
            total_removed += count
            update_folder_doc_count(folder_id, delta=-1)
        # Clean up table artifacts
        for tid in old_table_ids:
            artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
            if artifact_path.exists():
                artifact_path.unlink()
        # Clean up tree
        delete_tree(folder_id, part_file)

    logger.info(
        f"  Cleaned up {len(manifest['parts'])} old split parts "
        f"({total_removed} chunks removed)"
    )
    return total_removed


def _cleanup_single_doc(folder_id: str, collection_name: str, filename: str):
    """Clean up a partially-ingested single document (vectors, tree, table artifacts)."""
    try:
        table_ids = get_table_ids(collection_name, filename)
        count = delete_document(collection_name, filename)
        if count > 0:
            update_folder_doc_count(folder_id, delta=-1)
            logger.info(f"  Cancel cleanup: removed {count} chunks for {filename}")
        for tid in table_ids:
            artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
            if artifact_path.exists():
                artifact_path.unlink()
        delete_tree(folder_id, filename)
    except Exception as e:
        logger.warning(f"  Cancel cleanup error for {filename}: {e}")


def ingest_with_split(
    file_path: str | Path,
    folder_id: str,
    job_id: str | None = None,
) -> dict:
    """
    Smart ingestion: automatically splits large PDFs before ingesting.

    For PDFs with fewer than SPLIT_THRESHOLD pages (and all DOCX files),
    delegates directly to ingest().

    For large PDFs:
      1. Runs the EPC Tender PDF Splitter (10-engine heuristic, no LLM)
      2. Cleans up any previous split parts (for re-ingest)
      3. Ingests each sub-document through the full pipeline
      4. Saves a manifest tracking parent->parts relationship
      5. Cleans up temporary split files

    Args:
        file_path: Path to a PDF or DOCX file
        folder_id: The folder ID to ingest into
        job_id: Optional job ID for progress tracking

    Returns:
        Summary dict with per-part results and aggregate stats
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not should_split(file_path):
        # Small PDF or non-PDF — clean up any stale manifest and ingest directly
        old_manifest = load_manifest(folder_id, file_path.name)
        if old_manifest:
            folder = get_folder(folder_id)
            _delete_split_parts(folder_id, folder["collection_name"], old_manifest)
            delete_manifest(folder_id, file_path.name)
        return ingest(file_path, folder_id, job_id=job_id)

    # ── Large PDF: split first ───────────────────────────────────────────
    folder = get_folder(folder_id)
    collection_name = folder["collection_name"]
    start = time.time()

    page_count = get_page_count(file_path)
    logger.info(
        f"=== Large PDF detected: {file_path.name} ({page_count} pages) — "
        f"splitting before ingestion ==="
    )
    _check_cancel(job_id)
    _emit(job_id, "splitting",
          message=f"Splitting {file_path.name} ({page_count} pages)...",
          progress=0.0, total_pages=page_count)

    # Clean up previous split parts if re-ingesting
    old_manifest = load_manifest(folder_id, file_path.name)
    if old_manifest:
        logger.info(
            f"  Re-ingest: removing {old_manifest['total_parts']} "
            f"previous split parts"
        )
        _delete_split_parts(folder_id, collection_name, old_manifest)
        delete_manifest(folder_id, file_path.name)

    # Split the PDF
    t0 = time.time()
    parts, split_dir = split_pdf(file_path)
    split_time = time.time() - t0
    logger.info(f"  Split into {len(parts)} parts in {split_time:.1f}s")

    if not parts:
        _emit(job_id, "error", message="Splitter produced no parts")
        return {
            "status": "warning",
            "message": "Splitter produced no parts",
            "source_file": file_path.name,
        }

    # If only one part (splitter decided not to split), ingest directly
    if len(parts) == 1 and parts[0]["filename"] == file_path.name:
        cleanup_split_dir(split_dir)
        return ingest(file_path, folder_id, job_id=job_id)

    _emit(job_id, "split_done",
          message=f"Split into {len(parts)} parts",
          total_parts=len(parts),
          detail=f"{len(parts)} parts",
          step_seconds=round(split_time, 1),
          progress=0.05)

    # Ingest each part
    SPLIT_WEIGHT = 0.05
    per_part_weight = (1.0 - SPLIT_WEIGHT) / len(parts)

    part_results = []
    cancelled = False
    for i, part in enumerate(parts, 1):
        # Check cancellation before each part
        if job_id and tracker.is_cancelled(job_id):
            cancelled = True
            logger.info(f"  Job {job_id} cancelled before part {i}/{len(parts)}")
            break

        part_base = SPLIT_WEIGHT + ((i - 1) * per_part_weight)

        logger.info(
            f"\n--- Ingesting part {i}/{len(parts)}: "
            f"{part['filename']} ({part['num_pages']} pages, "
            f"pp. {part['start_page']}-{part['end_page']}) ---"
        )

        _emit(job_id, "part_start",
              message=f"Part {i}/{len(parts)}: {part['filename']}",
              current_part=i, total_parts=len(parts),
              part_file=part["filename"],
              progress=part_base)

        try:
            result = ingest(
                part["path"], folder_id,
                job_id=job_id,
                _progress_base=part_base,
                _progress_scale=per_part_weight,
            )
            result["_split_part"] = {
                "part_number": i,
                "original_file": file_path.name,
                "original_pages": f"{part['start_page']}-{part['end_page']}",
                "label": part.get("label", ""),
            }
            part_results.append(result)

            _emit(job_id, "part_done",
                  message=f"Part {i}/{len(parts)} done: {result.get('total_chunks', 0)} chunks",
                  current_part=i, total_parts=len(parts),
                  part_file=part["filename"],
                  chunks=result.get("total_chunks", 0),
                  progress=part_base + per_part_weight)

        except IngestionCancelledError:
            cancelled = True
            logger.info(f"  Job {job_id} cancelled during part {i}/{len(parts)}")
            # Clean up the partially-ingested current part
            _cleanup_single_doc(folder_id, collection_name, part["filename"])
            break

        except Exception as e:
            logger.error(f"  Part {i} failed: {e}")
            _emit(job_id, "error",
                  message=f"Part {i} failed: {e}",
                  current_part=i, part_file=part["filename"])
            part_results.append({
                "status": "error",
                "source_file": part["filename"],
                "error": str(e),
            })

    if cancelled:
        # Clean up all successfully-ingested parts
        for r in part_results:
            if r.get("status") == "success":
                _cleanup_single_doc(folder_id, collection_name, r["source_file"])
        # Clean up temporary split files
        cleanup_split_dir(split_dir)
        raise IngestionCancelledError(f"Job {job_id} cancelled")

    # Save manifest for future re-ingest/delete operations
    save_manifest(folder_id, file_path.name, parts)

    # Clean up temporary split files
    cleanup_split_dir(split_dir)

    total_time = time.time() - start
    success_count = sum(1 for r in part_results if r.get("status") == "success")
    total_chunks = sum(r.get("total_chunks", 0) for r in part_results)
    total_tables = sum(r.get("table_chunks", 0) for r in part_results)
    total_tree_nodes = sum(r.get("tree_nodes", 0) for r in part_results)

    summary = {
        "status": "success" if success_count == len(parts) else "partial",
        "folder_id": folder["folder_id"],
        "folder_name": folder["name"],
        "source_file": file_path.name,
        "split": True,
        "total_parts": len(parts),
        "parts_succeeded": success_count,
        "parts_failed": len(parts) - success_count,
        "total_chunks": total_chunks,
        "table_chunks": total_tables,
        "tree_nodes": total_tree_nodes,
        "total_time_seconds": round(total_time, 1),
        "breakdown": {
            "split_seconds": round(split_time, 1),
        },
        "parts": part_results,
    }

    logger.info(
        f"\n=== Done: {file_path.name} -> '{folder['name']}' — "
        f"{len(parts)} parts, {total_chunks} chunks, "
        f"{total_tree_nodes} tree nodes in {total_time:.1f}s ==="
    )
    return summary
