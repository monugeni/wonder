"""
admin.py — REST API + static admin page for folder/document management

Mounted alongside the MCP SSE server at /admin/*.
Provides CRUD for folders and documents, file upload for ingestion,
real-time progress streaming via SSE, and raw vector search.

Endpoints:
  GET  /admin/                              — admin HTML page
  GET  /admin/api/folders                   — list all folders
  POST /admin/api/folders                   — create a folder
  DELETE /admin/api/folders/{id}            — delete a folder
  GET  /admin/api/folders/{id}/documents    — list documents in a folder
  DELETE /admin/api/folders/{id}/documents/{filename} — delete a document
  POST /admin/api/folders/{id}/ingest       — upload + start ingestion (returns job_id)
  GET  /admin/api/jobs                      — list all ingestion jobs
  GET  /admin/api/jobs/{job_id}             — get job status (polling)
  GET  /admin/api/jobs/{job_id}/events      — SSE stream of progress events
  POST /admin/api/query                     — raw vector search (no LLM)
"""

import asyncio
import json
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from embedder import embed_query
from folder_manager import (
    create_folder,
    delete_folder,
    get_folder,
    list_folders,
    rename_folder,
    resolve_folder,
    update_folder_doc_count,
)
from ingestor import IngestionCancelledError, ingest_with_split
from pdf_split import delete_all_manifests, delete_manifest, load_manifest, rename_manifest
from progress import tracker
from tree_store import (
    delete_all_trees,
    delete_tree,
    get_doc_summary,
    get_node_by_id,
    list_trees,
    load_tree,
    rename_tree,
    tree_summary_view,
)
from vector_store import (
    create_collection,
    delete_document,
    drop_collection,
    get_table_ids,
    list_documents,
    rename_document,
    search,
)


ADMIN_HTML = Path(__file__).parent / "static" / "admin.html"

# Thread pool for background ingestion tasks
_ingest_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")


# ── Background ingestion runner ──────────────────────────────────────────────

def _run_ingest_job(file_path: Path, folder_id: str, job_id: str):
    """Run ingestion in a background thread with progress tracking."""
    try:
        summary = ingest_with_split(str(file_path), folder_id, job_id=job_id)
        tracker.emit(job_id, "done", message="Ingestion complete", summary=summary)
    except IngestionCancelledError:
        logger.info(f"Job {job_id} was cancelled — cleanup complete")
        tracker.emit(job_id, "cancelled", message="Ingestion cancelled by user")
    except Exception as e:
        logger.exception(f"Background ingest failed: {e}")
        tracker.emit(job_id, "error", message=str(e))
    finally:
        # Clean up temp file
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass


# ── Handlers ─────────────────────────────────────────────────────────────────

async def index(request):
    return HTMLResponse(ADMIN_HTML.read_text())


async def api_list_folders(request):
    try:
        folders = list_folders()
        return JSONResponse({"folders": folders})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_create_folder(request):
    try:
        body = await request.json()
        name = body.get("name", "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)

        folder = create_folder(
            name=name,
            description=body.get("description", ""),
            tags=body.get("tags", []),
        )
        create_collection(folder["collection_name"])
        return JSONResponse({"folder": folder}, status_code=201)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_delete_folder(request):
    try:
        folder_id = request.path_params["folder_id"]
        folder = get_folder(folder_id)
        collection_name = folder["collection_name"]

        # Clean up table artifacts
        docs = list_documents(collection_name)
        for doc_info in docs:
            table_ids = get_table_ids(collection_name, doc_info["source_file"])
            for tid in table_ids:
                artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
                if artifact_path.exists():
                    artifact_path.unlink()

        drop_collection(collection_name)
        delete_all_trees(folder_id)
        delete_all_manifests(folder_id)
        delete_folder(folder_id)

        return JSONResponse({"status": "deleted", "folder_id": folder_id})
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_rename_folder(request):
    try:
        folder_id = request.path_params["folder_id"]
        body = await request.json()
        new_name = body.get("name", "").strip()
        if not new_name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        folder = rename_folder(folder_id, new_name)
        return JSONResponse({"folder": folder})
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_rename_document(request):
    try:
        folder_id = request.path_params["folder_id"]
        old_filename = request.path_params["filename"]
        body = await request.json()
        new_filename = body.get("name", "").strip()
        if not new_filename:
            return JSONResponse({"error": "name is required"}, status_code=400)

        folder = get_folder(folder_id)
        collection_name = folder["collection_name"]

        # Check if this is a split parent — rename the manifest, not parts
        manifest = load_manifest(folder_id, old_filename)
        if manifest:
            rename_manifest(folder_id, old_filename, new_filename)
            # Rename trees for each part
            for part in manifest["parts"]:
                rename_tree(folder_id, part["filename"], part["filename"])
            return JSONResponse({
                "status": "renamed",
                "old_name": old_filename,
                "new_name": new_filename,
            })

        # Single document — rename in Qdrant, tree, and table artifacts
        count = rename_document(collection_name, old_filename, new_filename)
        rename_tree(folder_id, old_filename, new_filename)

        return JSONResponse({
            "status": "renamed",
            "old_name": old_filename,
            "new_name": new_filename,
            "chunks_updated": count,
        })
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_list_documents(request):
    try:
        folder_id = request.path_params["folder_id"]
        folder = get_folder(folder_id)
        docs = list_documents(folder["collection_name"])
        # Enrich with document-level summary from tree store
        for d in docs:
            d["doc_summary"] = get_doc_summary(folder_id, d["source_file"]) or ""
        return JSONResponse({"folder": folder["name"], "documents": docs})
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_delete_document(request):
    try:
        folder_id = request.path_params["folder_id"]
        filename = request.path_params["filename"]
        folder = get_folder(folder_id)
        collection_name = folder["collection_name"]

        # Check if this is a split parent document
        manifest = load_manifest(folder_id, filename)
        if manifest:
            total_chunks = 0
            for part in manifest["parts"]:
                part_file = part["filename"]
                part_table_ids = get_table_ids(collection_name, part_file)
                count = delete_document(collection_name, part_file)
                if count > 0:
                    total_chunks += count
                    update_folder_doc_count(folder_id, delta=-1)
                for tid in part_table_ids:
                    artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
                    if artifact_path.exists():
                        artifact_path.unlink()
                delete_tree(folder_id, part_file)
            delete_manifest(folder_id, filename)
            return JSONResponse({
                "status": "deleted",
                "chunks_removed": total_chunks,
                "split_parts_removed": len(manifest["parts"]),
            })
        else:
            # Single document — original logic
            table_ids = get_table_ids(collection_name, filename)
            count = delete_document(collection_name, filename)
            for tid in table_ids:
                artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
                if artifact_path.exists():
                    artifact_path.unlink()
            delete_tree(folder_id, filename)
            if count > 0:
                update_folder_doc_count(folder_id, delta=-1)
            return JSONResponse({"status": "deleted", "chunks_removed": count})
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_ingest(request):
    """Upload a file and start background ingestion. Returns job_id immediately."""
    try:
        folder_id = request.path_params["folder_id"]
        folder = get_folder(folder_id)

        form = await request.form()
        upload = form.get("file")
        if not upload:
            return JSONResponse({"error": "No file uploaded"}, status_code=400)

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".doc"}:
            return JSONResponse(
                {"error": f"Unsupported file type: {suffix}. Use .pdf, .docx, or .doc"},
                status_code=400,
            )

        # Save upload to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Rename to preserve original filename
        final_path = Path(tmp_path).parent / upload.filename
        Path(tmp_path).rename(final_path)

        # Create a tracked job and start background ingestion
        job_id = tracker.create_job(upload.filename, folder["name"])
        _ingest_pool.submit(_run_ingest_job, final_path, folder_id, job_id)

        return JSONResponse(
            {"job_id": job_id, "status": "started", "source_file": upload.filename},
            status_code=202,
        )

    except (KeyError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception(f"Ingest error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Job status & SSE streaming ───────────────────────────────────────────────

async def api_list_jobs(request):
    """List all ingestion jobs (recent first)."""
    return JSONResponse({"jobs": tracker.list_jobs()})


async def api_job_status(request):
    """Get current status of a specific job (polling endpoint)."""
    job_id = request.path_params["job_id"]
    job = tracker.get_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(job)


async def api_job_events(request):
    """SSE stream of progress events for a job."""
    job_id = request.path_params["job_id"]
    job = tracker.get_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    async def event_stream():
        loop = asyncio.get_event_loop()
        q = tracker.subscribe(job_id, loop)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    # Stop streaming after terminal events
                    if event.get("type") in ("done", "error", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive comment to prevent connection timeout
                    yield ": keepalive\n\n"
        finally:
            tracker.unsubscribe(job_id, q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def api_cancel_job(request):
    """Cancel a running ingestion job."""
    job_id = request.path_params["job_id"]
    job = tracker.get_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job["status"] not in ("pending", "running"):
        return JSONResponse(
            {"error": f"Job is already {job['status']}"}, status_code=400
        )
    ok = tracker.cancel_job(job_id)
    if ok:
        return JSONResponse({"status": "cancelling", "job_id": job_id})
    return JSONResponse({"error": "Could not cancel job"}, status_code=400)


# ── Browse (document outline + read section) ─────────────────────────────────

async def api_document_outline(request):
    """Get the section tree outline for a document."""
    try:
        folder_id = request.path_params["folder_id"]
        filename = request.path_params["filename"]
        folder = get_folder(folder_id)
        tree = load_tree(folder_id, filename)
        if not tree:
            return JSONResponse({"error": f"No section tree for '{filename}'"}, status_code=404)
        summary = tree_summary_view(tree)
        return JSONResponse({"outline": summary})
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_read_section(request):
    """Read full text of a section by node_id."""
    try:
        folder_id = request.path_params["folder_id"]
        filename = request.path_params["filename"]
        node_id = request.path_params["node_id"]
        tree = load_tree(folder_id, filename)
        if not tree:
            return JSONResponse({"error": f"No section tree for '{filename}'"}, status_code=404)
        node = get_node_by_id(tree, node_id)
        if not node:
            return JSONResponse({"error": f"Section '{node_id}' not found"}, status_code=404)
        return JSONResponse({
            "node_id": node["node_id"],
            "title": node["title"],
            "level": node["level"],
            "page_start": node.get("page_start"),
            "page_end": node.get("page_end"),
            "summary": node.get("summary", ""),
            "full_text": node.get("full_text", ""),
            "children": [
                {"node_id": c["node_id"], "title": c["title"]}
                for c in node.get("children", [])
            ],
        })
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Query ────────────────────────────────────────────────────────────────────

async def api_query(request):
    """Raw vector search — no LLM, just cosine similarity results."""
    try:
        body = await request.json()
        folder_ref = body.get("folder", "").strip()
        query_text = body.get("query", "").strip()
        top_k = body.get("top_k", 10)
        source_file = body.get("source_file")
        doc_type = body.get("doc_type")

        if not folder_ref:
            return JSONResponse({"error": "folder is required"}, status_code=400)
        if not query_text:
            return JSONResponse({"error": "query is required"}, status_code=400)

        folder = resolve_folder(folder_ref)
        query_vec = embed_query(query_text)

        results = search(
            collection_name=folder["collection_name"],
            query_vector=query_vec,
            top_k=top_k,
            source_file=source_file,
            doc_type=doc_type,
            query_text=query_text,
        )

        return JSONResponse({
            "folder": folder["name"],
            "query": query_text,
            "results": results,
        })
    except (KeyError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Starlette app ────────────────────────────────────────────────────────────

admin_routes = [
    Route("/", index),
    Route("/api/folders", api_list_folders, methods=["GET"]),
    Route("/api/folders", api_create_folder, methods=["POST"]),
    Route("/api/folders/{folder_id}", api_delete_folder, methods=["DELETE"]),
    Route("/api/folders/{folder_id}", api_rename_folder, methods=["PATCH"]),
    Route("/api/folders/{folder_id}/documents", api_list_documents, methods=["GET"]),
    Route("/api/folders/{folder_id}/documents/{filename:path}/rename", api_rename_document, methods=["PATCH"]),
    Route("/api/folders/{folder_id}/documents/{filename:path}", api_delete_document, methods=["DELETE"]),
    Route("/api/folders/{folder_id}/documents/{filename:path}/outline", api_document_outline, methods=["GET"]),
    Route("/api/folders/{folder_id}/documents/{filename:path}/sections/{node_id}", api_read_section, methods=["GET"]),
    Route("/api/folders/{folder_id}/ingest", api_ingest, methods=["POST"]),
    Route("/api/jobs", api_list_jobs, methods=["GET"]),
    Route("/api/jobs/{job_id}", api_job_status, methods=["GET"]),
    Route("/api/jobs/{job_id}/cancel", api_cancel_job, methods=["POST"]),
    Route("/api/jobs/{job_id}/events", api_job_events, methods=["GET"]),
    Route("/api/query", api_query, methods=["POST"]),
]

admin_app = Starlette(routes=admin_routes)
