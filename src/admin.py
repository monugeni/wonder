"""
admin.py — REST API + static admin page for folder/document management

Mounted alongside the MCP SSE server at /admin/*.
Provides CRUD for folders and documents, file upload for ingestion,
and raw vector search (no LLM — just cosine similarity results).

Endpoints:
  GET  /admin/                              — admin HTML page
  GET  /admin/api/folders                   — list all folders
  POST /admin/api/folders                   — create a folder
  DELETE /admin/api/folders/{id}            — delete a folder
  GET  /admin/api/folders/{id}/documents    — list documents in a folder
  DELETE /admin/api/folders/{id}/documents/{filename} — delete a document
  POST /admin/api/folders/{id}/ingest       — upload + ingest a file
  POST /admin/api/query                     — raw vector search (no LLM)
"""

import json
import tempfile
import traceback
from pathlib import Path

from loguru import logger
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
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
    resolve_folder,
    update_folder_doc_count,
)
from ingestor import ingest
from tree_store import delete_all_trees, delete_tree
from vector_store import (
    create_collection,
    delete_document,
    drop_collection,
    get_table_ids,
    list_documents,
    search,
)


ADMIN_HTML = Path(__file__).parent / "static" / "admin.html"


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
        delete_folder(folder_id)

        return JSONResponse({"status": "deleted", "folder_id": folder_id})
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_list_documents(request):
    try:
        folder_id = request.path_params["folder_id"]
        folder = get_folder(folder_id)
        docs = list_documents(folder["collection_name"])
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

        # Clean up table artifacts
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

        # Save upload to temp file and ingest
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Rename to preserve original filename
        final_path = Path(tmp_path).parent / upload.filename
        Path(tmp_path).rename(final_path)

        try:
            summary = ingest(str(final_path), folder_id)
            return JSONResponse({"status": "success", "summary": summary})
        finally:
            if final_path.exists():
                final_path.unlink()

    except (KeyError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception(f"Ingest error: {e}")
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


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
    Route("/api/folders/{folder_id}/documents", api_list_documents, methods=["GET"]),
    Route("/api/folders/{folder_id}/documents/{filename:path}", api_delete_document, methods=["DELETE"]),
    Route("/api/folders/{folder_id}/ingest", api_ingest, methods=["POST"]),
    Route("/api/query", api_query, methods=["POST"]),
]

admin_app = Starlette(routes=admin_routes)
