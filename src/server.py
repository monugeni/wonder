"""
server.py — Engineering RAG MCP Server with folder/project isolation

Tools exposed to MCP clients:

  FOLDER MANAGEMENT
    create_folder      — create a new project/tender folder
    list_folders       — see all folders with document counts
    delete_folder      — remove a folder and all its documents

  DOCUMENT MANAGEMENT (folder-scoped)
    ingest_document    — parse and index a document into a folder
    list_documents     — list documents within a specific folder
    delete_document    — remove a document from a folder

  QUERY (folder-scoped — queries never cross folder boundaries)
    query              — fast semantic search within a specific folder
    deep_query         — hybrid: vector coarse filter + section tree reasoning
                         Use for precise technical questions that need cross-reference
                         following or exact section-level answers.

Run with:
  python src/server.py
"""

import asyncio
import json
import sys
from pathlib import Path

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from embedder import embed_query
from folder_manager import (
    create_folder,
    delete_folder,
    list_folders,
    rename_folder,
    resolve_folder,
    update_folder_doc_count,
)
from ingestor import ingest_with_split
from pdf_split import (
    delete_all_manifests,
    delete_manifest,
    load_manifest,
    rename_manifest,
)
from progress import tracker
from tree_retriever import deep_query as _deep_query
from tree_store import delete_all_trees, delete_tree, rename_tree
from vector_store import (
    create_collection,
    delete_document,
    drop_collection,
    get_table_ids,
    list_documents,
    rename_document,
    search,
)

# MCP servers must not write to stdout — it breaks the stdio protocol
logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL)

try:
    config.validate()
except ValueError as e:
    logger.error(str(e))
    sys.exit(1)


# ── Input schemas ─────────────────────────────────────────────────────────────

class CreateFolderInput(BaseModel):
    name: str
    """Human-readable project or tender name. E.g. 'Project 269 — BPCL Kochi'"""
    description: str = ""
    """Optional description of what this folder contains."""
    tags: list[str] = []
    """Optional tags for grouping. E.g. ['active', 'piping', 'BPCL']"""


class DeleteFolderInput(BaseModel):
    folder: str
    """Folder ID or exact folder name."""
    confirm: bool = False
    """Must be True to confirm deletion. This is irreversible."""


class IngestDocumentInput(BaseModel):
    folder: str
    """Folder ID or exact folder name to ingest into."""
    file_path: str
    """Absolute or relative path to the PDF or DOCX file."""


class ListDocumentsInput(BaseModel):
    folder: str
    """Folder ID or exact folder name."""


class DeleteDocumentInput(BaseModel):
    folder: str
    """Folder ID or exact folder name."""
    source_file: str
    """Filename of the document to remove. E.g. 'P269-PIPE-SPEC-REV2.pdf'"""


class QueryInput(BaseModel):
    folder: str
    """Folder ID or exact folder name to search within. Queries never cross folder boundaries."""
    query: str
    """Natural language question or search terms."""
    top_k: int = 8
    """Number of results to return (default 8)."""
    source_file: str | None = None
    """Optional: restrict to a specific document within the folder."""
    doc_type: str | None = None
    """Optional: restrict by type. One of: spec, standard, vendor_doc, correspondence, drawing, technical_doc"""
    headings_contain: str | None = None
    """Optional: restrict to chunks whose section headings contain this string."""
    include_tables: bool = True
    """Whether to include table chunks in results (default True)."""


class RenameFolderInput(BaseModel):
    folder: str
    """Folder ID or exact folder name."""
    new_name: str
    """New name for the folder."""


class RenameDocumentInput(BaseModel):
    folder: str
    """Folder ID or exact folder name."""
    source_file: str
    """Current filename of the document."""
    new_name: str
    """New filename for the document."""


class CancelIngestionInput(BaseModel):
    job_id: str
    """The job ID to cancel. Use list_ingestion_jobs to find active job IDs."""


class DeepQueryInput(BaseModel):
    folder: str
    """Folder ID or exact folder name to search within."""
    query: str
    """Technical question in natural language."""
    top_documents: int = 3
    """How many candidate documents to drill into after vector coarse filter (default 3)."""
    max_nodes_per_doc: int = 5
    """Max sections to retrieve per document during tree reasoning (default 5)."""
    doc_type: str | None = None
    """Optional: restrict vector stage to one document type."""
    follow_xrefs: bool = True
    """Whether to automatically follow cross-references like 'see Appendix B' (default True)."""


# ── Server ────────────────────────────────────────────────────────────────────

app = Server("engineering-rag")


@app.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(tools=[

        Tool(
            name="create_folder",
            description=(
                "Create a new project or tender folder. Each folder is an isolated search space — "
                "queries against this folder will never return results from other folders. "
                "Example: create_folder(name='Project 269 — BPCL Kochi', tags=['active', 'piping'])"
            ),
            inputSchema=CreateFolderInput.model_json_schema(),
        ),

        Tool(
            name="list_folders",
            description=(
                "List all project/tender folders with their IDs, document counts, and tags. "
                "Use folder IDs or names when calling other tools."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        Tool(
            name="delete_folder",
            description=(
                "Permanently delete a folder and ALL its indexed documents. "
                "You must pass confirm=True. This cannot be undone."
            ),
            inputSchema=DeleteFolderInput.model_json_schema(),
        ),

        Tool(
            name="rename_folder",
            description="Rename a project/tender folder. Does not affect documents or search.",
            inputSchema=RenameFolderInput.model_json_schema(),
        ),

        Tool(
            name="ingest_document",
            description=(
                "Parse a PDF or DOCX and index it into a specific folder. "
                "Large PDFs (100+ pages) are automatically split into sub-documents "
                "using a 10-engine heuristic splitter before ingestion — handles "
                "500MB tender packages with 10,000+ pages. "
                "Each sub-document gets full hierarchical parsing, contextual enrichment, "
                "and section tree building. OCR is applied automatically to scanned pages only. "
                "Example: ingest_document(folder='Project 269 — BPCL Kochi', file_path='/docs/tender_package.pdf')"
            ),
            inputSchema=IngestDocumentInput.model_json_schema(),
        ),

        Tool(
            name="list_documents",
            description="List all documents indexed within a specific folder.",
            inputSchema=ListDocumentsInput.model_json_schema(),
        ),

        Tool(
            name="delete_document",
            description=(
                "Remove a specific document and all its chunks from a folder. "
                "Use before re-ingesting an updated version of the same document."
            ),
            inputSchema=DeleteDocumentInput.model_json_schema(),
        ),

        Tool(
            name="rename_document",
            description=(
                "Rename a document within a folder. Updates metadata only — "
                "does not re-embed or affect search quality."
            ),
            inputSchema=RenameDocumentInput.model_json_schema(),
        ),

        Tool(
            name="query",
            description=(
                "Fast semantic search within a specific folder. "
                "Best for broad searches, keyword lookups, or when you need many results quickly. "
                "Results are strictly limited to that folder — no cross-project leakage. "
                "Example: query(folder='Project 269 — BPCL Kochi', query='bend radius process piping', doc_type='spec')"
            ),
            inputSchema=QueryInput.model_json_schema(),
        ),

        Tool(
            name="deep_query",
            description=(
                "Hybrid deep query: vector coarse filter to find candidate documents, "
                "then section-tree reasoning within those documents. "
                "Use this for precise technical questions, especially when the answer might be in a specific "
                "numbered clause, table, or appendix, or when the question involves cross-references "
                "like 'see Appendix G' or 'per Table 4.3-1'. "
                "Slower than query but gives section-level answers with page citations. "
                "Example: deep_query(folder='Project 269 — BPCL Kochi', query='What is the minimum wall thickness for 6 inch process piping per the project spec?')"
            ),
            inputSchema=DeepQueryInput.model_json_schema(),
        ),

        Tool(
            name="list_ingestion_jobs",
            description=(
                "List all ingestion jobs with their current status and progress. "
                "Shows running, completed, and failed jobs. "
                "Useful for monitoring long-running ingestion of large tender PDFs."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        Tool(
            name="cancel_ingestion_job",
            description=(
                "Cancel a running ingestion job. The job will stop at the next "
                "safe checkpoint and clean up any partially-ingested data. "
                "Use list_ingestion_jobs to find the job_id."
            ),
            inputSchema=CancelIngestionInput.model_json_schema(),
        ),
    ])


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    handlers = {
        "create_folder":        _handle_create_folder,
        "list_folders":         _handle_list_folders,
        "delete_folder":        _handle_delete_folder,
        "rename_folder":        _handle_rename_folder,
        "ingest_document":      _handle_ingest,
        "list_documents":       _handle_list_documents,
        "delete_document":      _handle_delete_document,
        "rename_document":      _handle_rename_document,
        "query":                _handle_query,
        "deep_query":           _handle_deep_query,
        "list_ingestion_jobs":  _handle_list_jobs,
        "cancel_ingestion_job": _handle_cancel_job,
    }
    handler = handlers.get(name)
    if not handler:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )
    return await handler(arguments)


# ── Handlers ──────────────────────────────────────────────────────────────────

async def _handle_create_folder(arguments: dict) -> CallToolResult:
    try:
        args = CreateFolderInput(**arguments)
        folder = create_folder(
            name=args.name,
            description=args.description,
            tags=args.tags,
        )
        # Create the Qdrant collection immediately
        create_collection(folder["collection_name"])

        output = (
            f"Folder created successfully.\n\n"
            f"Name:        {folder['name']}\n"
            f"Folder ID:   {folder['folder_id']}\n"
            f"Description: {folder['description'] or '—'}\n"
            f"Tags:        {', '.join(folder['tags']) or '—'}\n\n"
            f"You can now ingest documents using:\n"
            f"  ingest_document(folder='{folder['folder_id']}', file_path='...')\n"
            f"or\n"
            f"  ingest_document(folder='{folder['name']}', file_path='...')"
        )
        return CallToolResult(content=[TextContent(type="text", text=output)])
    except ValueError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"create_folder error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


async def _handle_list_folders(arguments: dict) -> CallToolResult:
    try:
        folders = list_folders()
        if not folders:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="No folders yet. Create one with create_folder()."
                )]
            )

        lines = [
            f"{'ID':<12} {'Docs':>5}  {'Name':<40} Tags",
            "-" * 80,
        ]
        for f in folders:
            tags_str = ", ".join(f["tags"]) if f["tags"] else "—"
            lines.append(
                f"{f['folder_id']:<12} {f['doc_count']:>5}  "
                f"{f['name']:<40} {tags_str}"
            )
            if f["description"]:
                lines.append(f"{'':12}       {f['description']}")

        return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])
    except Exception as e:
        logger.exception(f"list_folders error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


async def _handle_delete_folder(arguments: dict) -> CallToolResult:
    try:
        args = DeleteFolderInput(**arguments)
        if not args.confirm:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=(
                        "Deletion not confirmed. "
                        "Pass confirm=True to permanently delete this folder and all its documents. "
                        "This cannot be undone."
                    )
                )],
                isError=True,
            )

        folder = resolve_folder(args.folder)
        collection_name = folder["collection_name"]
        folder_id = folder["folder_id"]

        # Collect table artifact IDs before dropping the collection
        docs = list_documents(collection_name)
        all_table_ids = []
        for doc_info in docs:
            all_table_ids.extend(
                get_table_ids(collection_name, doc_info["source_file"])
            )

        # Drop Qdrant collection
        drop_collection(collection_name)

        # Clean up table JSON artifacts
        tables_removed = 0
        for tid in all_table_ids:
            artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
            if artifact_path.exists():
                artifact_path.unlink()
                tables_removed += 1

        # Clean up all trees for this folder
        trees_removed = delete_all_trees(folder_id)

        # Clean up all split manifests for this folder
        manifests_removed = delete_all_manifests(folder_id)

        # Remove from registry
        delete_folder(folder_id)

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=(
                    f"Folder '{folder['name']}' deleted.\n"
                    f"Qdrant collection '{collection_name}' dropped.\n"
                    f"All {folder['doc_count']} document(s) and their chunks removed.\n"
                    f"{trees_removed} tree(s), {tables_removed} table artifact(s), "
                    f"and {manifests_removed} split manifest(s) cleaned up."
                )
            )]
        )
    except (KeyError, ValueError) as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"delete_folder error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


async def _handle_rename_folder(arguments: dict) -> CallToolResult:
    try:
        args = RenameFolderInput(**arguments)
        folder = resolve_folder(args.folder)
        updated = rename_folder(folder["folder_id"], args.new_name)
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Folder renamed: '{folder['name']}' -> '{updated['name']}'"
            )]
        )
    except (KeyError, ValueError) as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"rename_folder error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


async def _handle_ingest(arguments: dict) -> CallToolResult:
    try:
        args = IngestDocumentInput(**arguments)
        folder = resolve_folder(args.folder)
        job_id = tracker.create_job(args.file_path, folder["name"])
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None, ingest_with_split, args.file_path, folder["folder_id"], job_id
        )
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(summary, indent=2))]
        )
    except FileNotFoundError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"File not found: {e}")], isError=True
        )
    except (KeyError, ValueError) as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"ingest_document error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Ingestion failed: {e}")], isError=True
        )


async def _handle_list_documents(arguments: dict) -> CallToolResult:
    try:
        args = ListDocumentsInput(**arguments)
        folder = resolve_folder(args.folder)
        docs = list_documents(folder["collection_name"])

        if not docs:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"No documents in folder '{folder['name']}' yet."
                )]
            )

        lines = [
            f"Folder: {folder['name']}  ({folder['folder_id']})",
            f"{'Document':<50} {'Type':<15} {'Chunks':>7} {'Tables':>7}",
            "-" * 85,
        ]
        for d in sorted(docs, key=lambda x: x["source_file"]):
            lines.append(
                f"{d['source_file']:<50} {d['doc_type']:<15} "
                f"{d['chunk_count']:>7} {d['table_count']:>7}"
            )

        return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])
    except (KeyError, ValueError) as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"list_documents error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


async def _handle_delete_document(arguments: dict) -> CallToolResult:
    try:
        args = DeleteDocumentInput(**arguments)
        folder = resolve_folder(args.folder)
        collection_name = folder["collection_name"]
        folder_id = folder["folder_id"]

        details = []

        # Check if this is a split parent document (has a manifest)
        manifest = load_manifest(folder_id, args.source_file)
        if manifest:
            # Delete all split parts
            total_chunks = 0
            total_tables = 0
            for part in manifest["parts"]:
                part_file = part["filename"]
                # Clean up table artifacts
                part_table_ids = get_table_ids(collection_name, part_file)
                count = delete_document(collection_name, part_file)
                if count > 0:
                    total_chunks += count
                    update_folder_doc_count(folder_id, delta=-1)
                for tid in part_table_ids:
                    artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
                    if artifact_path.exists():
                        artifact_path.unlink()
                        total_tables += 1
                delete_tree(folder_id, part_file)

            delete_manifest(folder_id, args.source_file)
            details.append(
                f"Deleted {len(manifest['parts'])} split parts "
                f"({total_chunks} chunks) for '{args.source_file}' "
                f"from folder '{folder['name']}'."
            )
            if total_tables:
                details.append(f"{total_tables} table artifact(s) removed.")
        else:
            # Single document (not split) — original delete logic
            table_ids = get_table_ids(collection_name, args.source_file)
            count = delete_document(collection_name, args.source_file)
            tree_deleted = delete_tree(folder_id, args.source_file)

            tables_removed = 0
            for tid in table_ids:
                artifact_path = config.TABLE_STORE_DIR / f"{tid}.json"
                if artifact_path.exists():
                    artifact_path.unlink()
                    tables_removed += 1

            if count > 0:
                update_folder_doc_count(folder_id, delta=-1)

            details.append(
                f"Deleted {count} chunks for '{args.source_file}' "
                f"from folder '{folder['name']}'."
            )
            if tree_deleted:
                details.append("Section tree removed.")
            if tables_removed:
                details.append(f"{tables_removed} table artifact(s) removed.")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(details))]
        )
    except (KeyError, ValueError) as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"delete_document error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


async def _handle_rename_document(arguments: dict) -> CallToolResult:
    try:
        args = RenameDocumentInput(**arguments)
        folder = resolve_folder(args.folder)
        collection_name = folder["collection_name"]
        folder_id = folder["folder_id"]

        manifest = load_manifest(folder_id, args.source_file)
        if manifest:
            rename_manifest(folder_id, args.source_file, args.new_name)
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Renamed split document: '{args.source_file}' -> '{args.new_name}'"
                )]
            )

        count = rename_document(collection_name, args.source_file, args.new_name)
        rename_tree(folder_id, args.source_file, args.new_name)

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=(
                    f"Renamed: '{args.source_file}' -> '{args.new_name}'\n"
                    f"{count} chunk(s) updated in folder '{folder['name']}'."
                )
            )]
        )
    except (KeyError, ValueError) as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"rename_document error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


async def _handle_query(arguments: dict) -> CallToolResult:
    try:
        args = QueryInput(**arguments)
        folder = resolve_folder(args.folder)
        collection_name = folder["collection_name"]

        query_vector = embed_query(args.query)

        results = search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=args.top_k,
            source_file=args.source_file,
            doc_type=args.doc_type,
            headings_contain=args.headings_contain,
            query_text=args.query,
        )

        if not args.include_tables:
            results = [r for r in results if not r["is_table"]]

        formatted = []
        for i, r in enumerate(results, 1):
            headings_str = " > ".join(r["headings"]) if r["headings"] else "—"
            pages_str = (
                ", ".join(str(p) for p in r["page_numbers"])
                if r["page_numbers"] else "—"
            )
            flags = []
            if r["is_table"]:
                flags.append("TABLE")
            if r.get("ocr_applied"):
                flags.append("OCR")
            flags_str = " | " + " | ".join(flags) if flags else ""

            entry = (
                f"[{i}] Score: {r['score']} | {r['source_file']} | {r['doc_type']}{flags_str}\n"
                f"    Section: {headings_str}\n"
                f"    Pages:   {pages_str}\n"
            )
            if r["context"]:
                entry += f"    Context: {r['context']}\n"
            entry += f"\n{r['text']}\n"
            formatted.append(entry)

        if not formatted:
            output = (
                f"No results found in folder '{folder['name']}' for query: {args.query}\n"
                "Try broadening your query or removing filters."
            )
        else:
            header = [
                f"Folder:  {folder['name']}  ({folder['folder_id']})",
                f"Query:   {args.query}",
            ]
            if args.source_file:
                header.append(f"Filter:  document={args.source_file}")
            if args.doc_type:
                header.append(f"Filter:  type={args.doc_type}")
            if args.headings_contain:
                header.append(f"Filter:  section contains '{args.headings_contain}'")
            header.append(f"Results: {len(formatted)}")
            header.append("=" * 60)

            output = "\n".join(header) + "\n\n"
            output += ("\n" + "-" * 60 + "\n\n").join(formatted)

        return CallToolResult(content=[TextContent(type="text", text=output)])

    except (KeyError, ValueError) as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"query error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Query failed: {e}")], isError=True
        )




async def _handle_deep_query(arguments: dict) -> CallToolResult:
    try:
        args = DeepQueryInput(**arguments)
        folder = resolve_folder(args.folder)

        result = _deep_query(
            collection_name=folder["collection_name"],
            folder_id=folder["folder_id"],
            question=args.query,
            top_documents=args.top_documents,
            max_nodes_per_doc=args.max_nodes_per_doc,
            doc_type=args.doc_type,
            follow_xrefs=args.follow_xrefs,
        )

        # Format output
        lines = [
            f"Folder:  {folder['name']}",
            f"Query:   {args.query}",
            "=" * 60,
            "",
            result["answer"],
            "",
            "Sources:",
        ]
        for src in result["sources"]:
            pages = (
                f"pp. {src['page_start']}-{src['page_end']}"
                if src.get("page_start") else "pages unknown"
            )
            lines.append(f"  {src['source_file']}  |  {src['section']}  |  {pages}")

        return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])

    except (KeyError, ValueError) as e:
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))], isError=True
        )
    except Exception as e:
        logger.exception(f"deep_query error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Deep query failed: {e}")], isError=True
        )


async def _handle_list_jobs(arguments: dict) -> CallToolResult:
    try:
        jobs = tracker.list_jobs()
        if not jobs:
            return CallToolResult(
                content=[TextContent(type="text", text="No ingestion jobs.")]
            )

        lines = [
            f"{'Job ID':<12} {'Status':<12} {'Progress':>8}  {'Elapsed':>8}  {'File'}",
            "-" * 80,
        ]
        for j in jobs:
            pct = f"{round(j.get('progress', 0) * 100)}%"
            elapsed = f"{j.get('elapsed', 0):.0f}s"
            lines.append(
                f"{j['job_id']:<12} {j['status']:<12} {pct:>8}  "
                f"{elapsed:>8}  {j['source_file']}"
            )
            msg = j.get("message", "")
            if msg:
                lines.append(f"{'':12} {msg}")

        return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])
    except Exception as e:
        logger.exception(f"list_ingestion_jobs error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


async def _handle_cancel_job(arguments: dict) -> CallToolResult:
    try:
        args = CancelIngestionInput(**arguments)
        job = tracker.get_job(args.job_id)
        if not job:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Job '{args.job_id}' not found.")],
                isError=True,
            )
        if job["status"] not in ("pending", "running"):
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Job '{args.job_id}' is already {job['status']}. Cannot cancel."
                )],
                isError=True,
            )
        ok = tracker.cancel_job(args.job_id)
        if ok:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=(
                        f"Cancellation signal sent to job '{args.job_id}'.\n"
                        f"The job will stop at the next checkpoint and clean up "
                        f"partially-ingested data. Use list_ingestion_jobs to verify."
                    )
                )]
            )
        return CallToolResult(
            content=[TextContent(type="text", text="Could not cancel job.")],
            isError=True,
        )
    except Exception as e:
        logger.exception(f"cancel_ingestion_job error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed: {e}")], isError=True
        )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main_stdio():
    """Run over stdio (default — for local Claude Desktop or Claude Code)."""
    logger.info("Starting Engineering RAG MCP Server (stdio)")
    logger.info(f"  Qdrant:     {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    logger.info(f"  Embeddings: {config.EMBEDDING_MODEL}")
    logger.info(f"  Context LM: {config.CONTEXT_MODEL}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main_sse(host: str = "0.0.0.0", port: int = 8080):
    """Run over SSE/HTTP (for remote Claude Desktop connections)."""
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    from admin import admin_app

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(
                streams[0], streams[1], app.create_initialization_options()
            )
        return Response()

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
            Mount("/admin", app=admin_app),
        ],
    )

    logger.info(f"Starting Engineering RAG MCP Server (SSE) on {host}:{port}")
    logger.info(f"  Qdrant:     {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    logger.info(f"  Embeddings: {config.EMBEDDING_MODEL}")
    logger.info(f"  Context LM: {config.CONTEXT_MODEL}")
    logger.info(f"  SSE endpoint: http://{host}:{port}/sse")
    logger.info(f"  Admin UI:    http://{host}:{port}/admin/")

    uvicorn.run(starlette_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Engineering RAG MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode: stdio (local) or sse (remote HTTP)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="SSE bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="SSE bind port (default: 8080)")
    args = parser.parse_args()

    if args.transport == "sse":
        main_sse(host=args.host, port=args.port)
    else:
        asyncio.run(main_stdio())
