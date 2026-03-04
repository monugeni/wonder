# Engineering RAG — MCP Server

A context-aware RAG system for large, hierarchical technical documents. Built for EPC engineering use cases where the same word (e.g. "bend") means something completely different depending on which document, chapter, and section it appears in.

## Architecture

```
PDF / DOCX
    │
    ▼
Docling (parse + hierarchical chunking + table extraction)
    │
    ▼
Anthropic Claude (contextual retrieval — stamps each chunk with doc/chapter/section context)
    │
    ▼
sentence-transformers (local embeddings)
    │
    ▼
Qdrant (local vector store with metadata filtering)
    │
    ▼
MCP Server (exposes ingest_document + query tools)
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `ingest_document` | Parse a PDF/DOCX, chunk it hierarchically, enrich with context, embed, store in Qdrant |
| `query` | Query with optional filters (document name, chapter, section) |
| `list_documents` | List all ingested documents and their metadata |
| `delete_document` | Remove a document and all its chunks from the index |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant (Docker)

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 4. Run the MCP server

```bash
python src/server.py
```

### 5. Connect from Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "engineering-rag": {
      "command": "python",
      "args": ["/path/to/engineering-rag/src/server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Key Design Decisions

### Why Contextual Retrieval matters here

Standard RAG embeds chunks in isolation. A chunk containing "the bend shall have a minimum radius of 3D" has no idea it came from "Chapter 4 — Piping Design" of "Project 269 Piping Spec." When retrieved, it could be confused with a bend in a road alignment or an ergonomic body position.

Contextual Retrieval prepends a short context sentence to every chunk before embedding:

> *"This chunk is from the document 'P269-PIPE-SPEC-REV2.pdf', Chapter 4: Piping Design, Section 4.3: Pipe Bends and Elbows. It describes minimum bend radius requirements for process piping."*

The embedding now carries the situated meaning, not just the raw words.

### Table handling

Tables are extracted by Docling's TableFormer model and stored in two ways:
1. As Markdown in the vector store (for semantic search)
2. As a separate JSON artifact on disk, linked by a table ID (so the full table can always be retrieved even if only one cell matched the query)

### Metadata filtering

Every chunk stores:
- `source_file` — original filename
- `doc_type` — detected type (spec, standard, correspondence, vendor_doc)
- `headings` — list of heading breadcrumbs e.g. `["Chapter 4", "4.3 Pipe Bends"]`
- `page_numbers` — source pages
- `table_id` — if this chunk is or references a table

You can filter queries by any of these fields.
