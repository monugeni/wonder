"""
parser.py — parse PDFs/DOCX using Docling, extract hierarchical chunks and tables.

Key behaviours:
- Detects image-only (scanned) pages using PyMuPDF before invoking Docling
- OCR is applied ONLY to pages that have no selectable text — saves significant time
- Uses Docling's HybridChunker to respect document structure (headings, sections, subsections)
- Tables are extracted separately via TableFormer and stored as JSON artifacts
- Each chunk carries heading breadcrumbs so downstream context enrichment knows where it came from
- Multi-page tables are kept intact (not split across chunks)

Selective OCR logic:
  A page is considered "image-only" (needs OCR) when:
    1. PyMuPDF finds zero text characters on the page, AND
    2. PyMuPDF finds at least one image object on the page
  This correctly handles:
    - Fully digital PDFs (no OCR needed — fast path)
    - Fully scanned PDFs (all pages OCR'd)
    - Mixed PDFs (e.g. a spec with scanned appendix pages — only appendix pages OCR'd)
    - Pages with watermarks/logos that have text elsewhere (watermark has text, so not flagged)
"""

import json
import signal
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — used only for page-type detection, not for text extraction
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions
from docling.chunking import HybridChunker
from loguru import logger

from config import config


# Minimum character count below which a page is considered to have "no text".
# A page with only a page number or a single word is still effectively an image page.
TEXT_CHAR_THRESHOLD = 20


@contextmanager
def _suppress_signal_in_thread():
    """Suppress 'signal only works in main thread' errors from libraries like Docling.

    Some PDF processing libraries call signal.signal() for timeout protection.
    Python forbids this in non-main threads, causing ValueError. We temporarily
    patch signal.signal to be a no-op when running in a background thread.
    """
    if threading.current_thread() is threading.main_thread():
        yield
        return
    original = signal.signal
    signal.signal = lambda *a, **kw: signal.SIG_DFL
    try:
        yield
    finally:
        signal.signal = original


@dataclass
class ParsedChunk:
    """A single chunk from a parsed document, with full structural context."""
    chunk_id: str
    text: str                          # The actual text content
    headings: list[str]                # Breadcrumb trail: ["Chapter 4", "4.3 Pipe Bends"]
    page_numbers: list[int]            # Source pages this chunk came from
    source_file: str                   # Original filename
    doc_type: str                      # Detected document type
    is_table: bool = False             # True if this chunk IS a table
    table_id: Optional[str] = None     # Links text chunks to their source table
    table_markdown: Optional[str] = None  # Full table in markdown (only if is_table=True)
    ocr_applied: bool = False          # True if this chunk came from an OCR'd page
    metadata: dict = field(default_factory=dict)


def detect_doc_type(filename: str) -> str:
    """
    Heuristic classification of engineering document types based on filename patterns.
    This drives metadata filtering — users can query "only in standards" etc.
    """
    name = filename.lower()

    if any(k in name for k in ["api", "asme", "iso", "astm", "is-", "din", "bs-", "nfpa"]):
        return "standard"
    if any(k in name for k in ["spec", "specification", "datasheet", "data-sheet", "ds-"]):
        return "spec"
    if any(k in name for k in ["vendor", "supplier", "quotation", "quote", "offer", "vdr"]):
        return "vendor_doc"
    if any(k in name for k in ["mail", "letter", "memo", "correspondence", "minutes", "mom"]):
        return "correspondence"
    if any(k in name for k in ["drawing", "dwg", "layout", "p&id", "pfd", "ga-"]):
        return "drawing"

    return "technical_doc"  # fallback


def detect_image_only_pages(file_path: Path) -> set[int]:
    """
    Use PyMuPDF to scan every page and identify those that are image-only.

    Returns a set of 1-based page numbers that need OCR.

    A page needs OCR when:
      - It has fewer than TEXT_CHAR_THRESHOLD selectable characters, AND
      - It contains at least one raster image object

    This is fast (PyMuPDF reads the PDF index, not the full pixel data)
    and adds only a second or two even for a 500-page document.
    """
    if file_path.suffix.lower() != ".pdf":
        return set()  # DOCX files are always digital — no OCR needed

    image_only_pages: set[int] = set()

    try:
        pdf = fitz.open(str(file_path))
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1  # Docling uses 1-based page numbers

            # Count selectable text characters
            text = page.get_text("text")
            char_count = len(text.strip())

            # Count image objects on the page
            image_list = page.get_images(full=False)
            image_count = len(image_list)

            if char_count < TEXT_CHAR_THRESHOLD and image_count > 0:
                image_only_pages.add(page_number)
                logger.debug(
                    f"  Page {page_number}: image-only "
                    f"(chars={char_count}, images={image_count}) → OCR"
                )

        pdf.close()
    except Exception as e:
        logger.warning(f"PyMuPDF page scan failed ({e}) — will use Docling defaults")
        return set()

    return image_only_pages


def _build_pipeline_options(ocr_page_numbers: set[int]) -> PdfPipelineOptions:
    """
    Build Docling pipeline options.

    If any pages need OCR, enable OCR globally but pass the page list
    so Docling only OCRs those specific pages.
    If no pages need OCR, disable OCR entirely for maximum speed.
    """
    options = PdfPipelineOptions()
    options.do_table_structure = True
    options.table_structure_options.do_cell_matching = True

    if ocr_page_numbers:
        options.do_ocr = True
        ocr_opts = TesseractOcrOptions()
        options.ocr_options = ocr_opts

        # Pass specific page numbers to OCR
        # Docling will skip OCR on all other pages
        if hasattr(options, "ocr_page_numbers"):
            options.ocr_page_numbers = sorted(ocr_page_numbers)
    else:
        options.do_ocr = False

    return options


def _extract_page_numbers(chunk_items: list) -> list[int]:
    """Pull page numbers from Docling chunk provenance data."""
    pages = set()
    for item in chunk_items:
        try:
            for prov in item.prov:
                pages.add(prov.page_no)
        except (AttributeError, TypeError):
            pass
    return sorted(pages)


def _extract_headings(chunk_meta) -> list[str]:
    """Extract the heading breadcrumb from Docling chunk metadata."""
    try:
        headings = chunk_meta.headings or []
        return [h.strip() for h in headings if h.strip()]
    except AttributeError:
        return []


def parse_document_with_doc(file_path: str | Path) -> tuple[list["ParsedChunk"], object]:
    """
    Like parse_document() but also returns the raw DoclingDocument object.
    Used by the ingestor when tree building is enabled, so Docling only runs once.

    Returns:
        (chunks, docling_document)
    """
    file_path = Path(file_path)
    filename = file_path.name
    doc_type = detect_doc_type(filename)

    logger.info(f"Parsing: {filename} (detected type: {doc_type})")

    image_only_pages = detect_image_only_pages(file_path)
    if image_only_pages:
        logger.info(
            f"  Image-only pages requiring OCR: {sorted(image_only_pages)} "
            f"({len(image_only_pages)} page(s))"
        )
    else:
        logger.info("  No image-only pages detected — OCR disabled (fast path)")

    pipeline_options = _build_pipeline_options(image_only_pages)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    with _suppress_signal_in_thread():
        result = converter.convert(str(file_path))
    doc = result.document

    # Re-use the existing chunk extraction logic by calling the internal helper
    chunks = _extract_chunks(doc, filename, doc_type, image_only_pages)
    return chunks, doc


def _extract_chunks(doc, filename: str, doc_type: str, image_only_pages: set[int]) -> list["ParsedChunk"]:
    """
    Internal: extract ParsedChunks from an already-converted DoclingDocument.
    Separated out so parse_document and parse_document_with_doc share the logic.
    """
    chunks: list[ParsedChunk] = []
    table_index: dict[str, str] = {}

    for table in doc.tables:
        table_id = str(uuid.uuid4())
        try:
            table_md = table.export_to_markdown()
        except Exception:
            table_md = str(table)

        table_page_nums = _extract_page_numbers(
            table.prov if hasattr(table, "prov") else []
        )
        table_ocr = bool(image_only_pages & set(table_page_nums))

        artifact = {
            "table_id": table_id,
            "source_file": filename,
            "markdown": table_md,
            "page_numbers": table_page_nums,
            "ocr_applied": table_ocr,
        }
        artifact_path = config.TABLE_STORE_DIR / f"{table_id}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2))

        if hasattr(table, "self_ref"):
            table_index[table.self_ref] = table_id

        table_chunk = ParsedChunk(
            chunk_id=str(uuid.uuid4()),
            text=table_md,
            headings=[],
            page_numbers=table_page_nums,
            source_file=filename,
            doc_type=doc_type,
            is_table=True,
            table_id=table_id,
            table_markdown=table_md,
            ocr_applied=table_ocr,
        )
        chunks.append(table_chunk)
        logger.debug(f"  Extracted table {table_id} ({len(table_md)} chars)" + (" [OCR]" if table_ocr else ""))

    chunker = HybridChunker()
    for raw_chunk in chunker.chunk(doc):
        meta = raw_chunk.meta
        headings = _extract_headings(meta)
        page_nums = _extract_page_numbers(meta.doc_items if hasattr(meta, "doc_items") else [])
        text = raw_chunk.text.strip()
        if not text:
            continue

        chunk_ocr = bool(image_only_pages & set(page_nums))
        is_table_chunk = False
        linked_table_id = None
        try:
            for item in meta.doc_items:
                ref = getattr(item, "self_ref", None)
                if ref and ref in table_index:
                    is_table_chunk = True
                    linked_table_id = table_index[ref]
                    break
        except (AttributeError, TypeError):
            pass

        if is_table_chunk:
            continue

        chunks.append(ParsedChunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            headings=headings,
            page_numbers=page_nums,
            source_file=filename,
            doc_type=doc_type,
            is_table=False,
            table_id=linked_table_id,
            ocr_applied=chunk_ocr,
        ))

    # Sort chunks by page order so tables (extracted first) are interleaved
    # with text chunks at the correct position. Pageless chunks keep their
    # original extraction order (tables first, then text) since their true
    # position is unknown — the tree enrichment step assigns correct headings.
    max_page = max((max(c.page_numbers) for c in chunks if c.page_numbers), default=0)
    _pageless_idx = [0]
    def _sort_key(c):
        if c.page_numbers:
            return (min(c.page_numbers), 0 if not c.is_table else 1)
        # Pageless: place after all paged chunks, preserve extraction order
        _pageless_idx[0] += 1
        return (max_page + _pageless_idx[0], 0)
    chunks.sort(key=_sort_key)

    # Forward heading propagation: chunks inherit from nearest preceding chunk
    last_headings: list[str] = []
    for chunk in chunks:
        if chunk.headings:
            last_headings = chunk.headings
        elif last_headings:
            chunk.headings = list(last_headings)

    # Backward heading propagation: leading orphan chunks (e.g. tables with
    # no page numbers placed at the end) inherit from the next chunk that has headings
    next_headings: list[str] = []
    for chunk in reversed(chunks):
        if chunk.headings:
            next_headings = chunk.headings
        elif next_headings:
            chunk.headings = list(next_headings)

    ocr_count = sum(1 for c in chunks if c.ocr_applied)
    logger.info(
        f"  Parsed {filename}: {len(chunks)} chunks "
        f"({sum(1 for c in chunks if c.is_table)} tables, {ocr_count} from OCR'd pages)"
    )
    return chunks


def parse_document(file_path: str | Path) -> list[ParsedChunk]:
    """
    Parse a PDF or DOCX file and return a list of structured chunks.

    Each chunk preserves:
    - Heading breadcrumbs (document hierarchy)
    - Source page numbers
    - Whether it is or references a table
    - Whether it came from an OCR'd page
    - Document type classification

    Tables are extracted separately and stored as JSON in TABLE_STORE_DIR.
    The table's full markdown is embedded; a reference chunk is also created
    so surrounding text can link to the table by ID.
    """
    file_path = Path(file_path)
    filename = file_path.name
    doc_type = detect_doc_type(filename)

    logger.info(f"Parsing: {filename} (detected type: {doc_type})")

    # Step 1: Identify image-only pages (fast pre-scan with PyMuPDF)
    image_only_pages = detect_image_only_pages(file_path)
    total_pages_hint = ""
    if image_only_pages:
        logger.info(
            f"  Image-only pages requiring OCR: {sorted(image_only_pages)} "
            f"({len(image_only_pages)} page(s))"
        )
    else:
        logger.info("  No image-only pages detected — OCR disabled (fast path)")

    # Step 2: Build Docling pipeline with selective OCR
    pipeline_options = _build_pipeline_options(image_only_pages)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    with _suppress_signal_in_thread():
        result = converter.convert(str(file_path))
    doc = result.document

    return _extract_chunks(doc, filename, doc_type, image_only_pages)
