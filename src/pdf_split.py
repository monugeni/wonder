"""
pdf_split.py — Split large PDFs using the vendored EPC Tender PDF Splitter.

Provides the bridge between pdfsplitter and Wonder's ingestion pipeline.
Large PDFs (>SPLIT_THRESHOLD pages) are automatically split into
sub-documents before being individually ingested into the RAG system.

The manifest system tracks which split parts belong to which original
file, enabling clean re-ingest and deletion of all related parts.

Flow:
  1. should_split() — check page count against threshold
  2. split_pdf()    — run PDFSplitter, return list of parts with metadata
  3. Manifest CRUD  — track parent→parts relationship for re-ingest/delete
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger

from config import config


# ── Split decision ───────────────────────────────────────────────────────────

def get_page_count(file_path: Path) -> int:
    """Get PDF page count without fully loading the document."""
    try:
        doc = fitz.open(str(file_path))
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def should_split(file_path: Path) -> bool:
    """Check if a PDF exceeds the page threshold and should be split."""
    if file_path.suffix.lower() != ".pdf":
        return False
    return get_page_count(file_path) >= config.SPLIT_THRESHOLD


# ── PDF splitting ────────────────────────────────────────────────────────────

def split_pdf(file_path: Path, output_dir: Path | None = None) -> tuple[list[dict], Path]:
    """
    Split a large PDF into sub-documents using the EPC Tender PDF Splitter.

    Args:
        file_path: Path to the source PDF
        output_dir: Where to write split files (default: temp directory)

    Returns:
        (parts, output_dir) where parts is a list of dicts:
            - path: Path to the split PDF file
            - filename: Just the filename (used as source_file in RAG)
            - label: Descriptive label from splitter
            - start_page: Original start page (1-based)
            - end_page: Original end page (1-based)
            - num_pages: Number of pages in this part
    """
    from splitter import PDFSplitter

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="wonder_split_"))

    page_count = get_page_count(file_path)
    logger.info(
        f"Splitting {file_path.name} ({page_count} pages) "
        f"with threshold={config.SPLIT_SCORE_THRESHOLD}, "
        f"min_pages={config.SPLIT_MIN_DOC_PAGES}"
    )

    splitter = PDFSplitter(
        pdf_path=str(file_path),
        output_dir=str(output_dir),
        threshold=config.SPLIT_SCORE_THRESHOLD,
        min_doc_pages=config.SPLIT_MIN_DOC_PAGES,
    )
    report = splitter.run()

    if report.get("status") == "skipped":
        logger.info(f"  Splitter skipped ({report.get('reason')}), treating as single document")
        return [{
            "path": file_path,
            "filename": file_path.name,
            "label": file_path.stem,
            "start_page": 1,
            "end_page": page_count,
            "num_pages": page_count,
        }], output_dir

    segments = report.get("segments", [])
    if not segments:
        logger.warning("  Splitter returned no segments, treating as single document")
        return [{
            "path": file_path,
            "filename": file_path.name,
            "label": file_path.stem,
            "start_page": 1,
            "end_page": page_count,
            "num_pages": page_count,
        }], output_dir

    # Collect the split files produced by PDFSplitter
    parts = []
    for segment in segments:
        part_num = segment["segment"]
        # PDFSplitter names files: {stem}_part{NNN}_p{start}-{end}_{label}.pdf
        part_pattern = f"*_part{part_num:03d}_*"
        matches = list(output_dir.glob(part_pattern))
        if matches:
            part_path = matches[0]
            parts.append({
                "path": part_path,
                "filename": part_path.name,
                "label": segment.get("label", ""),
                "start_page": segment["start_page"],
                "end_page": segment["end_page"],
                "num_pages": segment["num_pages"],
            })
        else:
            logger.warning(
                f"  Split part {part_num} file not found in {output_dir} "
                f"(pattern: {part_pattern})"
            )

    logger.info(
        f"  Split into {len(parts)} parts: "
        + ", ".join(f"{p['filename']} ({p['num_pages']}pp)" for p in parts[:5])
        + (f" ... +{len(parts)-5} more" if len(parts) > 5 else "")
    )
    return parts, output_dir


# ── Manifest management ──────────────────────────────────────────────────────
# Manifests track which split parts belong to which original source file.
# This enables clean re-ingest (delete all old parts) and clean deletion.

def _manifest_dir(folder_id: str) -> Path:
    return config.TABLE_STORE_DIR.parent / "split_manifests" / folder_id


def _manifest_path(folder_id: str, source_file: str) -> Path:
    stem = Path(source_file).stem
    return _manifest_dir(folder_id) / f"{stem}.manifest.json"


def save_manifest(folder_id: str, source_file: str, parts: list[dict]) -> None:
    """Save a split manifest recording which parts came from which source file."""
    manifest = {
        "source_file": source_file,
        "split_at": datetime.now(timezone.utc).isoformat(),
        "total_parts": len(parts),
        "parts": [
            {
                "filename": p["filename"],
                "label": p.get("label", ""),
                "start_page": p["start_page"],
                "end_page": p["end_page"],
                "num_pages": p["num_pages"],
            }
            for p in parts
        ],
    }
    path = _manifest_path(folder_id, source_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    logger.debug(f"Saved split manifest: {path}")


def load_manifest(folder_id: str, source_file: str) -> dict | None:
    """Load a split manifest. Returns None if no manifest exists."""
    path = _manifest_path(folder_id, source_file)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read manifest {path}: {e}")
        return None


def delete_manifest(folder_id: str, source_file: str) -> bool:
    """Delete a split manifest. Returns True if a manifest was deleted."""
    path = _manifest_path(folder_id, source_file)
    if path.exists():
        path.unlink()
        logger.debug(f"Deleted split manifest: {path}")
        return True
    return False


def delete_all_manifests(folder_id: str) -> int:
    """Delete all split manifests for a folder. Returns count deleted."""
    manifest_dir = _manifest_dir(folder_id)
    if not manifest_dir.exists():
        return 0
    count = 0
    for f in manifest_dir.glob("*.manifest.json"):
        f.unlink()
        count += 1
    # Remove the directory if empty
    try:
        manifest_dir.rmdir()
    except OSError:
        pass
    return count


def list_manifests(folder_id: str) -> list[dict]:
    """List all split manifests for a folder."""
    manifest_dir = _manifest_dir(folder_id)
    if not manifest_dir.exists():
        return []
    manifests = []
    for f in manifest_dir.glob("*.manifest.json"):
        try:
            manifests.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return manifests


def cleanup_split_dir(split_dir: Path) -> None:
    """Remove a temporary split directory and all its contents."""
    try:
        if split_dir.exists() and "wonder_split_" in str(split_dir):
            shutil.rmtree(split_dir)
            logger.debug(f"Cleaned up split directory: {split_dir}")
    except OSError as e:
        logger.warning(f"Failed to clean up {split_dir}: {e}")
