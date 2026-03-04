#!/usr/bin/env python3
"""
ingest_batch.py — bulk ingest documents from the command line into a folder

Usage:
  python scripts/ingest_batch.py --folder "Project 269 — BPCL Kochi" /path/to/docs/*.pdf
  python scripts/ingest_batch.py --folder f_3a8b1c2d /path/to/spec.pdf /path/to/standard.pdf

The folder must already exist. Create it first:
  python scripts/create_folder.py --name "Project 269 — BPCL Kochi" --tags active piping
"""

import argparse
import glob
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from folder_manager import resolve_folder
from ingestor import ingest


def main():
    parser = argparse.ArgumentParser(description="Batch ingest documents into a RAG folder")
    parser.add_argument("--folder", required=True, help="Folder ID or name")
    parser.add_argument("files", nargs="+", help="Files or globs to ingest")
    args = parser.parse_args()

    # Resolve folder
    try:
        folder = resolve_folder(args.folder)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nTarget folder: {folder['name']}  ({folder['folder_id']})")

    # Expand globs
    files = []
    for pattern in args.files:
        matched = glob.glob(pattern)
        files.extend(matched if matched else [pattern])

    if not files:
        print("No files found.")
        sys.exit(1)

    print(f"Ingesting {len(files)} file(s)...\n")

    results = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(f)}")
        try:
            summary = ingest(f, folder["folder_id"])
            results.append(summary)
            print(
                f"  Done: {summary['total_chunks']} chunks "
                f"({summary['table_chunks']} tables, {summary['ocr_chunks']} OCR'd) "
                f"in {summary['total_time_seconds']}s"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"status": "error", "file": f, "error": str(e)})

    success = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - success
    total_chunks = sum(r.get("total_chunks", 0) for r in results)

    print(f"\n{'='*50}")
    print(f"Batch complete: {success} succeeded, {failed} failed")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Folder: {folder['name']}")


if __name__ == "__main__":
    main()
