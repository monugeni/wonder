#!/usr/bin/env python3
"""
create_folder.py — create a project/tender folder from the command line

Usage:
  python scripts/create_folder.py --name "Project 269 — BPCL Kochi"
  python scripts/create_folder.py --name "Tender T-2024-HMEL" --description "HMEL Gurdaspur expansion" --tags active tender
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from folder_manager import create_folder
from vector_store import create_collection


def main():
    parser = argparse.ArgumentParser(description="Create a RAG folder for a project or tender")
    parser.add_argument("--name", required=True, help="Human-readable folder name")
    parser.add_argument("--description", default="", help="Optional description")
    parser.add_argument("--tags", nargs="*", default=[], help="Optional tags")
    args = parser.parse_args()

    try:
        folder = create_folder(
            name=args.name,
            description=args.description,
            tags=args.tags,
        )
        create_collection(folder["collection_name"])

        print(f"\nFolder created:")
        print(f"  Name:        {folder['name']}")
        print(f"  Folder ID:   {folder['folder_id']}")
        print(f"  Description: {folder['description'] or '—'}")
        print(f"  Tags:        {', '.join(folder['tags']) or '—'}")
        print(f"  Collection:  {folder['collection_name']}")
        print(f"\nNow ingest documents:")
        print(f"  python scripts/ingest_batch.py --folder \"{folder['folder_id']}\" /path/to/*.pdf")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
