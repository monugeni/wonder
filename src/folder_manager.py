"""
folder_manager.py — project/tender folder registry

Each folder maps to its own Qdrant collection, giving hard query isolation.
A "folder" in this system corresponds to a project (e.g. "P269") or a tender.

The folder registry is persisted as a JSON file on disk so it survives
server restarts. It stores folder metadata alongside the Qdrant collection name.

Collection naming: Qdrant collection names must be alphanumeric + underscores.
We sanitise folder names on creation: "Project 269 / BPCL" → "proj_project_269_bpcl"
with a unique suffix to prevent collisions.

Folder registry schema (folders.json):
  {
    "folder_id": {
      "folder_id": str,          -- short unique ID e.g. "proj_p269"
      "name": str,               -- human name e.g. "Project 269 — BPCL Kochi"
      "description": str,        -- optional free-text description
      "collection_name": str,    -- Qdrant collection name (sanitised)
      "created_at": str,         -- ISO datetime
      "doc_count": int,          -- number of documents ingested
      "tags": list[str]          -- optional tags e.g. ["active", "piping"]
    },
    ...
  }
"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from config import config

REGISTRY_PATH = config.TABLE_STORE_DIR.parent / "folders.json"
COLLECTION_PREFIX = "rag_"


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save_registry(registry: dict):
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def _sanitise_collection_name(name: str, folder_id: str) -> str:
    """
    Convert a human folder name to a valid Qdrant collection name.
    Qdrant allows: letters, digits, underscores, hyphens.
    Max length: 255 chars (we cap at 60 to keep it readable).
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
    slug = slug[:40]
    short_id = folder_id[:8]
    return f"{COLLECTION_PREFIX}{slug}_{short_id}"


# ── Public API ────────────────────────────────────────────────────────────────

def create_folder(
    name: str,
    description: str = "",
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Create a new project/tender folder.

    Args:
        name:        Human-readable name, e.g. "Project 269 — BPCL Kochi"
        description: Optional description
        tags:        Optional list of tags e.g. ["active", "piping", "BPCL"]

    Returns:
        The created folder record.
    """
    registry = _load_registry()

    # Check for duplicate name (case-insensitive)
    for existing in registry.values():
        if existing["name"].lower() == name.lower():
            raise ValueError(
                f"A folder named '{name}' already exists "
                f"(id: {existing['folder_id']}). "
                "Use a different name or delete the existing folder first."
            )

    folder_id = "f_" + str(uuid.uuid4())[:8]
    collection_name = _sanitise_collection_name(name, folder_id)

    folder = {
        "folder_id": folder_id,
        "name": name,
        "description": description,
        "collection_name": collection_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "doc_count": 0,
        "tags": tags or [],
    }

    registry[folder_id] = folder
    _save_registry(registry)

    logger.info(f"Created folder '{name}' → collection '{collection_name}'")
    return folder


def get_folder(folder_id: str) -> dict:
    """Get a folder record by ID. Raises KeyError if not found."""
    registry = _load_registry()
    if folder_id not in registry:
        raise KeyError(
            f"Folder '{folder_id}' not found. "
            f"Use list_folders() to see available folders."
        )
    return registry[folder_id]


def get_folder_by_name(name: str) -> Optional[dict]:
    """Case-insensitive name lookup. Returns None if not found."""
    registry = _load_registry()
    for folder in registry.values():
        if folder["name"].lower() == name.lower():
            return folder
    return None


def list_folders() -> list[dict]:
    """Return all folders sorted by name."""
    registry = _load_registry()
    return sorted(registry.values(), key=lambda f: f["name"])


def update_folder_doc_count(folder_id: str, delta: int):
    """Increment (or decrement) the doc_count for a folder."""
    registry = _load_registry()
    if folder_id in registry:
        registry[folder_id]["doc_count"] = max(
            0, registry[folder_id]["doc_count"] + delta
        )
        _save_registry(registry)


def delete_folder(folder_id: str) -> dict:
    """
    Remove a folder from the registry.
    Does NOT delete the Qdrant collection — call vector_store.drop_collection() separately.
    Returns the deleted folder record.
    """
    registry = _load_registry()
    if folder_id not in registry:
        raise KeyError(f"Folder '{folder_id}' not found.")

    folder = registry.pop(folder_id)
    _save_registry(registry)

    logger.info(f"Removed folder '{folder['name']}' from registry")
    return folder


def resolve_folder(folder_id_or_name: str) -> dict:
    """
    Convenience: accept either a folder_id or a folder name.
    Tries ID first, then falls back to name lookup.
    Raises ValueError with a helpful message if neither matches.
    """
    registry = _load_registry()

    # Try direct ID match
    if folder_id_or_name in registry:
        return registry[folder_id_or_name]

    # Try name match (case-insensitive)
    for folder in registry.values():
        if folder["name"].lower() == folder_id_or_name.lower():
            return folder

    available = "\n".join(
        f"  {f['folder_id']}  {f['name']}"
        for f in sorted(registry.values(), key=lambda x: x["name"])
    )
    raise ValueError(
        f"No folder found matching '{folder_id_or_name}'.\n"
        f"Available folders:\n{available or '  (none yet)'}"
    )
