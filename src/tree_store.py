"""
tree_store.py — persist and retrieve document section trees

Trees are stored as JSON files under:
    {TABLE_STORE_DIR}/../trees/{folder_id}/{source_file_stem}.tree.json

Each file contains the full tree dict produced by tree_builder.build_tree().

The tree is small enough to load entirely into memory — even a 300-page
engineering spec typically produces 80-150 section nodes, and only the
summary fields (not full_text) are loaded into the LLM's context at query time.
"""

import json
from pathlib import Path

from loguru import logger

from config import config

TREE_STORE_DIR = config.TABLE_STORE_DIR.parent / "trees"


def _tree_path(folder_id: str, source_file: str) -> Path:
    folder_dir = TREE_STORE_DIR / folder_id
    folder_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_file).stem
    return folder_dir / f"{stem}.tree.json"


def save_tree(folder_id: str, tree: dict):
    """Persist a document tree to disk."""
    path = _tree_path(folder_id, tree["source_file"])
    path.write_text(json.dumps(tree, indent=2, ensure_ascii=False))
    logger.info(f"  Tree saved: {path.name}")


def load_tree(folder_id: str, source_file: str) -> dict | None:
    """Load a document tree. Returns None if no tree exists for this document."""
    path = _tree_path(folder_id, source_file)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def tree_exists(folder_id: str, source_file: str) -> bool:
    return _tree_path(folder_id, source_file).exists()


def get_doc_summary(folder_id: str, source_file: str) -> str | None:
    """Return just the document-level summary, or None if no tree exists."""
    path = _tree_path(folder_id, source_file)
    if not path.exists():
        return None
    try:
        tree = json.loads(path.read_text())
        return tree.get("doc_summary")
    except Exception:
        return None


def list_trees(folder_id: str) -> list[str]:
    """Return source_file names for all trees in a folder."""
    folder_dir = TREE_STORE_DIR / folder_id
    if not folder_dir.exists():
        return []
    return [p.stem.replace(".tree", "") for p in folder_dir.glob("*.tree.json")]


def rename_tree(folder_id: str, old_source_file: str, new_source_file: str) -> bool:
    """Rename a tree file and update source_file inside it. Returns True if renamed."""
    old_path = _tree_path(folder_id, old_source_file)
    if not old_path.exists():
        return False
    tree = json.loads(old_path.read_text())
    tree["source_file"] = new_source_file
    new_path = _tree_path(folder_id, new_source_file)
    new_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False))
    if old_path != new_path:
        old_path.unlink()
    logger.info(f"  Tree renamed: {old_path.name} -> {new_path.name}")
    return True


def delete_tree(folder_id: str, source_file: str) -> bool:
    """Delete a tree. Returns True if it existed."""
    path = _tree_path(folder_id, source_file)
    if path.exists():
        path.unlink()
        logger.info(f"  Tree deleted: {path.name}")
        return True
    return False


def delete_all_trees(folder_id: str) -> int:
    """Delete all trees for a folder. Returns count of trees deleted."""
    folder_dir = TREE_STORE_DIR / folder_id
    if not folder_dir.exists():
        return 0
    count = 0
    for p in folder_dir.glob("*.tree.json"):
        p.unlink()
        count += 1
    # Remove the now-empty folder directory
    try:
        folder_dir.rmdir()
    except OSError:
        pass
    if count:
        logger.info(f"  Deleted {count} tree(s) for folder {folder_id}")
    return count


def tree_summary_view(tree: dict) -> dict:
    """
    Return a lightweight copy of the tree with only summaries and titles —
    no full_text. This is what gets loaded into the LLM's context at query time.

    Keeps the same nested structure so Claude can reason about hierarchy.
    """
    def _strip(nodes):
        return [
            {
                "node_id": n["node_id"],
                "title": n["title"],
                "level": n["level"],
                "page_start": n["page_start"],
                "page_end": n["page_end"],
                "summary": n["summary"],
                "children": _strip(n["children"]),
            }
            for n in nodes
        ]

    return {
        "source_file": tree["source_file"],
        "doc_type": tree["doc_type"],
        "doc_summary": tree.get("doc_summary", ""),
        "nodes": _strip(tree["nodes"]),
    }


def get_node_by_id(tree: dict, node_id: str) -> dict | None:
    """Find and return a node (with full_text) by its node_id."""
    def _find(nodes):
        for node in nodes:
            if node["node_id"] == node_id:
                return node
            found = _find(node["children"])
            if found:
                return found
        return None

    return _find(tree["nodes"])


def get_nodes_by_ids(tree: dict, node_ids: list[str]) -> list[dict]:
    """Return all nodes matching the given IDs, preserving order."""
    result = []
    for nid in node_ids:
        node = get_node_by_id(tree, nid)
        if node:
            result.append(node)
    return result
