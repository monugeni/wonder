"""
tree_builder.py — build a section tree from a DoclingDocument

Docling parses every document into a DoclingDocument with a body tree:
headings, paragraphs, tables all nested under their parent headings via
JSON pointer references. We walk that tree once and produce a flat list of
sections, then nest them by heading level.

Each section node:
    node_id    : "n_0001"
    title      : heading text, e.g. "4.3 Pipe Bends"
    level      : 1 = chapter, 2 = section, 3 = subsection ...
    page_start : first page of this section
    page_end   : last page (may span several pages)
    summary    : 1-2 sentence LLM summary — used at query time so Claude
                 can decide which nodes are relevant WITHOUT reading full text
    full_text  : complete content of this section (paragraphs + tables)
    children   : list of child section nodes (same schema, nested)

At query time only the summaries are loaded into context. Full text is
fetched only for nodes Claude selects. This is the core of the hybrid approach:
vector search narrows to which documents, tree reasoning narrows to which sections.
"""

import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from parser import detect_doc_type


_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

TREE_STORE_DIR = config.TABLE_STORE_DIR.parent / "trees"

SUMMARY_PROMPT = """\
You are indexing a section from a technical engineering document.

Document: {source_file}
Section: {breadcrumb}
Pages: {pages}

Content:
<content>
{content}
</content>

Write 1-2 sentences (max 60 words) summarising what this section covers.
Be specific: include key terms, materials, standards, limits, or conditions mentioned.
A reader should know from your summary alone whether this section might answer their question.
Output only the summary, nothing else."""


_node_lock = threading.Lock()
_node_counter = 0


def _next_id() -> str:
    global _node_counter
    with _node_lock:
        _node_counter += 1
        return f"n_{_node_counter:04d}"


def _reset_counter():
    global _node_counter
    with _node_lock:
        _node_counter = 0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _summarise(source_file: str, breadcrumb: str, pages: str, content: str) -> str:
    resp = _client.messages.create(
        model=config.CONTEXT_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(
            source_file=source_file,
            breadcrumb=breadcrumb,
            pages=pages,
            content=content[:3000],
        )}],
    )
    return resp.content[0].text.strip()


# ── Docling tree walking ───────────────────────────────────────────────────────

def _label(item) -> str:
    try:
        lbl = item.label
        return str(lbl.value) if hasattr(lbl, "value") else str(lbl)
    except Exception:
        return ""


def _is_heading(item) -> bool:
    lbl = _label(item).lower()
    return "heading" in lbl or lbl in ("title", "section_header", "chapter")


def _is_table(item) -> bool:
    return "table" in _label(item).lower()


def _item_text(item) -> str:
    try:
        if hasattr(item, "text") and item.text:
            return item.text
        if hasattr(item, "export_to_markdown"):
            return item.export_to_markdown()
    except Exception:
        pass
    return ""


def _item_pages(item) -> list[int]:
    pages = set()
    try:
        for prov in (item.prov or []):
            if hasattr(prov, "page_no"):
                pages.add(prov.page_no)
    except Exception:
        pass
    return sorted(pages)


def _heading_level(item) -> int:
    """Estimate heading depth from label name or numbered text like '4.3.1'."""
    lbl = _label(item)
    m = re.search(r"level[_\s]?(\d)", lbl, re.IGNORECASE)
    if m:
        return int(m.group(1))
    text = _item_text(item).split()[0] if _item_text(item).split() else ""
    dots = text.count(".")
    return min(dots + 1, 4)


def _deref(doc, ref: str):
    """Follow a JSON pointer like '#/texts/5' into a DoclingDocument."""
    try:
        parts = ref.lstrip("#/").split("/")
        obj = doc
        for p in parts:
            obj = obj[int(p)] if isinstance(obj, (list, tuple)) else getattr(obj, p, None)
            if obj is None:
                return None
        return obj
    except Exception:
        return None


def _walk_body(doc) -> list:
    """
    DFS walk of the Docling body tree, returning items in reading order.
    Returns list of DocItem objects (headings, paragraphs, tables, etc.)
    """
    result = []

    def _visit(node):
        item = _deref(doc, node.self_ref) if hasattr(node, "self_ref") else node
        if item is not None and item is not node:
            result.append(item)
        elif item is node:
            result.append(item)

        children = getattr(node, "children", []) or []
        for child_ref in children:
            child = None
            if hasattr(child_ref, "cref"):
                child = _deref(doc, child_ref.cref)
            elif hasattr(child_ref, "self_ref"):
                child = child_ref
            if child is not None:
                _visit(child)

    body = getattr(doc, "body", None)
    if body is None:
        return result

    for child_ref in (getattr(body, "children", []) or []):
        child = None
        if hasattr(child_ref, "cref"):
            child = _deref(doc, child_ref.cref)
        elif hasattr(child_ref, "self_ref"):
            child = child_ref
        if child is not None:
            _visit(child)

    return result


# ── Section grouping ──────────────────────────────────────────────────────────

def _items_to_flat_sections(items: list, source_file: str) -> list[dict]:
    """
    Group body items into sections. A new section starts at each heading.
    Tables and paragraphs accumulate into the current section's full_text.
    """
    sections: list[dict] = []
    current: Optional[dict] = None

    def _new_section(title: str, level: int, pages: list[int]) -> dict:
        return {
            "_id": _next_id(),
            "title": title,
            "level": level,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "full_text": "",
        }

    def _extend_pages(sec: dict, pages: list[int]):
        if not pages:
            return
        if sec["page_start"] is None:
            sec["page_start"] = min(pages)
        else:
            sec["page_start"] = min(sec["page_start"], min(pages))
        if sec["page_end"] is None:
            sec["page_end"] = max(pages)
        else:
            sec["page_end"] = max(sec["page_end"], max(pages))

    def _append_text(sec: dict, text: str):
        if text.strip():
            sep = "\n\n" if sec["full_text"] else ""
            sec["full_text"] += sep + text.strip()

    for item in items:
        pages = _item_pages(item)
        text = _item_text(item)

        if _is_heading(item) and text.strip():
            if current:
                sections.append(current)
            current = _new_section(text.strip(), _heading_level(item), pages)

        elif _is_table(item):
            if current is None:
                current = _new_section(source_file, 0, pages)
            _extend_pages(current, pages)
            table_md = text or ""
            if table_md:
                _append_text(current, f"[TABLE]\n{table_md}")

        else:
            if text.strip():
                if current is None:
                    current = _new_section(source_file, 0, pages)
                _extend_pages(current, pages)
                _append_text(current, text)

    if current:
        sections.append(current)

    return sections


def _nest(flat: list[dict]) -> list[dict]:
    """Convert a flat list of sections into a nested tree by heading level."""
    roots: list[dict] = []
    stack: list[dict] = []  # active ancestor chain

    for sec in flat:
        node = {
            "node_id": sec["_id"],
            "title": sec["title"],
            "level": sec["level"],
            "page_start": sec["page_start"],
            "page_end": sec["page_end"],
            "summary": "",        # filled by summarisation pass below
            "full_text": sec["full_text"],
            "children": [],
        }

        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)

        stack.append(node)

    return roots


# ── Summarisation pass ────────────────────────────────────────────────────────

def _breadcrumb(node: dict, ancestors: list[str]) -> str:
    return " > ".join(ancestors + [node["title"]]) if ancestors else node["title"]


def _collect_summary_tasks(nodes: list[dict], source_file: str, ancestors: list[str] = []) -> list[tuple[dict, str, str, str]]:
    """Collect all (node, source_file, breadcrumb, content) tuples for parallel summarisation."""
    tasks = []
    for node in nodes:
        if node["children"]:
            tasks.extend(_collect_summary_tasks(node["children"], source_file, ancestors + [node["title"]]))

        pages_str = (
            f"{node['page_start']}-{node['page_end']}"
            if node["page_start"] and node["page_end"]
            else "unknown"
        )
        breadcrumb = _breadcrumb(node, ancestors)
        content = node["full_text"]

        if not content.strip():
            child_titles = ", ".join(c["title"] for c in node["children"][:5])
            content = f"Section covering: {child_titles}" if child_titles else node["title"]

        tasks.append((node, source_file, breadcrumb, pages_str, content))
    return tasks


def _summarise_tree(nodes: list[dict], source_file: str, ancestors: list[str] = [], max_workers: int = 10):
    """
    Generate summaries for all tree nodes using parallel API calls.
    """
    tasks = _collect_summary_tasks(nodes, source_file, ancestors)
    total = len(tasks)
    done_count = 0

    def _do_summary(task):
        node, sf, breadcrumb, pages_str, content = task
        try:
            node["summary"] = _summarise(sf, breadcrumb, pages_str, content)
            logger.debug(f"  Summarised: {node['title'][:60]}")
        except Exception as e:
            logger.warning(f"  Summary failed for '{node['title']}': {e}")
            node["summary"] = f"Section: {node['title']} (pages {pages_str})"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_do_summary, t) for t in tasks]
        for future in as_completed(futures):
            future.result()  # raise any unexpected exceptions
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                logger.info(f"  Summarised {done_count}/{total} sections...")


# ── Public API ────────────────────────────────────────────────────────────────

def build_tree(doc, source_file: str, show_progress: bool = True) -> dict:
    """
    Build a section tree from a Docling DoclingDocument.

    Args:
        doc:         The DoclingDocument returned by DocumentConverter
        source_file: Filename, e.g. 'P269-PIPE-SPEC-REV2.pdf'

    Returns:
        Tree dict:
            {
                "source_file": str,
                "doc_type":    str,
                "nodes":       [...]   # top-level nodes, each with children
            }
    """
    _reset_counter()

    if show_progress:
        logger.info(f"  Building section tree for {source_file}...")

    items = _walk_body(doc)
    if show_progress:
        logger.info(f"  Walking body: {len(items)} items found")

    flat = _items_to_flat_sections(items, source_file)
    if show_progress:
        logger.info(f"  Grouped into {len(flat)} sections")

    nodes = _nest(flat)

    if show_progress:
        logger.info(f"  Generating section summaries (1 LLM call per section)...")

    _summarise_tree(nodes, source_file)

    total_nodes = sum(1 for _ in _iter_all_nodes(nodes))
    if show_progress:
        logger.info(f"  Tree complete: {total_nodes} nodes across {len(nodes)} top-level sections")

    return {
        "source_file": source_file,
        "doc_type": detect_doc_type(source_file),
        "nodes": nodes,
    }


def _iter_all_nodes(nodes: list[dict]):
    """Flat iterator over every node in the tree regardless of depth."""
    for node in nodes:
        yield node
        yield from _iter_all_nodes(node["children"])


def node_count(tree: dict) -> int:
    return sum(1 for _ in _iter_all_nodes(tree["nodes"]))
