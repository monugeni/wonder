"""
tree_retriever.py — hybrid retrieval: vector coarse filter + tree reasoning

Two-stage process per deep_query call:

  Stage 1 — Vector coarse filter (existing infrastructure, unchanged)
    Run a normal vector search across the folder to identify the top-N documents
    that contain semantically relevant chunks. This is fast (milliseconds) and
    gives us a shortlist of 1-3 documents worth drilling into.

  Stage 2 — Tree reasoning per candidate document
    For each candidate document that has a tree:
      a. Load the tree's summary view (titles + summaries only, no full_text)
      b. Send tree + query to Claude → Claude returns node_ids it wants to read
      c. Fetch full_text of those nodes from the tree
      d. If a fetched section mentions a cross-reference ("see Appendix B"),
         optionally follow it by looking up the referenced node

  Stage 3 — Answer synthesis
    Combine retrieved node texts across all candidate documents and ask Claude
    to produce a final answer with citations (document name, section, page range).

Cross-reference following:
    Engineering documents are full of "refer to Table 4.3-1", "see Appendix G",
    "per Section 6.2.1". After retrieving initial nodes, we scan the text for
    these patterns and attempt to resolve the reference within the same tree.
    This is the key capability that pure vector search cannot do.
"""

import json
import re
from typing import Optional

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from embedder import embed_query
from tree_store import (
    get_nodes_by_ids,
    load_tree,
    tree_exists,
    tree_summary_view,
)
from vector_store import search


_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


# ── Prompts ───────────────────────────────────────────────────────────────────

NODE_SELECTION_PROMPT = """\
You are a retrieval assistant for technical engineering documents.

A user has asked the following question:
<question>
{question}
</question>

Below is the section index for the document "{source_file}".
Each node shows its ID, title, page range, and a brief summary.
The structure is hierarchical — indentation shows parent/child sections.

<section_index>
{index_text}
</section_index>

Your task: identify which section node IDs are most likely to contain the answer.

Rules:
- Select only nodes whose summary directly indicates relevant content
- Prefer specific subsections over broad parent sections
- If a parent section is vague but a child section is clearly relevant, select the child
- Select at most {max_nodes} nodes
- If no section looks relevant, return an empty list

Return ONLY a JSON object in this exact format, nothing else:
{{
  "thinking": "brief reasoning about which sections are relevant",
  "node_ids": ["n_0001", "n_0005"]
}}"""

ANSWER_SYNTHESIS_PROMPT = """\
You are a technical engineering assistant. Answer the question using ONLY the retrieved document sections provided below.

Question: {question}

Retrieved sections:
{sections_text}

Instructions:
- Answer directly and technically
- Cite each claim: (Document: filename, Section: title, Pages: X-Y)
- If a section mentions "see Section X" or "refer to Appendix Y" and that reference was also retrieved, incorporate it
- If the retrieved sections do not contain enough information to answer, say so clearly
- Do not invent specifications, values, or standards not present in the sections

Answer:"""

CROSS_REF_PATTERN = re.compile(
    r"(?:see|refer to|per|as per|refer|reference|section|clause|appendix|table|figure)"
    r"\s+([A-Z]?\d[\d\.]*\w*)",
    re.IGNORECASE,
)


# ── Stage 1: Vector coarse filter ─────────────────────────────────────────────

def _vector_coarse_filter(
    collection_name: str,
    query: str,
    top_k: int = 20,
    doc_type: Optional[str] = None,
) -> list[str]:
    """
    Run vector search and return deduplicated source_file names,
    ordered by best hit score for that document.
    """
    query_vec = embed_query(query)
    results = search(
        collection_name=collection_name,
        query_vector=query_vec,
        top_k=top_k,
        doc_type=doc_type,
        query_text=query,
    )

    # Deduplicate: keep the highest score per document
    doc_scores: dict[str, float] = {}
    for r in results:
        sf = r["source_file"]
        if sf not in doc_scores or r["score"] > doc_scores[sf]:
            doc_scores[sf] = r["score"]

    # Sort by descending score
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [sf for sf, _ in ranked]


# ── Stage 2: Tree reasoning ───────────────────────────────────────────────────

def _format_index_text(nodes: list[dict], indent: int = 0) -> str:
    """Render the summary tree as indented text for the LLM prompt."""
    lines = []
    pad = "  " * indent
    for node in nodes:
        pages = (
            f"pp. {node['page_start']}-{node['page_end']}"
            if node.get("page_start") and node.get("page_end")
            else ""
        )
        summary = node.get("summary", "")
        lines.append(
            f"{pad}[{node['node_id']}] {node['title']} {pages}\n"
            f"{pad}  Summary: {summary}"
        )
        if node.get("children"):
            lines.append(_format_index_text(node["children"], indent + 1))
    return "\n".join(lines)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _select_nodes(question: str, source_file: str, index_text: str, max_nodes: int = 5) -> dict:
    """Ask Claude which nodes to retrieve from this document's tree."""
    prompt = NODE_SELECTION_PROMPT.format(
        question=question,
        source_file=source_file,
        index_text=index_text,
        max_nodes=max_nodes,
    )
    resp = _client.messages.create(
        model=config.CONTEXT_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Node selection returned non-JSON: {raw[:200]}")
        return {"thinking": "", "node_ids": []}


def _find_cross_references(text: str) -> list[str]:
    """Extract cross-reference targets from section text, e.g. '4.3.1', 'Appendix G'."""
    matches = CROSS_REF_PATTERN.findall(text)
    return list(set(matches))


def _resolve_cross_refs(tree: dict, refs: list[str], already_fetched: set[str]) -> list[dict]:
    """
    Try to find tree nodes matching the cross-reference targets.
    Matches by checking if the ref string appears in a node's title.
    """
    resolved = []
    for ref in refs:
        ref_lower = ref.lower()
        for node in _iter_all_nodes(tree["nodes"]):
            if node["node_id"] in already_fetched:
                continue
            if ref_lower in node["title"].lower():
                resolved.append(node)
                already_fetched.add(node["node_id"])
                logger.debug(f"  Cross-ref '{ref}' → node '{node['title']}'")
                break
    return resolved


def _iter_all_nodes(nodes: list[dict]):
    for node in nodes:
        yield node
        yield from _iter_all_nodes(node["children"])


def _reason_over_tree(
    question: str,
    folder_id: str,
    source_file: str,
    follow_xrefs: bool = True,
    max_nodes: int = 5,
) -> tuple[list[dict], str]:
    """
    Stage 2: Load a document's tree, ask Claude which nodes to fetch,
    retrieve those nodes' full text, optionally follow cross-references.

    Returns (nodes, thinking) — a list of retrieved nodes and Claude's reasoning.
    """
    tree = load_tree(folder_id, source_file)
    if tree is None:
        logger.warning(f"  No tree for {source_file} — skipping tree reasoning")
        return [], ""

    summary_view = tree_summary_view(tree)
    index_text = _format_index_text(summary_view["nodes"])

    if not index_text.strip():
        logger.warning(f"  Empty section index for {source_file}")
        return [], ""

    # Ask Claude which nodes to read
    selection = _select_nodes(question, source_file, index_text, max_nodes)
    node_ids = selection.get("node_ids", [])
    thinking = selection.get("thinking", "")

    logger.info(
        f"  [{source_file}] Node selection: {node_ids} | {thinking[:80]}"
    )

    if not node_ids:
        return [], thinking

    # Fetch the full nodes from the tree
    nodes = get_nodes_by_ids(tree, node_ids)
    fetched_ids = {n["node_id"] for n in nodes}

    # Cross-reference following
    if follow_xrefs and nodes:
        all_text = " ".join(n.get("full_text", "") for n in nodes)
        refs = _find_cross_references(all_text)
        if refs:
            logger.info(f"  Cross-references found: {refs}")
            extra = _resolve_cross_refs(tree, refs, fetched_ids)
            nodes.extend(extra)

    return nodes, thinking


# ── Stage 3: Answer synthesis ─────────────────────────────────────────────────

def _format_sections_for_synthesis(doc_nodes: list[tuple[str, list[dict]]]) -> str:
    """
    Format retrieved nodes from multiple documents for the synthesis prompt.
    doc_nodes: list of (source_file, [node, ...])
    """
    parts = []
    for source_file, nodes in doc_nodes:
        for node in nodes:
            pages = (
                f"{node.get('page_start')}-{node.get('page_end')}"
                if node.get("page_start")
                else "unknown"
            )
            header = (
                f"--- Document: {source_file} | "
                f"Section: {node['title']} | "
                f"Pages: {pages} ---"
            )
            parts.append(header + "\n" + node.get("full_text", "").strip())

    return "\n\n".join(parts)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _synthesise_answer(question: str, sections_text: str) -> str:
    """Generate the final answer from retrieved sections."""
    prompt = ANSWER_SYNTHESIS_PROMPT.format(
        question=question,
        sections_text=sections_text,
    )
    resp = _client.messages.create(
        model=config.CONTEXT_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── Public entry point ────────────────────────────────────────────────────────

def deep_query(
    collection_name: str,
    folder_id: str,
    question: str,
    top_documents: int = 3,
    max_nodes_per_doc: int = 5,
    doc_type: Optional[str] = None,
    follow_xrefs: bool = True,
) -> dict:
    """
    Hybrid deep query: vector coarse filter + tree reasoning.

    Args:
        collection_name:  Qdrant collection for this folder
        folder_id:        For loading trees
        question:         User's question in natural language
        top_documents:    How many documents to drill into after coarse filter
        max_nodes_per_doc: Max sections to retrieve per document
        doc_type:         Optional filter for vector stage
        follow_xrefs:     Whether to auto-follow cross-references

    Returns:
        {
            "answer":    str,
            "sources":   [{"source_file", "section", "pages", "score"}, ...],
            "reasoning": {source_file: thinking_text, ...},
        }
    """
    logger.info(f"deep_query: '{question[:80]}...' | folder={folder_id}")

    # Stage 1: Vector coarse filter — find candidate documents
    candidate_docs = _vector_coarse_filter(
        collection_name=collection_name,
        query=question,
        top_k=top_documents * 8,  # fetch more chunks to get reliable doc ranking
        doc_type=doc_type,
    )[:top_documents]

    if not candidate_docs:
        return {
            "answer": "No relevant documents found in this folder for your query.",
            "sources": [],
            "reasoning": {},
        }

    logger.info(f"  Stage 1 candidates: {candidate_docs}")

    # Filter to docs that have trees; fall back gracefully for those that don't
    tree_docs = [sf for sf in candidate_docs if tree_exists(folder_id, sf)]
    no_tree_docs = [sf for sf in candidate_docs if not tree_exists(folder_id, sf)]

    if no_tree_docs:
        logger.warning(
            f"  No tree available for: {no_tree_docs} — "
            "these will be skipped in tree reasoning (re-ingest to build trees)"
        )

    if not tree_docs:
        return {
            "answer": (
                "Candidate documents were found but none have a section tree yet. "
                "Re-ingest the documents to enable deep query."
            ),
            "sources": [],
            "reasoning": {},
        }

    # Stage 2: Tree reasoning per candidate document
    reasoning: dict[str, str] = {}
    doc_nodes: list[tuple[str, list[dict]]] = []

    for source_file in tree_docs:
        nodes, thinking = _reason_over_tree(
            question=question,
            folder_id=folder_id,
            source_file=source_file,
            follow_xrefs=follow_xrefs,
            max_nodes=max_nodes_per_doc,
        )
        if thinking:
            reasoning[source_file] = thinking
        if nodes:
            doc_nodes.append((source_file, nodes))

    if not doc_nodes:
        return {
            "answer": (
                "The section trees were searched but no relevant sections were identified. "
                "Try rephrasing or use the regular query tool."
            ),
            "sources": [],
            "reasoning": reasoning,
        }

    # Stage 3: Synthesise answer
    sections_text = _format_sections_for_synthesis(doc_nodes)
    answer = _synthesise_answer(question, sections_text)

    # Build source citations
    sources = []
    for source_file, nodes in doc_nodes:
        for node in nodes:
            sources.append({
                "source_file": source_file,
                "section": node["title"],
                "page_start": node.get("page_start"),
                "page_end": node.get("page_end"),
            })

    return {
        "answer": answer,
        "sources": sources,
        "reasoning": reasoning,
    }
