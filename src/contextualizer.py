"""
contextualizer.py — Anthropic Contextual Retrieval

For each chunk, calls Claude to prepend a short context sentence that situates
the chunk within its document, chapter, and section. This is the key step that
solves the polysemy problem: "bend" in a piping spec gets a completely different
embedding than "bend" in an ergonomics manual because the context prefix differs.

The context sentence is prepended to the chunk text BEFORE embedding.
The original text is stored separately so it can be shown to users in results.

Batching: Contextualizing one chunk per API call is expensive. This module
processes chunks in batches and uses claude-haiku for cost efficiency.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from parser import ParsedChunk


# One-time client initialization
_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

CONTEXT_PROMPT = """\
You are processing a chunk from a large technical engineering document for a RAG system.

Document name: {source_file}
Document type: {doc_type}
Section breadcrumb: {headings}
Pages: {page_numbers}
{doc_outline}{neighbors}
Here is the chunk text:
<chunk>
{chunk_text}
</chunk>

Write a single sentence (maximum 50 words) that situates this chunk so a search engine can match it to relevant queries.

Requirements:
1. State the document filename as-is.
2. Use the document outline and neighboring chunks to determine which section this chunk belongs to.
   Do NOT say "Unknown section". Place it in the correct section from the outline.
3. Describe what specific information this chunk contains using the KEY TERMS a user would search for.
   For tables: describe what the table's columns and rows represent AND which section they belong to.
   For criteria/requirements: name the type (e.g. "technical qualification criteria", "financial eligibility", "safety requirements").

Be precise and technical. Use domain-specific keywords. Do not be vague.
Output only the single sentence, nothing else.

Example output:
"This chunk is from 'P269-PIPE-SPEC-REV2.pdf', Section 4.3 Pipe Bends, and specifies minimum bend radius requirements for high-pressure process piping."
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def _call_claude(prompt: str) -> str:
    """Call Claude with retry logic for transient API errors."""
    response = _client.messages.create(
        model=config.CONTEXT_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def build_context_sentence(
    chunk: ParsedChunk,
    doc_outline: str = "",
    neighbor_text: str = "",
) -> str:
    """
    Generate a context sentence for a single chunk.
    Returns the sentence, or an empty string if the API call fails after retries.

    Args:
        chunk: The chunk to contextualize
        doc_outline: Ordered list of section headings from the document
        neighbor_text: Text snippets from surrounding chunks for context
    """
    headings_str = " > ".join(chunk.headings) if chunk.headings else "Unknown section"
    pages_str = ", ".join(str(p) for p in chunk.page_numbers) if chunk.page_numbers else "unknown"

    # For tables, add a hint so Claude phrases it appropriately
    chunk_text = chunk.text
    if chunk.is_table:
        chunk_text = f"[TABLE]\n{chunk_text}"

    # Build optional context blocks
    outline_block = f"\nDocument outline (section headings in order):\n{doc_outline}\n" if doc_outline else ""
    neighbor_block = f"\nSurrounding content for context:\n{neighbor_text}\n" if neighbor_text else ""

    prompt = CONTEXT_PROMPT.format(
        source_file=chunk.source_file,
        doc_type=chunk.doc_type,
        headings=headings_str,
        page_numbers=pages_str,
        doc_outline=outline_block,
        neighbors=neighbor_block,
        chunk_text=chunk_text[:1500],  # Truncate very large chunks for context call
    )

    try:
        sentence = _call_claude(prompt)
        logger.debug(f"  Context: {sentence[:80]}...")
        return sentence
    except Exception as e:
        logger.warning(f"  Context generation failed for chunk {chunk.chunk_id}: {e}")
        # Fallback: construct a basic context sentence from metadata
        return (
            f"This content is from '{chunk.source_file}', "
            f"section: {headings_str}."
        )


def _build_doc_outline(chunks: list[ParsedChunk]) -> str:
    """Build an ordered list of unique section headings from the document."""
    seen = set()
    headings_list = []
    for chunk in chunks:
        if chunk.headings:
            key = " > ".join(chunk.headings)
            if key not in seen:
                seen.add(key)
                headings_list.append(key)
    return "\n".join(f"- {h}" for h in headings_list) if headings_list else ""


def _build_neighbor_text(chunks: list[ParsedChunk], index: int) -> str:
    """Build context from neighboring chunks (prev/next headings and text snippets)."""
    parts = []
    if index > 0:
        prev = chunks[index - 1]
        prev_hdg = " > ".join(prev.headings) if prev.headings else ""
        prev_txt = prev.text[:150].replace("\n", " ")
        if prev_hdg:
            parts.append(f"[PRECEDING CHUNK — Section: {prev_hdg}] {prev_txt}")
        else:
            parts.append(f"[PRECEDING CHUNK] {prev_txt}")
    if index < len(chunks) - 1:
        nxt = chunks[index + 1]
        nxt_hdg = " > ".join(nxt.headings) if nxt.headings else ""
        nxt_txt = nxt.text[:150].replace("\n", " ")
        if nxt_hdg:
            parts.append(f"[FOLLOWING CHUNK — Section: {nxt_hdg}] {nxt_txt}")
        else:
            parts.append(f"[FOLLOWING CHUNK] {nxt_txt}")
    return "\n".join(parts)


def enrich_chunks_with_context(
    chunks: list[ParsedChunk],
    show_progress: bool = True,
    max_workers: int = None,
    on_progress: callable = None,
    cancel_check: callable = None,
) -> list[tuple[ParsedChunk, str]]:
    """
    Enrich all chunks with context sentences using parallel API calls.

    Returns a list of (chunk, context_enriched_text) pairs where
    context_enriched_text = context_sentence + "\\n\\n" + chunk.text

    This enriched text is what gets embedded.
    The original chunk.text is kept for display to users.

    Args:
        chunks: List of parsed chunks
        show_progress: Whether to log progress (useful for large documents)
        max_workers: Max concurrent API calls (default 10)

    Returns:
        List of (ParsedChunk, enriched_text) tuples ready for embedding
    """
    if max_workers is None:
        max_workers = config.LLM_WORKERS

    total = len(chunks)
    # Pre-allocate results list to maintain chunk order
    context_sentences = [""] * total
    done_count = 0

    # Build document-level context for better section inference
    doc_outline = _build_doc_outline(chunks)

    def _process(index: int, chunk: ParsedChunk) -> tuple[int, str]:
        neighbor_text = _build_neighbor_text(chunks, index)
        return index, build_context_sentence(chunk, doc_outline, neighbor_text)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process, i, chunk): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            # Check for cancellation — cancel remaining futures
            if cancel_check:
                try:
                    cancel_check()
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise
            idx, sentence = future.result()
            context_sentences[idx] = sentence
            done_count += 1
            if show_progress and (done_count % 10 == 0 or done_count == total):
                logger.info(f"  Contextualized {done_count}/{total} chunks...")
            if on_progress and (done_count % 5 == 0 or done_count == total):
                on_progress(done_count, total)

    results = []
    for i, chunk in enumerate(chunks):
        # Prepend heading breadcrumb directly into the embedded text so the
        # embedding model always sees the section hierarchy, even if the
        # LLM context sentence didn't capture it correctly.
        heading_prefix = ""
        if chunk.headings:
            heading_prefix = f"[{' > '.join(chunk.headings)}]\n"
        enriched_text = f"{heading_prefix}{context_sentences[i]}\n\n{chunk.text}"
        results.append((chunk, enriched_text))

    logger.info(f"  Contextualization complete: {len(results)} chunks enriched")
    return results
