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

Here is the chunk text:
<chunk>
{chunk_text}
</chunk>

Write a single sentence (maximum 40 words) that situates this chunk in its document context.
The sentence must state:
1. What document this is from (use the filename as-is)
2. What chapter/section it belongs to (use the breadcrumb)
3. What the chunk is specifically about

Be precise and technical. Do not be vague. Do not repeat the chunk text verbatim.
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


def build_context_sentence(chunk: ParsedChunk) -> str:
    """
    Generate a context sentence for a single chunk.
    Returns the sentence, or an empty string if the API call fails after retries.
    """
    headings_str = " > ".join(chunk.headings) if chunk.headings else "Unknown section"
    pages_str = ", ".join(str(p) for p in chunk.page_numbers) if chunk.page_numbers else "unknown"

    # For tables, add a hint so Claude phrases it appropriately
    chunk_text = chunk.text
    if chunk.is_table:
        chunk_text = f"[TABLE]\n{chunk_text}"

    prompt = CONTEXT_PROMPT.format(
        source_file=chunk.source_file,
        doc_type=chunk.doc_type,
        headings=headings_str,
        page_numbers=pages_str,
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


def enrich_chunks_with_context(
    chunks: list[ParsedChunk],
    show_progress: bool = True,
) -> list[tuple[ParsedChunk, str]]:
    """
    Enrich all chunks with context sentences.

    Returns a list of (chunk, context_enriched_text) pairs where
    context_enriched_text = context_sentence + "\\n\\n" + chunk.text

    This enriched text is what gets embedded.
    The original chunk.text is kept for display to users.

    Args:
        chunks: List of parsed chunks
        show_progress: Whether to log progress (useful for large documents)

    Returns:
        List of (ParsedChunk, enriched_text) tuples ready for embedding
    """
    results = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if show_progress and (i % 10 == 0 or i == total - 1):
            logger.info(f"  Contextualizing chunk {i+1}/{total}...")

        context_sentence = build_context_sentence(chunk)
        enriched_text = f"{context_sentence}\n\n{chunk.text}"
        results.append((chunk, enriched_text))

    logger.info(f"  Contextualization complete: {len(results)} chunks enriched")
    return results
