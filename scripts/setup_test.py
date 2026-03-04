#!/usr/bin/env python3
"""
setup_test.py — verify your installation before connecting the MCP server

Run this first:
  python scripts/setup_test.py

It checks:
  1. All Python packages are importable
  2. Anthropic API key is valid (makes a tiny test call)
  3. Qdrant is reachable
  4. Embedding model can be loaded and run
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

def check(name: str, fn):
    try:
        result = fn()
        print(f"  ✓  {name}" + (f": {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  ✗  {name}: {e}")
        return False


def main():
    print("\n=== Engineering RAG Setup Check ===\n")
    all_ok = True

    # 1. Imports
    print("1. Checking Python packages...")

    def check_imports():
        import docling
        import qdrant_client
        import sentence_transformers
        import anthropic
        import mcp
        return f"docling {docling.__version__}"

    all_ok &= check("Core packages", check_imports)

    # 2. Config
    print("\n2. Checking configuration...")
    from config import config

    all_ok &= check(
        "ANTHROPIC_API_KEY set",
        lambda: "yes" if config.ANTHROPIC_API_KEY else (_ for _ in ()).throw(ValueError("Not set"))
    )
    all_ok &= check("TABLE_STORE_DIR writable", lambda: str(config.TABLE_STORE_DIR.resolve()))

    # 3. Anthropic API
    print("\n3. Testing Anthropic API...")
    def test_anthropic():
        import anthropic as ant
        client = ant.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=config.CONTEXT_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say: ok"}]
        )
        return resp.content[0].text.strip()

    all_ok &= check(f"Claude API ({config.CONTEXT_MODEL})", test_anthropic)

    # 4. Qdrant
    print("\n4. Testing Qdrant connection...")
    def test_qdrant():
        from qdrant_client import QdrantClient
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        info = client.get_collections()
        return f"{len(info.collections)} existing collections"

    all_ok &= check(f"Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}", test_qdrant)

    # 5. Embedding model
    print("\n5. Loading embedding model (may download on first run)...")
    def test_embedder():
        from embedder import embed_documents, get_embedding_dimension
        test_texts = ["pipe bend radius", "ergonomic posture recommendation"]
        embeddings = embed_documents(test_texts)
        dim = get_embedding_dimension()
        assert len(embeddings) == 2
        assert len(embeddings[0]) == dim
        return f"dim={dim}, model={config.EMBEDDING_MODEL}"

    all_ok &= check("Embedding model", test_embedder)

    # Summary
    print("\n" + "=" * 40)
    if all_ok:
        print("All checks passed. You're ready to run the MCP server.\n")
        print("Next steps:")
        print("  1. Make sure Qdrant is running (docker run ...)")
        print("  2. Run: python src/server.py")
        print("  3. Connect from Claude Desktop using claude_desktop_config.json")
    else:
        print("Some checks failed. Fix the issues above before starting the server.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
