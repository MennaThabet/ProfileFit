from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

BASE_DIR = Path(__file__).resolve().parent
ESSAY_PATH = BASE_DIR / "paul_graham_essay.txt"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
CONTEXTS_CACHE = DATA_DIR / "contexts.json"

DEFAULT_CHUNK_SIZE = 1500  # tokens
DEFAULT_CHUNK_OVERLAP = 150  # tokens

CONTEXT_PROMPT = """You are helping make chunks of a document easier to find in a search index.

Document title: {title}

Here is the chunk of the document that needs context:

<chunk>
{chunk_text}
</chunk>

Here is the text that immediately precedes the chunk:

<previous>
{previous_text}
</previous>

Here is the text that immediately follows the chunk:

<next>
{next_text}
</next>

Write a short, self-contained context (2-3 sentences) that says what this chunk
is about and how it fits into the document, as if you were writing an index
entry for it. Do not restate the chunk. Output only the context and nothing
else.
"""


def clean_output(text: str) -> str:
    """Gemini 3.6 sometimes returns U+FFFD where an em dash belongs."""
    return text.replace("\ufffd", "\u2014")


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_contexts() -> dict[str, str]:
    if CONTEXTS_CACHE.exists():
        return json.loads(CONTEXTS_CACHE.read_text(encoding="utf-8"))
    return {}


def save_contexts(contexts: dict[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXTS_CACHE.write_text(
        json.dumps(contexts, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_context_to_nodes(nodes: list, llm: GoogleGenAI) -> list:
    """Prefix every node with an LLM-written context paragraph."""
    contexts = load_contexts()
    total = len(nodes)
    for i, node in enumerate(nodes, start=1):
        key = chunk_hash(node.text)
        if key not in contexts:
            prompt = CONTEXT_PROMPT.format(
                title="What I Worked On (essay by Paul Graham)",
                chunk_text=node.text,
                previous_text=nodes[i - 2].text if i > 1 else "(start of document)",
                next_text=nodes[i].text if i < total else "(end of document)",
            )
            print(f"[{i}/{total}] Writing context for chunk...")
            response = llm.complete(prompt)
            contexts[key] = response.text.strip()
            save_contexts(contexts)

        context = contexts[key]
        node.metadata["context"] = context
        node.text = f"{context}\n\n{node.text}"
    return nodes


def build_index(args: argparse.Namespace, llm: GoogleGenAI, embed_model) -> VectorStoreIndex:
    if args.build and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)  # project-local cache; rebuilt from scratch below

    if INDEX_DIR.exists():
        print("Loading cached index...")
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)

    print(f"Reading {ESSAY_PATH.name}...")
    reader = SimpleDirectoryReader(input_files=[ESSAY_PATH])
    docs = reader.load_data()

    splitter = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"Split the essay into {len(nodes)} chunks.")

    if not args.no_context:
        nodes = add_context_to_nodes(nodes, llm)
    else:
        print("Skipping LLM context generation (--no-context).")

    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}.")
    return index


def answer_question(index: VectorStoreIndex, question: str, top_k: int) -> None:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    synthesizer = get_response_synthesizer(response_mode="compact")
    nodes = retriever.retrieve(question)
    response = synthesizer.synthesize(question, nodes)

    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print(clean_output(str(response)))

    print("\nSources:")
    for i, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        score = getattr(node_with_score, "score", None)
        snippet = " ".join(node.text.split())
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        score_str = f"{score:.3f}" if score is not None else "n/a"
        print(f"  [{i}] score={score_str} | {clean_output(snippet)}")


def main() -> None:
    load_dotenv(BASE_DIR / ".env")

    parser = argparse.ArgumentParser(
        description="Contextual RAG over Paul Graham's essay using Google Gemini."
    )
    parser.add_argument("question", nargs="?", help="Question to ask about the essay")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild the index from scratch",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Skip LLM-generated context per chunk (faster, cheaper)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in tokens (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in tokens (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "No API key found. Add GOOGLE_API_KEY (or GEMINI_API_KEY) to .env "
            "and run again."
        )
        sys.exit(1)

    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    embed_model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    llm = GoogleGenAI(model=model, api_key=api_key)
    embed_model = GoogleGenAIEmbedding(model_name=embed_model_name, api_key=api_key)

    Settings.llm = llm
    Settings.embed_model = embed_model

    index = build_index(args, llm, embed_model)

    if args.question:
        answer_question(index, args.question, args.top_k)
    else:
        print("\nIndex ready. Ask a question, e.g.:")
        print('  uv run rag.py "What did Paul Graham work on before college?"')


if __name__ == "__main__":
    main()