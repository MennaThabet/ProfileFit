import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
import google.genai.types as genai_types

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    QueryFusionRetriever,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
ESSAY_PATH = BASE_DIR / "paul_graham_essay.txt"
KNOWLEDGE_BASE_DIR = PROJECT_DIR / "data" / "knowledge_base"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
CONTEXTS_CACHE = DATA_DIR / "contexts.json"

DEFAULT_CHUNK_SIZE = 1000  # tokens
DEFAULT_CHUNK_OVERLAP = 200  # tokens
INDEXABLE_EXTS = {".txt", ".md", ".pdf"}

# Hybrid retrieval tuning: how many candidates each leg (dense, BM25) pulls
# before fusion trims down to the caller's requested top_k. Over-fetching a
# bit here gives the fusion step more to work with.
HYBRID_FETCH_MULTIPLIER = 3

_index_cache: VectorStoreIndex | None = None
_index_cache_source: Path | None = None
_index_cache_signature: str | None = None

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


def _indexable_files(source: Path) -> list[Path]:
    """Return the files in ``source`` that the indexer can read."""
    if source.is_file():
        return [source] if source.suffix.lower() in INDEXABLE_EXTS else []
    if source.is_dir():
        return sorted(
            p
            for p in source.rglob("*")
            if p.is_file()
            and p.suffix.lower() in INDEXABLE_EXTS
            and not p.name.startswith(".")
        )
    return []


def resolve_source() -> Path:
    """Pick the corpus to index.

    Order: the ``RAG_SOURCE_DIR`` env var if set, then the project knowledge
    base (``data/knowledge_base``) once it contains documents, then the
    bundled Paul Graham essay as a demo fallback.
    """
    env_source = os.getenv("RAG_SOURCE_DIR")
    if env_source:
        return Path(env_source)
    if _indexable_files(KNOWLEDGE_BASE_DIR):
        return KNOWLEDGE_BASE_DIR
    return ESSAY_PATH


def index_dir_for(source: Path) -> Path:
    """Persist each corpus's index in its own directory to avoid cache collisions."""
    if source == ESSAY_PATH:
        return INDEX_DIR  # keep the legacy essay cache
    key = str(source.resolve()).lower()
    return DATA_DIR / f"index_{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def _source_signature(source: Path) -> str:
    """Fingerprint the corpus (paths, sizes, mtimes) to detect document changes."""
    files = _indexable_files(source)
    payload = "\n".join(
        f"{p.resolve()}::{p.stat().st_size}::{p.stat().st_mtime_ns}" for p in files
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_environment() -> None:
    """Load .env from the agent folder and the project root (whichever exists)."""
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(PROJECT_DIR / ".env")


def _get_llm_and_embedding() -> tuple[GoogleGenAI, GoogleGenAIEmbedding]:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    embed_model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    llm = GoogleGenAI(
        model=model,
        api_key=api_key,
        generation_config=genai_types.GenerateContentConfig(
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True,
                maximum_remote_calls=None,
            )
        ),
    )
    embed_model = GoogleGenAIEmbedding(model_name=embed_model_name, api_key=api_key)
    return llm, embed_model


def add_context_to_nodes(nodes: list, llm: GoogleGenAI) -> list:
    """Prefix every node with an LLM-written context paragraph."""
    contexts = load_contexts()
    total = len(nodes)
    for i, node in enumerate(nodes, start=1):
        key = chunk_hash(node.text)
        if key not in contexts:
            prompt = CONTEXT_PROMPT.format(
                title="Knowledge base document",
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


def build_index(
    args: argparse.Namespace,
    llm: GoogleGenAI,
    embed_model,
    source: Path,
    expected_signature: str | None = None,
) -> VectorStoreIndex:
    target_index_dir = index_dir_for(source)
    manifest = target_index_dir / "_source_signature.txt"
    stale = (
        expected_signature is not None
        and manifest.exists()
        and manifest.read_text(encoding="utf-8").strip() != expected_signature
    )
    if (args.build or stale) and target_index_dir.exists():
        shutil.rmtree(target_index_dir)  # project-local cache; rebuilt below

    if target_index_dir.exists():
        print("Loading cached index...")
        storage = StorageContext.from_defaults(persist_dir=target_index_dir)
        return load_index_from_storage(storage)

    files = _indexable_files(source)

    print(f"Reading {len(files)} document(s) from {source}...")
    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()

    splitter = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"Split the documents into {len(nodes)} chunks.")

    if not args.no_context:
        nodes = add_context_to_nodes(nodes, llm)
    else:
        print("Skipping LLM context generation (--no-context).")

    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
    index.storage_context.persist(persist_dir=target_index_dir)
    manifest.write_text(expected_signature or "", encoding="utf-8")
    print(f"Index saved to {target_index_dir}.")
    return index


def get_index() -> VectorStoreIndex:
    """Load (or lazily build) the RAG index for the configured source.

    The cache is invalidated whenever the corpus changes (new, removed, or
    modified documents), so freshly saved profiles are immediately queryable.
    """
    global _index_cache, _index_cache_source, _index_cache_signature

    _load_environment()
    source = resolve_source()
    signature = _source_signature(source)

    if (_index_cache is not None and _index_cache_source == source and _index_cache_signature == signature):
        return _index_cache

    llm, embed_model = _get_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model

    args = argparse.Namespace(
        build=False,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        no_context=True,  # fast tool-side builds; skip LLM chunk context
    )
    _index_cache = build_index(args, llm, embed_model, source, expected_signature=signature)
    _index_cache_source = source
    _index_cache_signature = signature
    return _index_cache


def _build_hybrid_retriever(index: VectorStoreIndex, top_k: int) -> QueryFusionRetriever:
    """
    Combine dense (vector) and sparse (BM25) retrieval via reciprocal rank fusion.

    Dense retrieval catches semantic/paraphrased matches (e.g. "built ML
    models" matching a query about "deep learning experience"). BM25 catches
    exact keyword matches (e.g. "PyTorch", "Kubernetes") that dense embeddings
    can under-rank when the surrounding phrasing differs. Fusing both and
    re-ranking the combined pool gives better recall than either alone.
    """
    fetch_k = max(top_k * HYBRID_FETCH_MULTIPLIER, top_k)

    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=fetch_k)

    # BM25 needs the raw nodes, pulled from the index's docstore.
    all_nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes,
        similarity_top_k=fetch_k,
    )

    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,  # don't generate query variations; keep it deterministic
        mode="reciprocal_rerank",
        use_async=False,
    )


def rag_search(
    query: str,
    top_k: int = 10,
    source_filter: str | None = None,
) -> str:
    """
    Hybrid (dense + BM25) search over the knowledge base.

    Args:
        query: natural-language search query.
        top_k: number of results to return after fusion.
        source_filter: optional substring matched against each candidate's ``metadata.file_path``
    """
    print(
        f"RAG search for query: {query} (top_k={top_k}, "
        f"source_filter={source_filter!r})"
    )

    try:
        top_k = max(1, min(int(top_k), 10))
        index = get_index()
        retriever = _build_hybrid_retriever(
            index, top_k=(top_k * HYBRID_FETCH_MULTIPLIER if source_filter else top_k)
        )
        nodes = retriever.retrieve(query)
    except ValueError as e:
        return json.dumps({"sources": [], "message": str(e)}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 - the caller model should never crash
        return json.dumps(
            {"sources": [], "message": f"RAG search failed: {e}"},
            ensure_ascii=False,
        )

    if source_filter:
        needle = source_filter.lower()
        nodes = [
            n
            for n in nodes
            if needle in str(n.node.metadata.get("file_path", "")).lower()
        ]
        nodes = nodes[:top_k]

    sources = []
    for rank, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        score = getattr(node_with_score, "score", None)
        sources.append(
            {
                "rank": rank,
                "score": round(float(score), 4) if score is not None else None,
                "text": clean_output(node.text),
                "metadata": {
                    k: v for k, v in node.metadata.items() if k != "context"
                },
            }
        )

    if not sources:
        message = "No relevant chunks found for that query."
        if source_filter:
            message += f" (scoped to source_filter={source_filter!r})"
        return json.dumps({"sources": [], "message": message}, ensure_ascii=False)
    return json.dumps({"sources": sources}, ensure_ascii=False, default=str)


def rag_search_posting(posting_identifier: str, query: str, top_k: int = 10) -> str:
    """
    Convenience wrapper for scoping a search to exactly one job posting.

    ``posting_identifier`` should match (a substring of) the posting's saved
    filename under ``data/knowledge_base/job_postings/`` — e.g. the slug
    returned by ``requirement_extractor.save_posting`` or ``.md`` filename.
    Use this instead of ``rag_search`` whenever a step should reason about a
    single job posting (tailoring, coverage-critic checks, etc.) rather than
    the entire indexed set of postings.
    """
    return rag_search(query, top_k=top_k, source_filter=posting_identifier)


def answer_question(index: VectorStoreIndex, question: str, top_k: int) -> None:
    retriever = _build_hybrid_retriever(index, top_k=top_k)
    synthesizer = get_response_synthesizer(response_mode="compact")
    nodes = retriever.retrieve(question)
    response = synthesizer.synthesize(question, nodes)

    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print(response)

    print("\nSources:")
    for i, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        score = getattr(node_with_score, "score", None)
        snippet = " ".join(node.text.split())
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        score_str = f"{score:.3f}" if score is not None else "n/a"
        print(f"  [{i}] score={score_str} | {snippet}")


def main() -> None:
    _load_environment()

    parser = argparse.ArgumentParser(
        description="Contextual hybrid RAG (dense + BM25) over the knowledge base using Google Gemini."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask about the indexed documents",
    )
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

    try:
        llm, embed_model = _get_llm_and_embedding()
    except ValueError as e:
        print(
            e
        )
        sys.exit(1)

    Settings.llm = llm
    Settings.embed_model = embed_model

    source = resolve_source()
    print(f"RAG source: {source}")
    index = build_index(
        args,
        llm,
        embed_model,
        source,
        expected_signature=_source_signature(source),
    )

    if args.question:
        answer_question(index, args.question, args.top_k)
    else:
        print("\nIndex ready. Ask a question, e.g.:")
        print('  uv run agent/rag.py "What did Paul Graham work on before college?"')


if __name__ == "__main__":
    main()