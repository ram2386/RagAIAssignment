"""FAISS vector store creation, loading, and search."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot be built or loaded."""


def build_faiss_index(
    chunks: list[Document],
    embeddings: OllamaEmbeddings,
    index_path: Path,
) -> FAISS:
    """Build and save a FAISS index from document chunks."""
    if not chunks:
        raise VectorStoreError("Cannot build a FAISS index from zero chunks.")

    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        index_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_path))
    except Exception as exc:
        raise VectorStoreError(
            "Could not build FAISS index. Confirm Ollama is running and "
            "the embedding model is installed with: ollama pull nomic-embed-text"
        ) from exc

    return vector_store


def load_faiss_index(index_path: Path, embeddings: OllamaEmbeddings) -> FAISS:
    """Load an existing FAISS index from disk."""
    if not index_path.exists():
        raise VectorStoreError(f"FAISS index not found at {index_path}.")

    try:
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        raise VectorStoreError(
            "Could not load FAISS index. It may be missing or corrupted. "
            "Rebuild it with: python main.py --rebuild-index"
        ) from exc


def get_or_create_faiss_index(
    chunks: list[Document],
    embeddings: OllamaEmbeddings,
    index_path: Path,
    rebuild: bool = False,
) -> FAISS:
    """Load an existing index unless rebuilding was requested."""
    index_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"

    if not rebuild and index_file.exists() and pkl_file.exists():
        return load_faiss_index(index_path, embeddings)

    return build_faiss_index(chunks, embeddings, index_path)


def similarity_search_with_scores(
    vector_store: FAISS,
    question: str,
    top_k: int,
) -> list[tuple[Document, float]]:
    """Run FAISS similarity search and return documents with raw distances."""
    try:
        return vector_store.similarity_search_with_score(question, k=top_k)
    except Exception as exc:
        raise VectorStoreError(f"Similarity search failed: {exc}") from exc

