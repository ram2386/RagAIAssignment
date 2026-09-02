"""Retriever helpers kept separate from generation."""

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def create_retriever(vector_store: FAISS, top_k: int) -> VectorStoreRetriever:
    """Create a FAISS-backed retriever."""
    return vector_store.as_retriever(search_kwargs={"k": top_k})

