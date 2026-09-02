"""Local Ollama embedding model factory."""

from langchain_ollama import OllamaEmbeddings


def create_embeddings(model: str, base_url: str) -> OllamaEmbeddings:
    """Create local Ollama embeddings."""
    return OllamaEmbeddings(model=model, base_url=base_url)

