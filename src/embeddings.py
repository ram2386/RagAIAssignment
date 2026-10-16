"""Embedding model factory."""

import os

from langchain_core.embeddings import Embeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_ollama import OllamaEmbeddings


def create_embeddings(
    provider: str,
    model: str,
    base_url: str | None = None,
) -> Embeddings:
    """Create the selected embedding model."""
    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required when using Mistral embeddings.")
        return MistralAIEmbeddings(model=model, api_key=api_key)
    if provider == "ollama":
        return OllamaEmbeddings(model=model, base_url=base_url)
    raise ValueError(f"Unsupported embedding provider: {provider}")

