"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    pdf_path: Path = BASE_DIR / os.getenv("PDF_PATH", "data/HRPolicy.pdf")
    index_path: Path = BASE_DIR / os.getenv("FAISS_INDEX_PATH", "vector_store/faiss_index")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    local_llm: str = os.getenv("LOCAL_LLM", "qwen3:4b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "3"))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "300"))


config = AppConfig()
