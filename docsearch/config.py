"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend: str  # "ollama" (local models) or "offline" (hash embeddings + extractive answers)
    ollama_host: str
    chat_model: str
    embed_model: str
    store: str  # "faiss" or "chroma"
    index_dir: Path
    inbox_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance: float  # cosine similarity below this = "not in the documents"


def get_settings() -> Settings:
    backend = os.getenv("RAG_BACKEND", "ollama").lower()
    default_min = "0.35" if backend == "ollama" else "0.12"
    return Settings(
        backend=backend,
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b"),
        embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        store=os.getenv("VECTOR_STORE", "faiss").lower(),
        index_dir=Path(os.getenv("INDEX_DIR", "data/index")),
        inbox_dir=Path(os.getenv("INBOX_DIR", "data/inbox")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "4")),
        min_relevance=float(os.getenv("MIN_RELEVANCE", default_min)),
    )
