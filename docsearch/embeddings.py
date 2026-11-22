"""Embedding models: Ollama in production, deterministic hashing for offline use and tests."""
from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from .config import Settings

_STOP = set(
    "a an and are as at be by for from has have how in is it its of on or that the this to was "
    "were what when where which who will with you your we our do does can i".split()
)


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashEmbeddings(Embeddings):
    """Feature-hashing bag-of-words vectors. Lexical, not semantic, but deterministic and offline."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]
        for tok in tokens:
            idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % self.dim
            vec[idx] += 1.0
        return _l2([math.log1p(v) for v in vec])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class NormalizedEmbeddings(Embeddings):
    """L2-normalises any embedding model so inner product == cosine similarity in every store."""

    def __init__(self, inner: Embeddings):
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_l2(v) for v in self.inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return _l2(self.inner.embed_query(text))


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.backend == "offline":
        return HashEmbeddings()
    if settings.backend == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return NormalizedEmbeddings(
            OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_host)
        )
    raise ValueError(f"Unknown RAG_BACKEND {settings.backend!r} (use 'ollama' or 'offline')")
