"""Vector store backends behind one interface: FAISS (default) and ChromaDB.

Embeddings are L2-normalised, so every backend reports relevance as cosine similarity in [0, 1].
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorIndex:
    def __init__(self, backend: str, embeddings: Embeddings, index_dir: str | Path):
        self.backend = backend
        self.embeddings = embeddings
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if backend == "faiss":
            self._faiss = self._load_faiss()
        elif backend == "chroma":
            from langchain_chroma import Chroma

            self._chroma = Chroma(
                collection_name="documents",
                embedding_function=embeddings,
                persist_directory=str(self.dir / "chroma"),
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            raise ValueError(f"Unknown vector store {backend!r} (use 'faiss' or 'chroma')")

    # ---- FAISS helpers ----------------------------------------------------------
    @property
    def _faiss_dir(self) -> Path:
        return self.dir / "faiss"

    def _load_faiss(self):
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        if (self._faiss_dir / "index.faiss").exists():
            # The pickle is written by this app, so loading it is safe. Never load untrusted indexes.
            return FAISS.load_local(
                str(self._faiss_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        return None

    # ---- public API ---------------------------------------------------------------
    def add(self, docs: list[Document], ids: list[str]) -> None:
        if not docs:
            return
        if self.backend == "chroma":
            self._chroma.add_documents(docs, ids=ids)
            return
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        if self._faiss is None:
            self._faiss = FAISS.from_documents(
                docs, self.embeddings, ids=ids, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
        else:
            self._faiss.add_documents(docs, ids=ids)
        self._faiss.save_local(str(self._faiss_dir))

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        if self.backend == "chroma":
            self._chroma.delete(ids=ids)
        elif self._faiss is not None:
            self._faiss.delete(ids)
            self._faiss.save_local(str(self._faiss_dir))

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return (chunk, cosine similarity) pairs, best first."""
        if self.backend == "chroma":
            if self.count() == 0:
                return []
            hits = self._chroma.similarity_search_with_score(query, k=k)
            return [(doc, 1.0 - float(dist)) for doc, dist in hits]
        if self._faiss is None or self.count() == 0:
            return []
        hits = self._faiss.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in hits]

    def count(self) -> int:
        if self.backend == "chroma":
            return int(self._chroma._collection.count())
        return int(self._faiss.index.ntotal) if self._faiss is not None else 0
