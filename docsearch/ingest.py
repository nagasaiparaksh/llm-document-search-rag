"""Automated ingestion: drop a PDF in, it becomes searchable. No manual indexing step.

A manifest (file name -> content hash + chunk ids) makes ingestion idempotent: unchanged files are
skipped, edited files are re-indexed (old chunks removed), and deleted files can be purged.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .loaders import SUPPORTED, load_file
from .stores import VectorIndex


@dataclass
class IngestResult:
    file: str
    status: str  # indexed | updated | unchanged | failed
    chunks: int = 0
    error: str = ""


class Ingestor:
    def __init__(self, index: VectorIndex, settings: Settings):
        self.index, self.settings = index, settings
        self.manifest_path = Path(settings.index_dir) / "manifest.json"
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self.manifest: dict[str, dict] = (
            json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        )

    def _save(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def ingest_file(self, path: str | Path) -> IngestResult:
        path = Path(path)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            previous = self.manifest.get(path.name)
            if previous and previous["sha256"] == digest:
                return IngestResult(path.name, "unchanged", len(previous["chunk_ids"]))

            chunks = self.splitter.split_documents(load_file(path))
            if not chunks:
                raise ValueError(f"{path.name} contains no text to index.")
            ids = [f"{digest[:12]}:{i}" for i in range(len(chunks))]
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk"] = i

            if previous:  # file changed: drop stale chunks first
                self.index.delete(previous["chunk_ids"])
            self.index.add(chunks, ids)
            self.manifest[path.name] = {"sha256": digest, "chunk_ids": ids}
            self._save()
            return IngestResult(path.name, "updated" if previous else "indexed", len(ids))
        except Exception as exc:  # one bad file must not stop a batch
            return IngestResult(path.name, "failed", error=str(exc))

    def ingest_folder(self, folder: str | Path) -> list[IngestResult]:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED)
        return [self.ingest_file(p) for p in files]

    def remove(self, name: str) -> bool:
        entry = self.manifest.pop(name, None)
        if not entry:
            return False
        self.index.delete(entry["chunk_ids"])
        self._save()
        return True

    def documents(self) -> list[dict]:
        return [{"file": n, "chunks": len(m["chunk_ids"])} for n, m in sorted(self.manifest.items())]

    def watch(self, folder: str | Path, interval: float = 5.0, stop: Optional[Event] = None) -> None:
        """Poll a folder and index new or changed files as they appear."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stop = stop or Event()
        while not stop.is_set():
            for result in self.ingest_folder(folder):
                if result.status != "unchanged":
                    print(f"[watch] {result.file}: {result.status} {result.chunks or ''} {result.error}")
            stop.wait(interval)
