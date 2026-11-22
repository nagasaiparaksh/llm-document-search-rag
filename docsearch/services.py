"""Wires settings -> embeddings -> vector index -> ingestor -> answerer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import Settings, get_settings
from .embeddings import build_embeddings
from .ingest import Ingestor
from .qa import Answerer, build_chat
from .stores import VectorIndex


@dataclass
class Services:
    settings: Settings
    index: VectorIndex
    ingestor: Ingestor
    answerer: Answerer


def build_services(settings: Optional[Settings] = None, chat=None) -> Services:
    settings = settings or get_settings()
    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    index = VectorIndex(settings.store, build_embeddings(settings), settings.index_dir)
    ingestor = Ingestor(index, settings)
    answerer = Answerer(index, chat or build_chat(settings), settings)
    return Services(settings, index, ingestor, answerer)
