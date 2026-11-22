from pathlib import Path

import pytest

from docsearch.config import Settings
from docsearch.services import build_services

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"


class CountingChat:
    """Wraps the offline extractive chat and counts LLM calls."""

    def __init__(self):
        from docsearch.qa import ExtractiveChat

        self.inner, self.calls = ExtractiveChat(), 0

    def generate(self, system, user):
        self.calls += 1
        return self.inner.generate(system, user)


def make_settings(tmp_path, store="faiss") -> Settings:
    return Settings(
        backend="offline", ollama_host="", chat_model="", embed_model="", store=store,
        index_dir=tmp_path / "index", inbox_dir=tmp_path / "inbox",
        chunk_size=400, chunk_overlap=60, top_k=4, min_relevance=0.12,
    )


@pytest.fixture(params=["faiss", "chroma"])
def svc(request, tmp_path):
    chat = CountingChat()
    services = build_services(make_settings(tmp_path, request.param), chat=chat)
    services.chat = chat
    return services
