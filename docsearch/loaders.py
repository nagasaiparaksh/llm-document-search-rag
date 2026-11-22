"""Document loading: PDF (page-aware), plain text and markdown."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: str | Path) -> list[Document]:
    """Return one Document per PDF page (or one for text files), tagged with source and page."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext!r}. Supported: {sorted(SUPPORTED)}")

    docs: list[Document] = []
    if ext == ".pdf":
        reader = PdfReader(str(path))
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": path.name, "page": number}))
        if not docs:
            raise ValueError(f"{path.name} has no extractable text (scanned PDF? OCR is not included).")
    else:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": 1}))
    return docs
