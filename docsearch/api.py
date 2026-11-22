"""FastAPI service: upload a PDF and it is indexed immediately; ask questions with citations."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .loaders import SUPPORTED
from .services import Services, build_services

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: Optional[int] = Field(default=None, ge=1, le=20)


def create_app(services: Optional[Services] = None) -> FastAPI:
    svc = services or build_services()
    app = FastAPI(title="LLM-Powered Document Search", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": svc.settings.backend, "store": svc.settings.store,
                "chunks": svc.index.count()}

    @app.post("/documents")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or "").name  # strip any path components
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(415, f"Unsupported type. Use one of {sorted(SUPPORTED)}")
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (25 MB limit)")
        target = svc.settings.inbox_dir / name
        target.write_bytes(data)
        result = svc.ingestor.ingest_file(target)
        if result.status == "failed":
            target.unlink(missing_ok=True)
            raise HTTPException(422, result.error)
        return asdict(result)

    @app.get("/documents")
    def documents():
        return svc.ingestor.documents()

    @app.delete("/documents/{name}")
    def delete(name: str):
        name = Path(name).name
        if not svc.ingestor.remove(name):
            raise HTTPException(404, "Unknown document")
        (svc.settings.inbox_dir / name).unlink(missing_ok=True)
        return {"deleted": name}

    @app.post("/ask")
    def ask(req: AskRequest):
        return asdict(svc.answerer.ask(req.question, req.k))

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")  # web UI, registered last
    return app


def app_factory() -> FastAPI:  # for: uvicorn docsearch.api:app_factory --factory
    return create_app()
