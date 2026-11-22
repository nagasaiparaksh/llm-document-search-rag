import shutil

import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLES, make_settings
from docsearch.api import create_app
from docsearch.qa import NOT_FOUND
from docsearch.services import build_services


def test_ingest_and_cited_answer(svc):
    results = svc.ingestor.ingest_folder(SAMPLES)
    assert {r.status for r in results} == {"indexed"}
    ans = svc.answerer.ask("How many days of paid sick leave do employees get?")
    assert ans.grounded and "12" in ans.answer
    assert ans.sources[0].file == "leave_policy.md"
    assert any(s.cited for s in ans.sources)


def test_picks_the_right_document(svc):
    svc.ingestor.ingest_folder(SAMPLES)
    ans = svc.answerer.ask("How long must passwords be and how often are they rotated?")
    assert ans.sources[0].file == "it_security.md" and "14" in ans.answer


def test_refuses_without_calling_llm_when_not_in_documents(svc):
    svc.ingestor.ingest_folder(SAMPLES)
    ans = svc.answerer.ask("Who won the cricket world cup in 2011?")
    assert ans.answer == NOT_FOUND and not ans.grounded
    assert svc.chat.calls == 0


def test_empty_index_is_handled(svc):
    assert svc.answerer.ask("anything at all").answer == NOT_FOUND


def test_ingestion_is_idempotent_and_detects_changes(svc, tmp_path):
    doc = tmp_path / "policy.md"
    doc.write_text("The office opens at 9am. Visitors must sign in at reception.")
    assert svc.ingestor.ingest_file(doc).status == "indexed"
    assert svc.ingestor.ingest_file(doc).status == "unchanged"
    before = svc.index.count()
    doc.write_text("The office opens at 8am. Visitors must sign in at reception.")
    assert svc.ingestor.ingest_file(doc).status == "updated"
    assert svc.index.count() == before  # stale chunks were removed, not duplicated
    assert "8am" in svc.answerer.ask("When does the office open?").answer


def test_remove_document(svc):
    svc.ingestor.ingest_folder(SAMPLES)
    assert svc.ingestor.remove("it_security.md")
    ans = svc.answerer.ask("How long must passwords be?")
    assert all(s.file != "it_security.md" for s in ans.sources)


def test_index_persists_across_restarts(tmp_path):
    settings = make_settings(tmp_path)
    build_services(settings).ingestor.ingest_folder(SAMPLES)
    reopened = build_services(settings)
    assert reopened.index.count() > 0
    assert reopened.answerer.ask("What is the parental leave for the primary caregiver?").grounded


def test_pdf_upload_to_searchable_via_api(tmp_path):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf = tmp_path / "handbook.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 750, "Expense Policy")
    c.drawString(72, 730, "Meals during travel are reimbursed up to 40 dollars per day.")
    c.showPage()
    c.drawString(72, 750, "Taxi rides over 25 dollars need written approval from a manager.")
    c.save()

    client = TestClient(create_app(build_services(make_settings(tmp_path / "app"))))
    with open(pdf, "rb") as fh:
        up = client.post("/documents", files={"file": ("handbook.pdf", fh, "application/pdf")})
    assert up.status_code == 200 and up.json()["status"] == "indexed"

    body = client.post("/ask", json={"question": "How much are meals reimbursed per day?"}).json()
    assert body["grounded"] and "40" in body["answer"]
    assert body["sources"][0]["file"] == "handbook.pdf"
    assert client.get("/documents").json()[0]["file"] == "handbook.pdf"
    assert client.get("/health").json()["chunks"] >= 1


def test_api_rejects_bad_uploads(tmp_path):
    client = TestClient(create_app(build_services(make_settings(tmp_path))))
    assert client.post("/documents", files={"file": ("x.exe", b"hi", "application/octet-stream")}).status_code == 415
    assert client.post("/documents", files={"file": ("../../evil.md", b"", "text/markdown")}).status_code == 422
    assert client.delete("/documents/nope.md").status_code == 404


def test_web_ui_is_served(tmp_path):
    client = TestClient(create_app(build_services(make_settings(tmp_path))))
    page = client.get("/")
    assert page.status_code == 200 and "Ask your documents" in page.text
    assert client.get("/health").json()["status"] == "ok"  # API routes still win over the UI mount
