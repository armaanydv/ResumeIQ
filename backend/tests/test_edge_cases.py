"""
ResumeIQ - Edge Cases & Resilience Tests.
Validates boundary conditions: corrupted files, empty files, unsupported extensions,
input validations, and 404 error responses.
"""

import io
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

SAMPLE_JD = """
Senior Full-Stack Engineer
Requirements:
- 5+ years of experience in Python and FastAPI
- Strong understanding of PostgreSQL, Docker, and Redis
- Experience with React or modern JavaScript
- Solid communication and system architecture skills
"""


def test_empty_file_upload_rejected():
    """Verify that uploading a 0-byte file returns HTTP 400 Bad Request."""
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/api/analyze",
        files={"file": ("empty.pdf", empty_file, "application/pdf")},
        data={"job_description": SAMPLE_JD},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_unsupported_file_extension_rejected():
    """Verify that uploading a .txt file returns HTTP 415 Unsupported Media Type."""
    txt_file = io.BytesIO(b"Hello world, I am a plain text resume.")
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.txt", txt_file, "text/plain")},
        data={"job_description": SAMPLE_JD},
    )
    assert response.status_code == 415
    assert "unsupported" in response.json()["detail"].lower()


def test_corrupted_pdf_file_rejected():
    """Verify that uploading corrupted/invalid binary bytes as PDF returns HTTP 422."""
    corrupted_bytes = io.BytesIO(b"%PDF-1.4 garbage invalid corrupted binary stream content 12345")
    response = client.post(
        "/api/analyze",
        files={"file": ("corrupted.pdf", corrupted_bytes, "application/pdf")},
        data={"job_description": SAMPLE_JD},
    )
    assert response.status_code == 422
    assert "error" in response.json()["detail"].lower()


def test_short_job_description_rejected():
    """Verify that submitting a job description under 20 chars returns validation error HTTP 422."""
    dummy_pdf = io.BytesIO(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.pdf", dummy_pdf, "application/pdf")},
        data={"job_description": "Too short"},
    )
    assert response.status_code == 422


def test_browser_html_negotiation():
    """Verify that requesting '/' with 'Accept: text/html' returns the web application HTML."""
    response = client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "ResumeIQ" in response.text

