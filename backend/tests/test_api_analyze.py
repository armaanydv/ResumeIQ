"""
Tests for Phase 5: FastAPI /api/analyze Integration Endpoint.
"""

import io
import docx
from fastapi.testclient import TestClient
from backend.app.main import app


client = TestClient(app)

SAMPLE_JD = """
Senior Full-Stack Software Engineer
Company: CloudTech Solutions
We are seeking an experienced Full-Stack Engineer skilled in Python, FastAPI, Docker, and PostgreSQL.
Responsibilities:
- Build resilient microservices and RESTful APIs using Python and FastAPI.
- Manage Docker containerization and Kubernetes cluster deployment.
- Optimize relational database performance with PostgreSQL and Redis caching.
Requirements:
- Strong proficiency in Python and FastAPI.
- Experience with Docker, Kubernetes, and PostgreSQL.
- Degree in Computer Science or related technical discipline.
"""


def _create_sample_docx() -> bytes:
    """Helper to generate a valid in-memory DOCX resume."""
    doc = docx.Document()
    doc.add_heading("Jane Doe", level=0)
    doc.add_paragraph("jane.doe@example.com | San Francisco, CA")

    doc.add_heading("Summary", level=1)
    doc.add_paragraph("Experienced Software Engineer with 4 years building scalable Python web applications.")

    doc.add_heading("Technical Skills", level=1)
    doc.add_paragraph("Python, FastAPI, Docker, PostgreSQL, Redis, Kubernetes, Git")

    doc.add_heading("Experience", level=1)
    doc.add_paragraph("Backend Developer at Acme Corp (2021-Present)")
    doc.add_paragraph("- Built scalable backend APIs with FastAPI and Python.")
    doc.add_paragraph("- Deployed containerized applications using Docker and Kubernetes.")

    doc.add_heading("Projects", level=1)
    doc.add_paragraph("ResumeIQ - AI Resume Matching System")
    doc.add_paragraph("- Implemented RAG pipeline with FAISS vector store.")

    doc.add_heading("Education", level=1)
    doc.add_paragraph("B.S. in Computer Science - University of California")

    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


def _create_sample_pdf() -> bytes:
    """Helper to generate a valid in-memory text PDF."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 75 >> stream\n"
        b"BT /F1 12 Tf 72 712 Td (Jane Doe - Python FastAPI Docker PostgreSQL Developer) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000370 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n444\n%%EOF"
    )


def test_analyze_endpoint_docx_success():
    """Verify successful end-to-end resume analysis with DOCX upload."""
    docx_bytes = _create_sample_docx()

    response = client.post(
        "/api/analyze",
        files={"file": ("jane_resume.docx", io.BytesIO(docx_bytes), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"job_description": SAMPLE_JD},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "jane_resume.docx"
    assert "score_breakdown" in data
    assert 0.0 <= data["score_breakdown"]["overall_ats_score"] <= 100.0
    assert data["score_breakdown"]["verdict"] != ""
    assert len(data["matched_skills"]) > 0
    assert len(data["bullet_improvements"]) > 0
    assert len(data["actionable_recommendations"]) > 0
    assert data["id"] is not None


def test_analyze_endpoint_pdf_success():
    """Verify successful end-to-end resume analysis with PDF upload."""
    pdf_bytes = _create_sample_pdf()

    response = client.post(
        "/api/analyze",
        files={"file": ("jane_resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"job_description": SAMPLE_JD},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "jane_resume.pdf"
    assert data["score_breakdown"]["overall_ats_score"] >= 0.0


def test_analyze_unsupported_file_extension():
    """Verify that unsupported extensions return HTTP 415."""
    response = client.post(
        "/api/analyze",
        files={"file": ("resume.txt", io.BytesIO(b"Simple text file"), "text/plain")},
        data={"job_description": SAMPLE_JD},
    )
    assert response.status_code == 415
    assert "Unsupported file format" in response.json()["detail"]


def test_analyze_empty_file():
    """Verify that 0-byte file returns HTTP 400."""
    response = client.post(
        "/api/analyze",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        data={"job_description": SAMPLE_JD},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_short_job_description():
    """Verify that job descriptions shorter than 20 characters return HTTP 422."""
    docx_bytes = _create_sample_docx()
    response = client.post(
        "/api/analyze",
        files={"file": ("jane_resume.docx", io.BytesIO(docx_bytes), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"job_description": "Too short"},
    )
    assert response.status_code == 422


if __name__ == "__main__":
    print("Running Phase 5 integration tests...")
    test_analyze_endpoint_docx_success()
    print("✓ test_analyze_endpoint_docx_success passed")
    test_analyze_endpoint_pdf_success()
    print("✓ test_analyze_endpoint_pdf_success passed")
    test_analyze_unsupported_file_extension()
    print("✓ test_analyze_unsupported_file_extension passed")
    test_analyze_empty_file()
    print("✓ test_analyze_empty_file passed")
    test_analyze_short_job_description()
    print("✓ test_analyze_short_job_description passed")
    print("All Phase 5 tests passed successfully!")

