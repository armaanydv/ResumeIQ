"""
Tests for Phase 2: Text Sanitization, Section Segmentation, and Document Parsing.
"""

import io
import pytest
import pypdf
import docx

from backend.app.core.text_cleaner import clean_text, segment_sections
from backend.app.services.parser_service import (
    parse_document,
    DocumentParsingError,
    UnsupportedFileFormatError,
    EmptyDocumentError,
)


# --- Sample Resume Text Fixtures ---
SAMPLE_RAW_RESUME = """
Jane Doe
jane.doe@example.com | (555) 123-4567 | San Francisco, CA

PROFESSIONAL SUMMARY
Experienced Full-Stack Software Engineer with 5+ years of experience designing scalable microservices.

TECHNICAL SKILLS
• Languages: Python, JavaScript, TypeScript, Go
• Frameworks: FastAPI, React, Node.js, Express
• Tools: Docker, Kubernetes, PostgreSQL, Redis, Git

WORK EXPERIENCE
Senior Backend Engineer – Acme Corp (2021 – Present)
• Architected event-driven microservices processing 10M+ daily events.
• Reduced API p99 latency by 35% through Redis caching and query optimization.

Software Engineer – StartupX (2019 – 2021)
• Built RESTful APIs using Python and FastAPI for a fintech platform.
• Automated CI/CD pipelines using GitHub Actions.

PROJECTS
ResumeIQ – AI Resume Analyzer (2024)
• Developed an intelligent resume scoring engine using RAG and vector embeddings.
• Built deterministic ATS compatibility scoring logic in Python.

EDUCATION
B.S. in Computer Science – University of California, Berkeley (2015 – 2019)
"""


def test_clean_text_bullets_and_unicode():
    """Verify that fancy bullets, smart quotes, and dashes are normalized."""
    dirty_text = "• Point one\n\u25aa Point two \u201cquoted\u201d\n\u2013 Dash item\u200b"
    cleaned = clean_text(dirty_text)
    assert "- Point one" in cleaned
    assert "- Point two \"quoted\"" in cleaned
    assert "- Dash item" in cleaned
    assert "\u200b" not in cleaned  # Zero-width space stripped


def test_clean_text_spacing_and_newlines():
    """Verify excessive newlines and spaces are compressed."""
    messy = "Line one    with   spaces\n\n\n\n\nLine two"
    cleaned = clean_text(messy)
    assert "Line one with spaces" in cleaned
    assert "\n\nLine two" in cleaned
    assert "\n\n\n" not in cleaned


def test_segment_sections():
    """Verify resume text is split into correct section buckets."""
    cleaned = clean_text(SAMPLE_RAW_RESUME)
    sections = segment_sections(cleaned)

    assert "general" in sections
    assert "Jane Doe" in sections["general"]

    assert "summary" in sections
    assert "Experienced Full-Stack Software Engineer" in sections["summary"]

    assert "skills" in sections
    assert "FastAPI" in sections["skills"]

    assert "experience" in sections
    assert "Acme Corp" in sections["experience"]

    assert "projects" in sections
    assert "ResumeIQ" in sections["projects"]

    assert "education" in sections
    assert "University of California" in sections["education"]


def test_parse_docx_in_memory():
    """Create a sample DOCX document in-memory and parse it."""
    doc = docx.Document()
    doc.add_heading("Alex Mercer", level=0)
    doc.add_paragraph("alex@mercer.dev | Seattle, WA")

    doc.add_heading("Summary", level=1)
    doc.add_paragraph("AI Engineer specializing in RAG architectures.")

    doc.add_heading("Technical Skills", level=1)
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "AI/ML: PyTorch, FAISS"
    table.rows[0].cells[1].text = "Cloud: AWS, Docker"

    doc.add_heading("Experience", level=1)
    doc.add_paragraph("ML Engineer at TechCorp (2022-Present)")
    p = doc.add_paragraph("Deployed LLM pipelines with FAISS retrieval.", style="List Bullet")

    # Save to bytes stream
    stream = io.BytesIO()
    doc.save(stream)
    docx_bytes = stream.getvalue()

    # Parse through our service
    parsed = parse_document(docx_bytes, "alex_resume.docx")

    assert parsed.filename == "alex_resume.docx"
    assert parsed.file_type == "docx"
    assert parsed.word_count > 10
    assert "Alex Mercer" in parsed.cleaned_text
    assert "FAISS" in parsed.cleaned_text
    assert "skills" in parsed.sections
    assert "experience" in parsed.sections


def test_parse_pdf_with_text():
    """Verify parsing a valid PDF containing textual data."""
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 55 >> stream\n"
        b"BT /F1 12 Tf 72 712 Td (John Developer - Python Engineer) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000350 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n424\n%%EOF"
    )
    parsed = parse_document(pdf_bytes, "john_resume.pdf")
    assert parsed.filename == "john_resume.pdf"
    assert parsed.file_type == "pdf"
    assert "John Developer - Python Engineer" in parsed.cleaned_text
    assert parsed.page_count == 1
    assert parsed.word_count >= 4


def test_parse_pdf_in_memory():
    """Create an empty PDF in-memory and verify EmptyDocumentError is raised."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    
    stream = io.BytesIO()
    writer.write(stream)
    pdf_bytes = stream.getvalue()

    with pytest.raises(EmptyDocumentError):
        parse_document(pdf_bytes, "empty_scan.pdf")


def test_unsupported_extension_error():
    """Verify that unsupported file formats (.txt, .png, etc.) raise an error."""
    with pytest.raises(UnsupportedFileFormatError):
        parse_document(b"Sample text content", "resume.txt")


def test_empty_bytes_error():
    """Verify that passing 0-byte content raises EmptyDocumentError."""
    with pytest.raises(EmptyDocumentError):
        parse_document(b"", "blank.pdf")


if __name__ == "__main__":
    print("Running Phase 2 tests...")
    test_clean_text_bullets_and_unicode()
    print("✓ test_clean_text_bullets_and_unicode passed")
    test_clean_text_spacing_and_newlines()
    print("✓ test_clean_text_spacing_and_newlines passed")
    test_segment_sections()
    print("✓ test_segment_sections passed")
    test_parse_docx_in_memory()
    print("✓ test_parse_docx_in_memory passed")
    test_parse_pdf_in_memory()
    print("✓ test_parse_pdf_in_memory passed")
    test_unsupported_extension_error()
    print("✓ test_unsupported_extension_error passed")
    test_empty_bytes_error()
    print("✓ test_empty_bytes_error passed")
    print("All Phase 2 tests passed successfully!")
