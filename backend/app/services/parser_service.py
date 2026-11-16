"""
ResumeIQ Document Parser Service.
Extracts, cleans, and structures textual content from PDF and DOCX resume files.
"""

import io
import os
from typing import Dict, Optional
from pydantic import BaseModel, Field
import pypdf
import docx

from backend.app.core.text_cleaner import clean_text, segment_sections


# --- Domain Exceptions ---
class DocumentParsingError(Exception):
    """Base exception for document parsing failures."""
    pass


class UnsupportedFileFormatError(DocumentParsingError):
    """Raised when an unsupported file extension or MIME type is uploaded."""
    pass


class EncryptedDocumentError(DocumentParsingError):
    """Raised when a password-protected PDF is encountered."""
    pass


class EmptyDocumentError(DocumentParsingError):
    """Raised when no extractable text is found in the document."""
    pass


# --- Structured Output Schema ---
class ParsedDocument(BaseModel):
    """Structured result returned by the document parser."""
    filename: str
    file_type: str = Field(description="Normalized file format: 'pdf' or 'docx'")
    raw_text: str
    cleaned_text: str
    sections: Dict[str, str] = Field(default_factory=dict)
    word_count: int
    char_count: int
    page_count: int = 1


def _extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract text content and page count from a PDF byte-stream.
    """
    try:
        stream = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(stream)

        if reader.is_encrypted:
            try:
                # Try decrypting with empty password
                reader.decrypt("")
            except Exception:
                raise EncryptedDocumentError("The uploaded PDF is password-protected and cannot be read.")

        page_count = len(reader.pages)
        if page_count == 0:
            raise EmptyDocumentError("The uploaded PDF has 0 pages.")

        extracted_pages = []
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            extracted_pages.append(text)

        raw_text = "\n\n".join(extracted_pages).strip()
        return raw_text, page_count

    except EncryptedDocumentError:
        raise
    except EmptyDocumentError:
        raise
    except Exception as exc:
        raise DocumentParsingError(f"Failed to parse PDF document: {str(exc)}") from exc


def _extract_text_from_docx(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract text content from a DOCX byte-stream in exact document order,
    traversing both paragraphs and tables sequentially.
    """
    try:
        stream = io.BytesIO(file_bytes)
        doc = docx.Document(stream)

        text_parts = []

        # Traverse document body elements in visual document order
        for child in doc.element.body:
            # Paragraph element (w:p)
            if child.tag.endswith("p"):
                p = docx.text.paragraph.Paragraph(child, doc)
                p_text = p.text.strip()
                if p_text:
                    text_parts.append(p_text)

            # Table element (w:tbl)
            elif child.tag.endswith("tbl"):
                table = docx.table.Table(child, doc)
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        # De-duplicate identical adjacent merged cells
                        unique_cells = []
                        for c in row_cells:
                            if not unique_cells or c != unique_cells[-1]:
                                unique_cells.append(c)
                        text_parts.append(" | ".join(unique_cells))

        raw_text = "\n".join(text_parts).strip()
        # Word documents do not have explicit page numbers in XML without rendering;
        # estimate ~350 words per page as an approximate heuristic
        word_count = len(raw_text.split()) if raw_text else 0
        est_pages = max(1, (word_count + 349) // 350)

        return raw_text, est_pages

    except Exception as exc:
        raise DocumentParsingError(f"Failed to parse DOCX document: {str(exc)}") from exc


def parse_document(file_content: bytes, filename: str) -> ParsedDocument:
    """
    Unified entry point to parse, sanitize, and segment any supported resume file.

    Parameters:
    - file_content: Raw bytes of the uploaded file
    - filename: Original filename with extension (.pdf or .docx)

    Returns:
    - ParsedDocument model with raw text, sanitized text, structured sections, and metrics.
    """
    if not file_content:
        raise EmptyDocumentError(f"The uploaded file '{filename}' is empty (0 bytes).")

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        raw_text, page_count = _extract_text_from_pdf(file_content)
        file_type = "pdf"
    elif ext in [".docx", ".doc"]:
        if ext == ".doc":
            raise UnsupportedFileFormatError(
                "Legacy .doc format is not supported. Please save or export your resume as .docx or .pdf."
            )
        raw_text, page_count = _extract_text_from_docx(file_content)
        file_type = "docx"
    else:
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{ext}'. Only .pdf and .docx files are supported."
        )

    if not raw_text or not raw_text.strip():
        raise EmptyDocumentError(
            f"No readable text could be extracted from '{filename}'. "
            "Please ensure the document contains selectable text and is not an image-only scan."
        )

    # Sanitize and normalize text
    cleaned_text = clean_text(raw_text)

    # Segment into structured sections
    sections = segment_sections(cleaned_text)

    # Compute descriptive metrics
    words = cleaned_text.split()
    word_count = len(words)
    char_count = len(cleaned_text)

    return ParsedDocument(
        filename=filename,
        file_type=file_type,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        sections=sections,
        word_count=word_count,
        char_count=char_count,
        page_count=page_count,
    )