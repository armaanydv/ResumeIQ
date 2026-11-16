
"""
ResumeIQ Analysis API Routes.
Provides multipart endpoint for uploading resumes and analyzing them against job descriptions.
"""

import os
import uuid
from typing import Annotated
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.app.api.schemas import AnalysisResponse
from backend.app.services.parser_service import (
    parse_document,
    DocumentParsingError,
    EmptyDocumentError,
    EncryptedDocumentError,
    UnsupportedFileFormatError,
)
from backend.app.services.rag_service import chunk_resume, ResumeVectorStore
from backend.app.services.llm_service import analyze_resume_with_llm


router = APIRouter()

# Constraints
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze resume against a job description",
    description="Upload a PDF or DOCX resume along with target job description text to receive an ATS compatibility report.",
)
async def analyze_resume_endpoint(
    file: Annotated[UploadFile, File(description="Resume file in .pdf or .docx format")],
    job_description: Annotated[
        str,
        Form(
            min_length=20,
            max_length=20000,
            description="Target job description text (minimum 20 characters)",
        ),
    ],
) -> AnalysisResponse:
    """
    End-to-end resume analysis pipeline:
    1. Validates file extension, size, and non-empty content.
    2. Parses document into structured sections (Summary, Skills, Experience, Projects, Education).
    3. Semantically chunks the document and indexes it into an in-memory FAISS vector store.
    4. Evaluates qualifications and rewrites weak bullet points using RAG-grounded AI analysis.
    5. Computes deterministic Python weighted ATS scores.
    """
    # 1. Validate file extension
    filename = file.filename or "resume.pdf"
    _, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if ext_lower not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{ext}'. ResumeIQ supports only .pdf and .docx documents.",
        )

    # 2. Read and validate file size
    try:
        content_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        ) from exc

    if not content_bytes or len(content_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes). Please provide a valid resume.",
        )

    if len(content_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowable limit of 10 MB.",
        )

    # 3. Document Parsing & Section Segmentation
    try:
        parsed_doc = parse_document(content_bytes, filename)
    except UnsupportedFileFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except EncryptedDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DocumentParsingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error processing resume document: {str(exc)}",
        ) from exc

    # 4. RAG Pipeline: Semantic Chunking & In-Memory Vector Store
    chunks = chunk_resume(parsed_doc.sections)
    if not chunks:
        # Fallback to chunking the cleaned text directly if section headings were non-standard
        chunks = chunk_resume({"general": parsed_doc.cleaned_text})

    vector_store = ResumeVectorStore(chunks)

    # 5. AI Evaluation & Deterministic ATS Scoring
    try:
        analysis_result = analyze_resume_with_llm(
            job_description=job_description,
            parsed_doc=parsed_doc,
            vector_store=vector_store,
        )
        analysis_result.id = uuid.uuid4().hex[:12]
        return analysis_result

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline execution failed: {str(exc)}",
        ) from exc

