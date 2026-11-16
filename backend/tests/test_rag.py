"""
Tests for Phase 3: Semantic Chunking, Embeddings, FAISS Vector Indexing, and RAG Retrieval.
"""

import pytest
import numpy as np

from backend.app.services.rag_service import (
    EmbeddingService,
    ResumeChunk,
    ResumeVectorStore,
    chunk_resume,
    build_rag_context,
)


@pytest.fixture(scope="module")
def sample_sections():
    """Sample resume sections for RAG tests."""
    return {
        "summary": "Full Stack Engineer with 4 years of experience building Python and FastAPI web services.",
        "skills": "Python, FastAPI, Docker, Kubernetes, PostgreSQL, Redis, React, TypeScript, Git",
        "experience": (
            "Senior Backend Developer at CloudScale (2022-Present)\n"
            "- Designed microservices architecture using FastAPI and PostgreSQL handling 5,000 req/sec.\n"
            "- Orchestrated multi-node container deployments with Kubernetes and Docker.\n"
            "- Reduced database query latency by 45% using Redis caching and optimized SQL indexes."
        ),
        "projects": (
            "ResumeIQ - AI Resume Analyzer\n"
            "- Built RAG pipeline using FAISS vector store and Sentence Transformers.\n"
            "- Integrated Gemini LLM for structured resume critique."
        ),
        "education": "B.S. in Computer Science - State University (2018-2022)",
    }


def test_chunk_resume(sample_sections):
    """Verify semantic chunking splits sections into atomic chunks with metadata."""
    chunks = chunk_resume(sample_sections, chunk_size=200, chunk_overlap=30)
    assert len(chunks) >= 4

    # Verify chunk structure
    for idx, c in enumerate(chunks):
        assert c.chunk_id == idx
        assert c.section in sample_sections
        assert len(c.text) >= 15

    # Check specific section chunk presence
    sections_found = {c.section for c in chunks}
    assert "experience" in sections_found
    assert "skills" in sections_found
    assert "projects" in sections_found


def test_embedding_service_singleton():
    """Verify EmbeddingService implements a clean singleton pattern."""
    service1 = EmbeddingService()
    service2 = EmbeddingService()
    assert service1 is service2


def test_embedding_generation():
    """Verify embedding output shape, dtype, and unit-length normalization."""
    service = EmbeddingService()
    texts = [
        "FastAPI microservices in Python",
        "Containerization with Docker and Kubernetes",
    ]
    vectors = service.embed_texts(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == 2
    assert vectors.shape[1] == 384  # bge-small-en or MiniLM dimension
    assert vectors.dtype == np.float32

    # Verify unit-vector normalization: norm should equal 1.0 (+/- epsilon)
    norm0 = np.linalg.norm(vectors[0])
    norm1 = np.linalg.norm(vectors[1])
    assert abs(norm0 - 1.0) < 1e-4
    assert abs(norm1 - 1.0) < 1e-4


def test_vector_store_semantic_retrieval(sample_sections):
    """Verify that querying for a requirement retrieves the most semantically relevant chunk."""
    chunks = chunk_resume(sample_sections)
    vector_store = ResumeVectorStore(chunks)

    # 1. Query for Kubernetes / container orchestration
    k8s_results = vector_store.search("Kubernetes container orchestration and deployment", top_k=2)
    assert len(k8s_results) > 0
    top_result = k8s_results[0]
    assert top_result.similarity > 0.40
    # Must retrieve the container or kubernetes bullet point
    assert "Kubernetes" in top_result.text or "Docker" in top_result.text

    # 2. Query for database performance & caching
    db_results = vector_store.search("Database query optimization and caching strategies", top_k=2)
    assert len(db_results) > 0
    top_db = db_results[0]
    assert "Redis" in top_db.text or "PostgreSQL" in top_db.text or "indexes" in top_db.text


def test_build_rag_context(sample_sections):
    """Verify formatted LLM context generation with evidence citations."""
    chunks = chunk_resume(sample_sections)
    vector_store = ResumeVectorStore(chunks)

    requirements = [
        "Experience with Kubernetes container clustering",
        "Proficiency with Python and FastAPI backend development",
        "Expertise in Rust programming language",  # Missing skill
    ]

    context = build_rag_context(vector_store, requirements, top_k_per_req=1)

    assert "### RETRIEVED RESUME EVIDENCE" in context
    assert "Requirement 1" in context
    assert "Kubernetes" in context
    assert "Requirement 2" in context
    assert "FastAPI" in context
    assert "Requirement 3" in context


if __name__ == "__main__":
    print("Running Phase 3 tests manually...")
    service = EmbeddingService()
    print("Embedding service initialized.")
    sections = {
        "skills": "Python, Docker, FastAPI",
        "experience": "Engineered Python microservices with Docker.",
    }
    chunks = chunk_resume(sections)
    vs = ResumeVectorStore(chunks)
    res = vs.search("Python backend development")
    print(f"Top result: {res[0].text} (Similarity: {res[0].similarity})")
    print("All manual tests passed!")

