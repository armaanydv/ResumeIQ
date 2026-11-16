"""
Tests for Phase 4: Deterministic ATS Scoring Engine and LLM Service Pipeline.
"""

import pytest
from backend.app.core.scoring import (
    calculate_technical_skills_score,
    calculate_experience_relevance_score,
    calculate_project_relevance_score,
    calculate_keyword_match_score,
    calculate_education_score,
    compute_overall_ats_score,
)
from backend.app.services.rag_service import chunk_resume, ResumeVectorStore
from backend.app.services.parser_service import ParsedDocument
from backend.app.services.llm_service import analyze_resume_with_llm


def test_technical_skills_scoring_math():
    """Verify technical skill scoring adheres to exact ratio calculations."""
    assert calculate_technical_skills_score(5, 10) == 50.0
    assert calculate_technical_skills_score(8, 10) == 80.0
    assert calculate_technical_skills_score(10, 10) == 100.0
    assert calculate_technical_skills_score(0, 10) == 0.0
    assert calculate_technical_skills_score(0, 0) == 75.0  # Safe default


def test_education_scoring_heuristics():
    """Verify education and certification detection."""
    # 1. Degree only
    score_degree = calculate_education_score("B.S. in Computer Science", "")
    assert score_degree == 80.0

    # 2. Degree + Certification
    score_full = calculate_education_score("Master of Science in Software Engineering", "AWS Certified Solutions Architect")
    assert score_full == 100.0

    # 3. Empty education
    score_empty = calculate_education_score("", "")
    assert score_empty == 30.0


def test_project_scoring_penalties():
    """Verify that missing project sections trigger appropriate ATS penalties."""
    sections = {
        "experience": "Senior Engineer at Acme Corp.",
    }
    chunks = chunk_resume(sections)
    vector_store = ResumeVectorStore(chunks)

    # Missing project text
    score_missing = calculate_project_relevance_score("", vector_store, ["Python", "Docker"])
    assert score_missing == 35.0

    # Projects present with keyword matches
    sections_with_proj = {
        "projects": "Built an intelligent API using Python and Docker containerization.",
    }
    chunks_with_proj = chunk_resume(sections_with_proj)
    vs_proj = ResumeVectorStore(chunks_with_proj)
    score_with_proj = calculate_project_relevance_score(
        sections_with_proj["projects"], vs_proj, ["Python", "Docker"]
    )
    assert score_with_proj >= 70.0


def test_weighted_overall_ats_calculation():
    """Verify the 30/25/20/15/10 weighted formula mathematically."""
    sections = {
        "skills": "Python, FastAPI, Docker, Kubernetes, PostgreSQL",
        "experience": "Backend Developer at CloudScale designing microservices.",
        "projects": "ResumeIQ - AI resume evaluator with vector search.",
        "education": "B.S. in Computer Science",
        "certifications": "AWS Certified Developer",
    }
    chunks = chunk_resume(sections)
    vs = ResumeVectorStore(chunks)

    score = compute_overall_ats_score(
        matched_skills_count=4,
        total_skills_count=5,
        vector_store=vs,
        jd_responsibilities=["Design and deploy microservices with FastAPI and Docker."],
        jd_keywords=["Python", "FastAPI", "Docker", "PostgreSQL", "Kubernetes"],
        sections=sections,
        full_resume_text="Python FastAPI Docker Kubernetes PostgreSQL B.S. in Computer Science AWS Certified Developer",
    )

    # Verify score boundaries
    assert 0.0 <= score.overall_ats_score <= 100.0
    assert score.technical_skills_score == 80.0
    assert score.education_score == 100.0
    assert score.verdict in ["Excellent Match", "Strong Match", "Moderate Match", "Needs Optimization"]

    # Verify manual weighted sum matches overall_ats_score
    expected = (
        (score.technical_skills_score * 0.30)
        + (score.experience_relevance_score * 0.25)
        + (score.project_relevance_score * 0.20)
        + (score.keyword_match_score * 0.15)
        + (score.education_score * 0.10)
    )
    assert abs(score.overall_ats_score - round(expected, 1)) < 0.15


def test_analyze_resume_pipeline():
    """Verify end-to-end analysis output structure with grounded evidence and rewrites."""
    sections = {
        "skills": "Python, FastAPI, Docker, PostgreSQL",
        "experience": (
            "Senior Backend Engineer at Acme\n"
            "- Built REST APIs using FastAPI and Python.\n"
            "- Managed database optimizations on PostgreSQL."
        ),
        "projects": "AI Search Engine - Built vector search pipeline.",
        "education": "B.S. in Computer Science",
    }
    chunks = chunk_resume(sections)
    vs = ResumeVectorStore(chunks)

    parsed_doc = ParsedDocument(
        filename="test_resume.pdf",
        file_type="pdf",
        raw_text="Full resume text",
        cleaned_text=(
            "Senior Backend Engineer\n"
            "- Built REST APIs using FastAPI and Python.\n"
            "- Managed database optimizations on PostgreSQL."
        ),
        sections=sections,
        word_count=50,
        char_count=300,
        page_count=1,
    )

    jd_text = """
    Senior Python Backend Engineer
    We are looking for a Senior Python Backend Engineer with experience in FastAPI and Docker.
    Responsibilities:
    - Build scalable microservices using Python and FastAPI.
    - Deploy containerized applications with Docker and Kubernetes.
    Requirements:
    - Proficiency with Python, FastAPI, Docker, and PostgreSQL.
    - Experience optimizing relational database systems.
    """

    result = analyze_resume_with_llm(jd_text, parsed_doc, vs)

    assert result.filename == "test_resume.pdf"
    assert result.target_role != ""
    assert result.score_breakdown.overall_ats_score > 0
    assert len(result.matched_skills) > 0
    assert any(s.name.lower() in ["python", "fastapi"] for s in result.matched_skills)
    assert len(result.bullet_improvements) > 0
    assert len(result.actionable_recommendations) > 0
    assert result.experience_critique != ""

