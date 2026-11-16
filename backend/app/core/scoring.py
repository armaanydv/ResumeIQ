"""
ResumeIQ Deterministic ATS Scoring Engine.
Computes empirical compatibility scores using mathematical formulations, not LLM guesses.
"""

import re
from typing import Dict, List, Optional
from backend.app.api.schemas import ScoreBreakdown
from backend.app.services.rag_service import ResumeVectorStore


# Common degree and credential markers for education scoring
DEGREE_KEYWORDS = [
    r"\bb\.?s\.?\b",
    r"\bm\.?s\.?\b",
    r"\bb\.?tech\b",
    r"\bm\.?tech\b",
    r"\bbachelor",
    r"\bmaster",
    r"\bph\.?d\b",
    r"\bdoctorate",
    r"\bassociate",
    r"\bdiploma",
    r"\bcomputer science\b",
    r"\bsoftware engineering\b",
    r"\binformation technology\b",
    r"\belectrical engineering\b",
    r"\bdata science\b",
]

CERTIFICATION_KEYWORDS = [
    r"\baws\b",
    r"\bazure\b",
    r"\bgcp\b",
    r"\bgoogle cloud\b",
    r"\bcertified\b",
    r"\bcertificate\b",
    r"\bcka\b",
    r"\bckad\b",
    r"\bcissp\b",
    r"\bpmp\b",
    r"\bcomptia\b",
]


def calculate_technical_skills_score(matched_count: int, total_required: int) -> float:
    """
    Computes technical skill match score (0.0 to 100.0).
    Represents 30% of total ATS score.
    """
    if total_required <= 0:
        return 75.0  # Default neutral score if JD has no explicit skills parsed

    ratio = matched_count / total_required
    score = ratio * 100.0
    return round(min(100.0, max(0.0, score)), 1)


def calculate_experience_relevance_score(
    vector_store: ResumeVectorStore,
    jd_responsibilities: List[str],
) -> float:
    """
    Computes experience relevance score (0.0 to 100.0) based on vector cosine similarity.
    Represents 25% of total ATS score.
    """
    if not jd_responsibilities or not vector_store.chunks:
        return 50.0

    similarities: List[float] = []
    for resp in jd_responsibilities:
        results = vector_store.search(resp, top_k=1, min_similarity=0.0)
        if results:
            similarities.append(results[0].similarity)

    if not similarities:
        return 50.0

    avg_sim = sum(similarities) / len(similarities)
    # Cosine similarities for sentence-transformers typically range between 0.25 (weak) and 0.80 (very high)
    # Map [0.25, 0.75] linearly to [20.0, 100.0]
    scaled = 20.0 + ((avg_sim - 0.25) / (0.75 - 0.25)) * 80.0
    return round(min(100.0, max(15.0, scaled)), 1)


def calculate_project_relevance_score(
    project_text: Optional[str],
    vector_store: ResumeVectorStore,
    jd_keywords: List[str],
) -> float:
    """
    Computes project relevance score (0.0 to 100.0).
    Represents 20% of total ATS score.
    Penalizes candidates with empty or missing project sections.
    """
    if not project_text or not project_text.strip():
        return 35.0  # Missing project section penalty

    # Query project chunks against JD requirements
    project_chunks = [c for c in vector_store.chunks if c.section == "projects"]
    if not project_chunks:
        # Candidate has project text but it wasn't chunked into the projects section
        return 55.0

    # Test how many JD tech keywords appear in the projects text
    proj_lower = project_text.lower()
    hits = sum(1 for kw in jd_keywords if kw.lower() in proj_lower)
    hit_ratio = hits / max(1, len(jd_keywords))

    score = 40.0 + (hit_ratio * 60.0)
    return round(min(100.0, max(20.0, score)), 1)


def calculate_keyword_match_score(resume_text: str, jd_keywords: List[str]) -> float:
    """
    Computes exact keyword presence density across the resume (0.0 to 100.0).
    Represents 15% of total ATS score.
    """
    if not jd_keywords:
        return 75.0

    text_lower = resume_text.lower()
    hits = sum(1 for kw in jd_keywords if kw.lower() in text_lower)
    ratio = hits / len(jd_keywords)

    score = ratio * 100.0
    return round(min(100.0, max(0.0, score)), 1)


def calculate_education_score(
    education_text: Optional[str],
    certifications_text: Optional[str],
) -> float:
    """
    Computes education and certification alignment score (0.0 to 100.0).
    Represents 10% of total ATS score.
    """
    score = 40.0  # Base score

    combined_text = f"{education_text or ''} {certifications_text or ''}".lower()
    if not combined_text.strip():
        return 30.0

    # Check for formal degrees
    has_degree = any(re.search(pat, combined_text) for pat in DEGREE_KEYWORDS)
    if has_degree:
        score += 40.0

    # Check for professional certifications
    has_cert = any(re.search(pat, combined_text) for pat in CERTIFICATION_KEYWORDS)
    if has_cert:
        score += 20.0

    return round(min(100.0, max(10.0, score)), 1)


def compute_overall_ats_score(
    matched_skills_count: int,
    total_skills_count: int,
    vector_store: ResumeVectorStore,
    jd_responsibilities: List[str],
    jd_keywords: List[str],
    sections: Dict[str, str],
    full_resume_text: str,
) -> ScoreBreakdown:
    """
    Calculates the complete deterministic weighted ATS Score Breakdown.

    Weights:
    - Technical Skills: 30%
    - Experience Relevance: 25%
    - Project Relevance: 20%
    - Keyword Match: 15%
    - Education: 10%
    """
    s_tech = calculate_technical_skills_score(matched_skills_count, total_skills_count)
    s_exp = calculate_experience_relevance_score(vector_store, jd_responsibilities)
    s_proj = calculate_project_relevance_score(
        sections.get("projects"), vector_store, jd_keywords
    )
    s_kw = calculate_keyword_match_score(full_resume_text, jd_keywords)
    s_edu = calculate_education_score(
        sections.get("education"), sections.get("certifications")
    )

    # Weighted calculation
    overall = (
        (s_tech * 0.30)
        + (s_exp * 0.25)
        + (s_proj * 0.20)
        + (s_kw * 0.15)
        + (s_edu * 0.10)
    )
    overall_rounded = round(min(100.0, max(0.0, overall)), 1)

    # Determine verdict label
    if overall_rounded >= 85.0:
        verdict = "Excellent Match"
    elif overall_rounded >= 70.0:
        verdict = "Strong Match"
    elif overall_rounded >= 50.0:
        verdict = "Moderate Match"
    else:
        verdict = "Needs Optimization"

    return ScoreBreakdown(
        technical_skills_score=s_tech,
        experience_relevance_score=s_exp,
        project_relevance_score=s_proj,
        keyword_match_score=s_kw,
        education_score=s_edu,
        overall_ats_score=overall_rounded,
        verdict=verdict,
    )
