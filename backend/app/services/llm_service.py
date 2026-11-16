"""
ResumeIQ LLM Integration Service.
Orchestrates prompt engineering, Gemini API calls, structured JSON generation,
and anti-hallucination guardrails with an offline fallback mode for testing.
"""

from datetime import datetime, timezone
import json
import re
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types

from backend.app.config import get_settings
from backend.app.api.schemas import (
    AnalysisResponse,
    BulletImprovement,
    ScoreBreakdown,
    SkillMatch,
)
from backend.app.services.parser_service import ParsedDocument
from backend.app.services.rag_service import ResumeVectorStore, build_rag_context
from backend.app.core.scoring import compute_overall_ats_score


# --- Extraction Prompts ---
ANALYSIS_SYSTEM_INSTRUCTION = """
You are a Senior Principal Technical Recruiter and ATS Optimization Expert.
Your objective is to evaluate a candidate's resume against a Job Description (JD) using strictly retrieved resume evidence.

CORE RULES:
1. STRICT TRUTHFULNESS: Never invent or hallucinate candidate experience, metrics, or technologies not present in the resume.
2. EVIDENCE GROUNDING: Only validate a skill if explicit evidence exists in the resume or retrieved evidence blocks.
3. GOOGLE X-Y-Z BULLET REWRITING: Select 3 weak bullet points from the candidate's experience or projects and rewrite them following:
   "Accomplished [X], as measured by [Y], by doing [Z]"
   Do NOT invent numerical metrics—if the candidate wrote "improved latency", specify "[quantify % or ms latency reduction]" in brackets as a guidance placeholder.
4. JSON ONLY: Return strictly valid JSON adhering to the specified schema.
"""


def _extract_jd_requirements(jd_text: str) -> Tuple[List[str], List[str], str]:
    """
    Extract core responsibilities, skill keywords, and estimated role title from JD text.
    """
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]

    # Extract title from first non-empty line
    target_role = "Target Role"
    if lines:
        first_line = lines[0]
        if len(first_line) < 80:
            target_role = re.sub(r"^(?:job\s+title|role|position)[:\s-]*", "", first_line, flags=re.IGNORECASE).strip()

    # Extract requirement bullet points or sentences
    responsibilities: List[str] = []
    keywords: List[str] = []

    # Common technical terms to look for
    common_terms = [
        "python", "javascript", "typescript", "go", "golang", "java", "c++", "rust",
        "fastapi", "django", "flask", "react", "next.js", "node.js", "angular", "vue",
        "docker", "kubernetes", "aws", "azure", "gcp", "postgresql", "mysql", "mongodb",
        "redis", "elasticsearch", "kafka", "graphql", "rest", "ci/cd", "git", "linux",
        "microservices", "rag", "llm", "machine learning", "pytorch", "tensorflow",
    ]

    jd_lower = jd_text.lower()
    for term in common_terms:
        # Match word boundaries
        if re.search(r"\b" + re.escape(term) + r"\b", jd_lower):
            keywords.append(term.capitalize() if len(term) > 3 else term.upper())

    # Find responsibility lines
    for line in lines:
        if line.startswith(("-", "•", "*")) or len(line) > 40:
            clean_l = re.sub(r"^[-•*\d.]+\s*", "", line).strip()
            if len(clean_l) >= 20 and len(clean_l) <= 200:
                responsibilities.append(clean_l)

    if not responsibilities:
        responsibilities = [jd_text[:150]]

    return responsibilities[:6], keywords, target_role


def _generate_offline_analysis(
    job_description: str,
    parsed_doc: ParsedDocument,
    vector_store: ResumeVectorStore,
    jd_responsibilities: List[str],
    jd_keywords: List[str],
    target_role: str,
) -> Tuple[List[SkillMatch], List[SkillMatch], List[BulletImprovement], str, str, List[str]]:
    """
    Deterministic offline heuristic engine.
    Used when GEMINI_API_KEY is not yet supplied, allowing full end-to-end testing without blocking.
    """
    resume_lower = parsed_doc.cleaned_text.lower()

    matched_skills: List[SkillMatch] = []
    missing_skills: List[SkillMatch] = []

    for kw in jd_keywords:
        kw_lower = kw.lower()
        if re.search(r"\b" + re.escape(kw_lower) + r"\b", resume_lower):
            # Retrieve evidence snippet from vector store
            ev = vector_store.search(kw, top_k=1)
            evidence_snippet = ev[0].text if ev else f"Mentioned in resume text."
            matched_skills.append(
                SkillMatch(
                    name=kw,
                    category="hard_skill",
                    status="matched",
                    evidence=evidence_snippet,
                )
            )
        else:
            missing_skills.append(
                SkillMatch(
                    name=kw,
                    category="hard_skill",
                    status="missing",
                    evidence=None,
                )
            )

    # Bullet point rewrites using existing resume lines
    bullet_improvements: List[BulletImprovement] = []
    lines = [
        l.strip()
        for l in parsed_doc.cleaned_text.splitlines()
        if l.strip().startswith("- ") and len(l) > 30
    ]

    sample_bullets = lines[:3]
    if not sample_bullets:
        # Fallback to any sentence
        sample_bullets = [
            "- Developed backend services using Python and FastAPI.",
            "- Maintained database queries and system performance.",
        ]

    for bullet in sample_bullets:
        clean_b = bullet.lstrip("- ").strip()
        bullet_improvements.append(
            BulletImprovement(
                original=clean_b,
                improved=f"Accelerated system throughput [by X%] by re-architecting {clean_b.lower()} utilizing industry best practices.",
                critique="Replaces passive description with active accomplishment verb and measurable impact metrics.",
            )
        )

    # Experience and project critiques
    exp_critique = (
        f"Your professional experience demonstrates strong foundational alignment with the {target_role} requirements. "
        "To strengthen ATS ranking, ensure all past position bullets begin with high-impact action verbs "
        "and explicitly tie technical implementations to business outcomes."
    )

    proj_critique = (
        "Your highlighted projects showcase practical engineering capability. "
        "We recommend explicitly listing the production architecture, hosting environment, "
        "and throughput benchmarks to mirror the technical depth expected in the job description."
    )

    recommendations = [
        f"Incorporate missing core skills: {', '.join([s.name for s in missing_skills[:4]]) or 'No critical hard skills missing'}.",
        "Reformat bullet points to follow Google's X-Y-Z formula (Accomplished X, measured by Y, by doing Z).",
        "Add a dedicated Skills summary section near the top of your resume for higher ATS parsing fidelity.",
        "Quantify project outcomes with concrete metrics (latency, revenue, users, or compute cost reductions).",
    ]

    return (
        matched_skills,
        missing_skills,
        bullet_improvements,
        exp_critique,
        proj_critique,
        recommendations,
    )


def analyze_resume_with_llm(
    job_description: str,
    parsed_doc: ParsedDocument,
    vector_store: ResumeVectorStore,
) -> AnalysisResponse:
    """
    Main analysis pipeline orchestrator:
    1. Extracts JD requirements & keywords.
    2. Retrieves grounded evidence from ResumeVectorStore.
    3. Calls Gemini API (or runs deterministic offline mode if API key is not yet set).
    4. Computes deterministic Python ATS scores.
    5. Returns unified AnalysisResponse.
    """
    settings = get_settings()

    # 1. Parse JD requirements & keywords
    jd_responsibilities, jd_keywords, target_role = _extract_jd_requirements(job_description)

    # 2. Retrieve grounded RAG evidence context
    rag_context = build_rag_context(vector_store, jd_responsibilities, top_k_per_req=2)

    # 3. Check if Gemini API Key is configured
    api_key = settings.GEMINI_API_KEY.strip()

    if api_key:
        try:
            client = genai.Client(api_key=api_key)

            prompt = f"""
TARGET ROLE: {target_role}

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME SUMMARY:
{parsed_doc.cleaned_text[:2000]}

{rag_context}

TASK:
Analyze the match between this candidate and the JD.
Return a valid JSON object matching this schema:
{{
  "matched_skills": [
    {{"name": "SkillName", "category": "hard_skill", "status": "matched", "evidence": "Resume evidence snippet"}}
  ],
  "missing_skills": [
    {{"name": "SkillName", "category": "hard_skill", "status": "missing", "evidence": null}}
  ],
  "bullet_improvements": [
    {{"original": "Original line", "improved": "Optimized XYZ line", "critique": "Why this is better"}}
  ],
  "experience_critique": "Detailed qualitative critique of experience section",
  "projects_critique": "Detailed qualitative critique of projects section",
  "actionable_recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"]
}}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=ANALYSIS_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.2,  # Low temperature for strict determinism
                ),
            )

            parsed_json = json.loads(response.text)

            matched_skills = [SkillMatch(**s) for s in parsed_json.get("matched_skills", [])]
            missing_skills = [SkillMatch(**s) for s in parsed_json.get("missing_skills", [])]
            bullet_improvements = [
                BulletImprovement(**b) for b in parsed_json.get("bullet_improvements", [])
            ]
            exp_critique = parsed_json.get("experience_critique", "Strong technical background.")
            proj_critique = parsed_json.get("projects_critique", "Relevant portfolio projects.")
            recommendations = parsed_json.get("actionable_recommendations", [])

        except Exception:
            # Fallback gracefully to offline heuristics if API quota or connection fails
            (
                matched_skills,
                missing_skills,
                bullet_improvements,
                exp_critique,
                proj_critique,
                recommendations,
            ) = _generate_offline_analysis(
                job_description,
                parsed_doc,
                vector_store,
                jd_responsibilities,
                jd_keywords,
                target_role,
            )
    else:
        # Offline mode when GEMINI_API_KEY is not yet supplied
        (
            matched_skills,
            missing_skills,
            bullet_improvements,
            exp_critique,
            proj_critique,
            recommendations,
        ) = _generate_offline_analysis(
            job_description,
            parsed_doc,
            vector_store,
            jd_responsibilities,
            jd_keywords,
            target_role,
        )

    # 4. Compute deterministic ATS scores mathematically (NEVER guessed by LLM)
    total_skills = len(matched_skills) + len(missing_skills)
    matched_count = len(matched_skills)

    score_breakdown = compute_overall_ats_score(
        matched_skills_count=matched_count,
        total_skills_count=total_skills,
        vector_store=vector_store,
        jd_responsibilities=jd_responsibilities,
        jd_keywords=jd_keywords,
        sections=parsed_doc.sections,
        full_resume_text=parsed_doc.cleaned_text,
    )

    # 5. Assemble and return unified response
    return AnalysisResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        filename=parsed_doc.filename,
        target_role=target_role,
        score_breakdown=score_breakdown,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        bullet_improvements=bullet_improvements,
        experience_critique=exp_critique,
        projects_critique=proj_critique,
        actionable_recommendations=recommendations,
        word_count=parsed_doc.word_count,
        page_count=parsed_doc.page_count,
    )
