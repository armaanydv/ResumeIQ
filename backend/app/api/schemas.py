"""
ResumeIQ Pydantic Schemas.
Defines strict data contracts for analysis requests, scoring breakdowns, and AI feedback.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SkillMatch(BaseModel):
    """Represents a technical or soft skill evaluated against the job description."""
    name: str
    category: str = Field(description="'hard_skill' or 'soft_skill'")
    status: str = Field(description="'matched', 'missing', or 'partial'")
    evidence: Optional[str] = Field(
        default=None,
        description="Verbatim or summarized snippet from resume proving this skill",
    )


class BulletImprovement(BaseModel):
    """Before-and-after critique of a resume bullet point."""
    original: str = Field(description="The original bullet point from the resume")
    improved: str = Field(
        description="Optimized bullet following Google's X-Y-Z formula without inventing facts"
    )
    critique: str = Field(
        description="Explanation of why the rewrite improves impact, action verbs, or clarity"
    )


class ScoreBreakdown(BaseModel):
    """
    Deterministic weighted ATS compatibility score breakdown (0.0 to 100.0).
    Formula:
    overall = (tech * 0.30) + (exp * 0.25) + (proj * 0.20) + (kw * 0.15) + (edu * 0.10)
    """
    technical_skills_score: float = Field(ge=0.0, le=100.0, description="30% weight")
    experience_relevance_score: float = Field(ge=0.0, le=100.0, description="25% weight")
    project_relevance_score: float = Field(ge=0.0, le=100.0, description="20% weight")
    keyword_match_score: float = Field(ge=0.0, le=100.0, description="15% weight")
    education_score: float = Field(ge=0.0, le=100.0, description="10% weight")
    overall_ats_score: float = Field(ge=0.0, le=100.0, description="Composite score")
    verdict: str = Field(description="High-level category, e.g. 'Strong Match'")


class AnalysisResponse(BaseModel):
    """Unified response payload sent to the frontend dashboard."""
    id: Optional[str] = Field(default=None, description="Unique analysis record identifier")
    timestamp: str
    filename: str
    target_role: Optional[str] = Field(default="Target Role", description="Role detected from JD")
    score_breakdown: ScoreBreakdown
    matched_skills: List[SkillMatch] = Field(default_factory=list)
    missing_skills: List[SkillMatch] = Field(default_factory=list)
    bullet_improvements: List[BulletImprovement] = Field(default_factory=list)
    experience_critique: str
    projects_critique: str
    actionable_recommendations: List[str] = Field(default_factory=list)
    word_count: int
    page_count: int

