"""
ResumeIQ - Main FastAPI Application Entrypoint.
Initializes middleware, routing, and system health checks.
"""

from datetime import datetime, timezone
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.app.config import get_settings
from backend.app.api.routes_analyze import router as analyze_router

# Load cached settings
settings = get_settings()

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="ResumeIQ API - AI-Powered Resume & Job Description Compatibility Analyzer",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS)
# Allows our frontend to call this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(analyze_router, prefix="/api", tags=["Analysis"])

# Mount static frontend directory if present
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    assets_dir = os.path.join(frontend_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import UploadFile, File, Form
from backend.app.services.parser_service import parse_document
from backend.app.services.rag_service import chunk_resume, ResumeVectorStore
from backend.app.services.llm_service import analyze_resume_with_llm

templates = Jinja2Templates(directory=frontend_dir) if os.path.exists(frontend_dir) else None


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    fav = os.path.join(frontend_dir, "assets", "favicon.png")
    if os.path.exists(fav):
        return FileResponse(fav)
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.get("/", tags=["General"])
async def root(request: Request):
    """
    Root endpoint.
    If requested from a web browser (Accept includes text/html), delivers the web application.
    If requested by an API client/test, returns application metadata.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and templates:
        return templates.TemplateResponse(request=request, name="index.html", context={"analysis_complete": False})

    return {
        "app": settings.APP_NAME,
        "tagline": "Optimize your resume. Match your dream job.",
        "status": "online",
        "docs": "/docs",
        "api_health": "/api/health",
    }


@app.post("/analyze", response_class=HTMLResponse, tags=["General"])
async def analyze_form_submit(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(...),
):
    """
    Standard HTML multipart form submission endpoint.
    Allows users to analyze their resume with zero client-side JavaScript complexity.
    """
    if not templates:
        return JSONResponse(status_code=500, content={"detail": "Frontend templates not configured."})

    try:
        content_bytes = await file.read()
        if not content_bytes or len(content_bytes) == 0:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"analysis_complete": False, "form_error": "The uploaded file is empty (0 bytes). Please upload a valid resume document."},
            )

        filename = file.filename or "resume.pdf"
        _, ext = os.path.splitext(filename)
        if ext.lower() not in {".pdf", ".docx"}:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"analysis_complete": False, "form_error": f"Unsupported file extension '{ext}'. Only .pdf and .docx are supported."},
            )

        parsed_doc = parse_document(content_bytes, filename)
        chunks = chunk_resume(parsed_doc.sections)
        if not chunks:
            chunks = chunk_resume({"general": parsed_doc.cleaned_text})
        vector_store = ResumeVectorStore(chunks)

        analysis = analyze_resume_with_llm(
            job_description=job_description,
            parsed_doc=parsed_doc,
            vector_store=vector_store,
        )

        matched_skills = [s.name for s in analysis.matched_skills]
        missing_skills = [
            {
                "name": s.name,
                "reason": s.evidence or "Required competency for target role",
                "priority": "High" if s.category == "hard_skill" else "Medium",
            }
            for s in analysis.missing_skills
        ]

        recommendations = [
            {
                "title": f"Quantify impact: {b.original[:42]}...",
                "badge": "HIGH IMPACT" if idx == 0 else "RECOMMENDED",
                "explanation": b.critique,
                "original": b.original,
                "improved": b.improved,
                "points": 5 if idx == 0 else 3,
            }
            for idx, b in enumerate(analysis.bullet_improvements)
        ]

        overall = analysis.score_breakdown.overall_ats_score
        potential = min(100, overall + 8)

        context = {
            "analysis_complete": True,
            "result_filename": filename,
            "result_target_role": analysis.target_role or "Target Role",
            "result_overall_score": str(overall),
            "result_verdict": analysis.score_breakdown.verdict,
            "result_potential_score": str(potential),
            "result_summary_title": "You’re in a competitive position." if overall >= 75 else "Good foundation with improvement areas.",
            "result_summary_desc": analysis.experience_critique or "Your resume aligns well with this role. Addressing the recommendations below could lift your score into the top applicant tier.",
            "result_ranking_title": f"Scored {overall}/100",
            "result_ranking_desc": "Evaluated against target JD using semantic RAG vector retrieval.",
            "result_tech_score": str(analysis.score_breakdown.technical_skills_score),
            "result_kw_score": str(analysis.score_breakdown.keyword_match_score),
            "result_exp_score": str(analysis.score_breakdown.experience_relevance_score),
            "result_proj_score": str(analysis.score_breakdown.project_relevance_score),
            "result_edu_score": str(analysis.score_breakdown.education_score),
            "matched_skills_count": str(len(analysis.matched_skills)),
            "total_skills_count": str(len(analysis.matched_skills) + len(analysis.missing_skills)),
            "missing_skills_count": str(len(analysis.missing_skills)),
            "matched_skills_list": matched_skills,
            "missing_skills_list": missing_skills,
            "recommendations_count": str(len(recommendations)),
            "recommendations_list": recommendations,
            "submitted_jd": job_description,
        }
        return templates.TemplateResponse(request=request, name="index.html", context=context)
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"analysis_complete": False, "form_error": f"Error analyzing document: {str(exc)}"},
        )


@app.get("/app", tags=["General"])
async def app_view(request: Request):
    """Direct route to serve the frontend web application."""
    if templates:
        return templates.TemplateResponse(request=request, name="index.html", context={"analysis_complete": False})
    return JSONResponse(
        status_code=404,
        content={"detail": "Frontend application assets not found."},
    )


@app.get("/api/health", tags=["Monitoring"])
async def health_check():
    """
    System health-check endpoint.
    Returns status, environment, server timestamp, and readiness checks.
    """
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gemini_ready": bool(settings.GEMINI_API_KEY.strip()),
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
