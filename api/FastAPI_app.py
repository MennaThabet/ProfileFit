
"""
ProfileFit API — FastAPI backend for the Master Profile -> Job-Tailored CV
Generator pipeline.

Place this file in the SAME directory as graph.py, graph_state.py,
schemas.py, profile_parser.py, etc. (i.e. wherever `from graph import graph`
already works for you today).

Run with:
    uvicorn api:app --reload --port 8000

Design notes
------------
- Profile parsing and CV tailoring are both potentially slow (multiple LLM
  calls, a bounded critic-revision loop). Both are exposed as background
  JOBS you poll, not synchronous request/response calls, so the frontend
  never has to hold a connection open for a minute+.
- Parsed profiles are cached server-side (PROFILES) so a user can parse
  their master profile ONCE and tailor it against many job postings without
  re-parsing every time.
- Storage is in-memory (dict) for simplicity — swap for Redis/DB if this
  needs to survive server restarts or run with multiple workers.
- graph.py's export_node always writes to the fixed path
  output/tailored_cv.docx. Since jobs can run concurrently, we copy that
  file to a job-specific path immediately after the graph finishes, so
  concurrent tailoring jobs don't clobber each other's output.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Support both `python api/FastAPI_app.py` and Uvicorn launched from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for import_path in (PROJECT_ROOT, AGENT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from agent.graph import graph
from agent.graph_state import MAX_REVISIONS, DecisionStatus, build_initial_state
from agent.profile_parser import get_contact_items, parse_profile_data
from agent.schemas import ProfileItem

app = FastAPI(title="ProfileFit API", version="1.0.0")

# Working directories
UPLOAD_DIR = Path("uploads")
JOB_OUTPUT_DIR = Path("output") / "jobs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory stores
PROFILES: dict[str, dict] = {}   # profile_id -> {items, contact_items, created_at}
JOBS: dict[str, dict] = {}       # job_id -> {type, status, result, error, created_at}


# Schemas (API-facing)

class JobStatus(BaseModel):
    job_id: str
    type: str
    status: str  # pending | running | done | error
    error: Optional[str] = None


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload").suffix
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _new_job(job_type: str) -> str:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "type": job_type,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return job_id


# Background workers

def _run_parse_profile_job(job_id: str, instruction: str) -> None:
    JOBS[job_id]["status"] = "running"
    try:
        items: list[ProfileItem] = parse_profile_data(instruction, save=False)
        contact_items = get_contact_items(items)

        profile_id = uuid.uuid4().hex
        PROFILES[profile_id] = {
            "items": items,
            "contact_items": contact_items,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "profile_id": profile_id,
            "item_count": len(items),
            "contact_found": len(contact_items) > 0,
            "items": [item.model_dump() for item in items],
        }
    except Exception as e:  # noqa: BLE001
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


def _run_tailor_job(
    job_id: str,
    posting_source: str,
    fetch_first: bool,
    profile_id: Optional[str],
    profile_instruction: Optional[str],
) -> None:
    JOBS[job_id]["status"] = "running"
    try:
        profile_items = None
        contact_items = None
        if profile_id:
            cached = PROFILES.get(profile_id)
            if cached is None:
                raise ValueError(f"Unknown profile_id: {profile_id}")
            profile_items = cached["items"]
            contact_items = cached["contact_items"]

        initial_state = build_initial_state(
            posting_source=posting_source,
            fetch_first=fetch_first,
            profile_instruction=profile_instruction if not profile_items else None,
            profile_items=profile_items,
            contact_items=contact_items,
        )
        final_state = graph.invoke(initial_state.model_dump())

        # Copy the exported files (if any) to a job-specific path so
        # concurrent jobs don't overwrite each other's output.
        docx_dest = pdf_dest = None
        raw_docx = final_state.get("docx_path")
        raw_pdf = final_state.get("pdf_path")
        if raw_docx and Path(raw_docx).exists():
            docx_dest = JOB_OUTPUT_DIR / f"{job_id}.docx"
            shutil.copy(raw_docx, docx_dest)
        if raw_pdf and Path(raw_pdf).exists():
            pdf_dest = JOB_OUTPUT_DIR / f"{job_id}.pdf"
            shutil.copy(raw_pdf, pdf_dest)

        decision = final_state.get("decision")
        tailored_cv = final_state.get("tailored_cv")
        coverage_result = final_state.get("coverage_result")
        fabrication_result = final_state.get("fabrication_result")

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "decision": decision.value if isinstance(decision, DecisionStatus) else decision,
            "revision_count": final_state.get("revision_count"),
            "critic_cycle_count": final_state.get("critic_cycle_count"),
            "max_revisions": MAX_REVISIONS,
            "security_flagged": final_state.get("security_flagged"),
            "flag_reason": getattr(final_state.get("job_extraction"), "flag_reason", None),
            "tailored_cv": tailored_cv.model_dump() if tailored_cv else None,
            "coverage_result": coverage_result.model_dump() if coverage_result else None,
            "fabrication_result": fabrication_result.model_dump() if fabrication_result else None,
            "docx_available": docx_dest is not None,
            "pdf_available": pdf_dest is not None,
        }
    except Exception as e:  # noqa: BLE001
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


# Endpoints — Profile

@app.post("/api/profiles", response_model=JobStatus)
def create_profile(
    background_tasks: BackgroundTasks,
    cv_file: Optional[UploadFile] = File(default=None, description="CV as PDF/DOCX/TXT"),
    github_url: Optional[str] = Form(default=None),
    linkedin_file: Optional[UploadFile] = File(default=None, description="LinkedIn export/profile PDF"),
    extra_instruction: Optional[str] = Form(
        default=None, description="Any additional free-text instruction for the parser"
    ),
):
    """Parse a master profile from a CV file, a GitHub link, and/or a LinkedIn
    export, and cache it server-side. Returns a job_id — poll GET /api/jobs/{job_id}."""
    parts: list[str] = []

    if cv_file is not None:
        cv_path = _save_upload(cv_file)
        parts.append(f"read the cv file at this path: {cv_path}, and extract the profile data from it.")
    if github_url:
        parts.append(f"the candidate's github link is {github_url}; extract relevant project data from it.")
    if linkedin_file is not None:
        li_path = _save_upload(linkedin_file)
        parts.append(f"also read the linkedin export/profile file at this path: {li_path}.")
    if extra_instruction:
        parts.append(extra_instruction)

    instruction = " ".join(parts).strip()
    if not instruction:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of cv_file, github_url, linkedin_file, or extra_instruction.",
        )

    job_id = _new_job("parse_profile")
    background_tasks.add_task(_run_parse_profile_job, job_id, instruction)
    return JobStatus(job_id=job_id, type="parse_profile", status="pending")


@app.get("/api/profiles/{profile_id}")
def get_profile(profile_id: str):
    cached = PROFILES.get(profile_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Unknown profile_id")
    return {
        "profile_id": profile_id,
        "item_count": len(cached["items"]),
        "contact_found": len(cached["contact_items"]) > 0,
        "items": [item.model_dump() for item in cached["items"]],
        "created_at": cached["created_at"],
    }


# Endpoints — Tailoring

@app.post("/api/tailor", response_model=JobStatus)
def create_tailor_job(
    background_tasks: BackgroundTasks,
    profile_id: Optional[str] = Form(
        default=None, description="A profile_id previously returned by POST /api/profiles"
    ),
    profile_instruction: Optional[str] = Form(
        default=None,
        description="Alternative to profile_id: parse the profile inline for this run only "
        "(not cached). Ignored if profile_id is given.",
    ),
    posting_text: Optional[str] = Form(default=None, description="Raw job posting text"),
    posting_url: Optional[str] = Form(default=None, description="URL of the job posting"),
    posting_file: Optional[UploadFile] = File(default=None, description="Job posting as a PDF"),
):
    """Run the full parse/extract/retrieve/tailor/critic/decision/export pipeline
    for ONE job posting against a (cached or inline) master profile."""
    if not profile_id and not profile_instruction:
        raise HTTPException(
            status_code=400,
            detail="Provide either profile_id (from /api/profiles) or profile_instruction.",
        )
    if profile_id and profile_id not in PROFILES:
        raise HTTPException(status_code=404, detail="Unknown profile_id")

    sources_given = sum(bool(x) for x in (posting_text, posting_url, posting_file))
    if sources_given != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of posting_text, posting_url, or posting_file.",
        )

    if posting_url:
        posting_source, fetch_first = f"job posting is: {posting_url}", True
    elif posting_file is not None:
        pdf_path = _save_upload(posting_file)
        posting_source, fetch_first = f"pdf path is {pdf_path}", True
    else:
        posting_source, fetch_first = posting_text, False

    job_id = _new_job("tailor")
    background_tasks.add_task(
        _run_tailor_job, job_id, posting_source, fetch_first, profile_id, profile_instruction
    )
    return JobStatus(job_id=job_id, type="tailor", status="pending")


# Endpoints — Jobs & downloads

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"job_id": job_id, **job}


@app.get("/api/download/{job_id}/docx")
def download_docx(job_id: str):
    path = JOB_OUTPUT_DIR / f"{job_id}.docx"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No .docx available for this job")
    return FileResponse(
        path,
        filename="tailored_cv.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.get("/api/download/{job_id}/pdf")
def download_pdf(job_id: str):
    path = JOB_OUTPUT_DIR / f"{job_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No .pdf available for this job")
    return FileResponse(path, filename="tailored_cv.pdf", media_type="application/pdf")


@app.get("/api/health")
def health():
    return {"status": "ok"}