"""
ProfileFit FastAPI backend.

Endpoints:
    POST /profile              Parse a master profile (text/PDF/GitHub link).
                                Fast enough to run synchronously.
    POST /tailor                Start a full graph run (parse-if-needed ->
                                extract -> tailor -> critics -> decision ->
                                export) as a BACKGROUND JOB. Returns
                                immediately with a job_id.
    GET  /tailor/{job_id}       Poll a tailoring job's status/result.
    GET  /tailor/{job_id}/file/{kind}   Download the resulting docx/pdf.

Background jobs run in a plain Python thread (not FastAPI BackgroundTasks,
which still ties up within the same worker) so polling requests are never
blocked by a long-running graph.invoke() call. Job state lives in an
in-memory dict — fine for a single-process demo/grading deployment; would
need Redis/DB-backed storage for anything multi-worker or production-grade.
"""

import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))

from graph import graph  # noqa: E402
from graph_state import DecisionStatus, GraphState, MAX_REVISIONS, build_initial_state  # noqa: E402
from profile_parser import get_contact_items, parse_profile_data  # noqa: E402
from schemas import ProfileItem  # noqa: E402

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "output" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ProfileFit API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory stores (see module docstring for the production caveat) ---
PROFILE_STORE: dict[str, dict] = {}  # profile_id -> {"items": [...], "contact": {...}}
JOB_STORE: dict[str, dict] = {}      # job_id -> {"status": ..., "state": ..., "error": ...}


# ==================== Schemas ====================

class ProfileParseResponse(BaseModel):
    profile_id: str
    items_count: int
    contact_name: Optional[str] = None
    contact_details: Optional[str] = None


class TailorStartResponse(BaseModel):
    job_id: str
    status: str


class TailorStatusResponse(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "error"
    error: Optional[str] = None

    # Populated once status == "done"
    decision: Optional[str] = None
    decision_summary: Optional[str] = None
    match_score: Optional[int] = None
    security_flagged: bool = False
    flag_reason: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    revision_count: Optional[int] = None
    max_revisions: int = MAX_REVISIONS
    critic_cycle_count: Optional[int] = None
    missing_skills: list[str] = []
    omitted_experience_ids: list[str] = []
    professional_summary: Optional[str] = None
    sections: list[dict] = []  # [{section, title, dates, relevance_score, bullets:[...]}]
    candidate_name: Optional[str] = None
    docx_filename: Optional[str] = None
    pdf_filename: Optional[str] = None


# ==================== Helpers ====================

def _profile_items_from_store(profile_id: str) -> list[ProfileItem]:
    entry = PROFILE_STORE.get(profile_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"profile_id {profile_id!r} not found")
    return entry["items"]


def _run_graph_job(job_id: str, initial_state: GraphState) -> None:
    """Executed in a background thread — this is where graph.invoke() (the
    slow, multi-LLM-call part) actually happens."""
    try:
        final_state = graph.invoke(initial_state.model_dump())
        JOB_STORE[job_id] = {"status": "done", "state": final_state, "error": None}
    except Exception as e:  # noqa: BLE001 - surface any failure to the poller
        JOB_STORE[job_id] = {"status": "error", "state": None, "error": str(e)}


def _serialize_state(state: dict) -> dict:
    """Flatten a raw graph final_state dict into the JSON-safe shape the UI needs."""
    decision = state.get("decision")
    decision_value = decision.value if hasattr(decision, "value") else decision

    job_extraction = state.get("job_extraction")

    def _get(obj, key):
        if obj is None:
            return None
        return getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key)

    tailored_cv = state.get("tailored_cv")
    sections_out = []
    if tailored_cv is not None:
        raw_sections = _get(tailored_cv, "sections") or []
        for item in raw_sections:
            section_val = _get(item, "section")
            section_val = section_val.value if hasattr(section_val, "value") else section_val
            bullets = _get(item, "tailored_bullets") or []
            bullet_texts = [
                (b.text if hasattr(b, "text") else b.get("text"))
                for b in bullets
            ]
            sections_out.append({
                "section": section_val,
                "title": _get(item, "title"),
                "dates": _get(item, "dates"),
                "relevance_score": _get(item, "relevance_score"),
                "bullets": bullet_texts,
            })

    contact_items = state.get("contact_items") or []
    candidate_name = None
    if contact_items:
        first = contact_items[0]
        candidate_name = first.title if hasattr(first, "title") else first.get("title")

    docx_path = state.get("docx_path")
    pdf_path = state.get("pdf_path")

    return {
        "decision": decision_value,
        "decision_summary": state.get("decision_summary"),
        "match_score": state.get("match_score"),
        "security_flagged": bool(state.get("security_flagged")),
        "flag_reason": _get(job_extraction, "flag_reason"),
        "job_title": _get(job_extraction, "job_title"),
        "company": _get(job_extraction, "company"),
        "revision_count": state.get("revision_count"),
        "critic_cycle_count": state.get("critic_cycle_count"),
        "missing_skills": (_get(tailored_cv, "missing_skills") or []) if tailored_cv else [],
        "omitted_experience_ids": (_get(tailored_cv, "omitted_experience_ids") or []) if tailored_cv else [],
        "professional_summary": _get(tailored_cv, "professional_summary") if tailored_cv else None,
        "sections": sections_out,
        "candidate_name": candidate_name,
        "docx_filename": Path(docx_path).name if docx_path else None,
        "pdf_filename": Path(pdf_path).name if pdf_path else None,
        "_docx_path": docx_path,
        "_pdf_path": pdf_path,
    }


# ==================== Endpoints ====================

@app.post("/profile", response_model=ProfileParseResponse)
async def parse_profile(
    instruction: Optional[str] = Form(
        None, description='e.g. "github link is https://github.com/USERNAME"'
    ),
    file: Optional[UploadFile] = File(None, description="Resume/CV PDF upload"),
):
    """
    Parse a master profile. Fast enough (one or two LLM calls) to run
    synchronously — no job/polling needed here, unlike /tailor.
    """
    if file is not None:
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
        dest.write_bytes(await file.read())
        parse_instruction = f"read the cv file at this path: {dest}, and extract the profile data from it."
    elif instruction:
        parse_instruction = instruction
    else:
        raise HTTPException(status_code=400, detail="Provide either 'instruction' or 'file'.")

    items = parse_profile_data(parse_instruction, save=False)
    contact_items = get_contact_items(items)

    profile_id = uuid.uuid4().hex
    PROFILE_STORE[profile_id] = {
        "items": items,
        "contact": contact_items[0] if contact_items else None,
    }

    contact = contact_items[0] if contact_items else None
    return ProfileParseResponse(
        profile_id=profile_id,
        items_count=len(items),
        contact_name=contact.title if contact else None,
        contact_details=contact.description if contact else None,
    )


@app.post("/tailor", response_model=TailorStartResponse, status_code=202)
async def start_tailor(
    profile_id: str = Form(..., description="From a prior POST /profile call"),
    text: Optional[str] = Form(None, description="Raw job posting text"),
    url: Optional[str] = Form(None, description="Job posting URL"),
    file: Optional[UploadFile] = File(None, description="Job posting PDF"),
):
    """
    Start a full tailoring run as a BACKGROUND JOB and return immediately.
    Poll GET /tailor/{job_id} for status/result.
    """
    profile_items = _profile_items_from_store(profile_id)
    contact = PROFILE_STORE[profile_id]["contact"]
    contact_items = [contact] if contact else []

    if url:
        posting_source, fetch_first = f"job posting is: {url}", True
    elif file is not None:
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
        content = await file.read()
        dest.write_bytes(content)
        posting_source, fetch_first = f"pdf path is {dest}", True
    elif text:
        posting_source, fetch_first = text, False
    else:
        raise HTTPException(status_code=400, detail="Provide one of: text, url, file.")

    initial_state = build_initial_state(
        posting_source=posting_source,
        fetch_first=fetch_first,
        profile_items=profile_items,
        contact_items=contact_items,
    )

    job_id = uuid.uuid4().hex
    JOB_STORE[job_id] = {"status": "running", "state": None, "error": None}
    thread = threading.Thread(target=_run_graph_job, args=(job_id, initial_state), daemon=True)
    thread.start()

    return TailorStartResponse(job_id=job_id, status="running")


@app.get("/tailor/{job_id}", response_model=TailorStatusResponse)
async def get_tailor_status(job_id: str):
    entry = JOB_STORE.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"job_id {job_id!r} not found")

    if entry["status"] == "running":
        return TailorStatusResponse(job_id=job_id, status="running")
    if entry["status"] == "error":
        return TailorStatusResponse(job_id=job_id, status="error", error=entry["error"])

    serialized = _serialize_state(entry["state"])
    return TailorStatusResponse(
        job_id=job_id,
        status="done",
        **{k: v for k, v in serialized.items() if not k.startswith("_")},
    )


@app.get("/tailor/{job_id}/file/{kind}")
async def download_file(job_id: str, kind: str):
    if kind not in ("docx", "pdf"):
        raise HTTPException(status_code=400, detail="kind must be 'docx' or 'pdf'")
    entry = JOB_STORE.get(job_id)
    if entry is None or entry["status"] != "done":
        raise HTTPException(status_code=404, detail="Job not found or not finished yet.")
    serialized = _serialize_state(entry["state"])
    path = serialized.get(f"_{kind}_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"No {kind} file available for this job.")
    filename = Path(path).name
    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if kind == "docx"
        else "application/pdf"
    )
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/health")
async def health():
    return {"status": "ok"}