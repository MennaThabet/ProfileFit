
"""
ProfileFit UI — Streamlit frontend for the Master Profile -> Job-Tailored CV
Generator.

Talks to the FastAPI backend in api.py. Run the two together:

    uvicorn api:app --reload --port 8000
    streamlit run streamlit_app.py

Set API_BASE_URL below (or via env var) if the API isn't on localhost:8000.
"""

import os
import time

import requests
import streamlit as st

API_BASE_URL = os.getenv("PROFILEFIT_API_URL", "http://localhost:8000")
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 300

st.set_page_config(page_title="ProfileFit — CV Tailor", page_icon="🧭", layout="wide")


# Helpers

def poll_job(job_id: str, label: str) -> dict:
    """Block (with a visible spinner) until a background job finishes."""
    start = time.time()
    with st.spinner(label):
        while True:
            resp = requests.get(f"{API_BASE_URL}/api/jobs/{job_id}", timeout=30)
            resp.raise_for_status()
            job = resp.json()
            if job["status"] == "done":
                return job["result"]
            if job["status"] == "error":
                raise RuntimeError(job.get("error") or "Job failed with no error message.")
            if time.time() - start > POLL_TIMEOUT_SECONDS:
                raise TimeoutError(f"Timed out waiting for job {job_id}.")
            time.sleep(POLL_INTERVAL_SECONDS)


def decision_badge(decision: str) -> None:
    colors = {
        "APPROVED": "green",
        "NEEDS_REVISION": "orange",
        "REJECTED_MAX_REVISIONS": "red",
        "REJECTED_FLAGGED_POSTING": "red",
        "PENDING": "gray",
    }
    color = colors.get(decision, "gray")
    st.markdown(f":{color}[**{decision}**]")


# Session state defaults

st.session_state.setdefault("profile_id", None)
st.session_state.setdefault("profile_summary", None)
st.session_state.setdefault("tailor_result", None)
st.session_state.setdefault("tailor_job_id", None)

st.title("🧭 ProfileFit")
st.caption("Master Profile → Job-Tailored CV Generator")

tab_profile, tab_tailor = st.tabs(["1️⃣ Build Master Profile", "2️⃣ Tailor a CV"])


# Tab 1 — Build Master Profile

with tab_profile:
    st.subheader("Build your master profile")
    st.write(
        "Provide a CV file, a GitHub link, and/or a LinkedIn export. "
        "This is parsed once and reused for every job you tailor against."
    )

    col1, col2 = st.columns(2)
    with col1:
        cv_file = st.file_uploader("CV (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
        github_url = st.text_input("GitHub profile URL", placeholder="https://github.com/username")
    with col2:
        linkedin_file = st.file_uploader("LinkedIn export / profile PDF", type=["pdf", "csv", "zip"])
        extra_instruction = st.text_area(
            "Additional notes (optional)",
            placeholder="Anything else the parser should know…",
            height=100,
        )

    if st.button("Parse Profile", type="primary", disabled=not (cv_file or github_url or linkedin_file or extra_instruction)):
        try:
            files = {}
            if cv_file is not None:
                files["cv_file"] = (cv_file.name, cv_file.getvalue())
            if linkedin_file is not None:
                files["linkedin_file"] = (linkedin_file.name, linkedin_file.getvalue())
            data = {}
            if github_url:
                data["github_url"] = github_url
            if extra_instruction:
                data["extra_instruction"] = extra_instruction

            resp = requests.post(f"{API_BASE_URL}/api/profiles", files=files or None, data=data, timeout=30)
            resp.raise_for_status()
            job_id = resp.json()["job_id"]

            result = poll_job(job_id, "Parsing your master profile…")
            st.session_state.profile_id = result["profile_id"]
            st.session_state.profile_summary = result
            st.success(f"Profile parsed: {result['item_count']} items found.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to parse profile: {e}")

    if st.session_state.profile_summary:
        summary = st.session_state.profile_summary
        st.divider()
        st.markdown(f"**Active profile:** `{st.session_state.profile_id}`")
        c1, c2 = st.columns(2)
        c1.metric("Items parsed", summary["item_count"])
        c2.metric("Contact info found", "Yes" if summary["contact_found"] else "No")

        with st.expander("View parsed items"):
            by_type: dict[str, list] = {}
            for item in summary["items"]:
                by_type.setdefault(item["type"], []).append(item)
            for item_type, items in by_type.items():
                st.markdown(f"**{item_type}**")
                for item in items:
                    st.write(f"- {item['title']}" + (f" ({item['dates']})" if item.get("dates") else ""))


# Tab 2 — Tailor a CV

with tab_tailor:
    st.subheader("Tailor your CV to a job posting")

    if not st.session_state.profile_id:
        st.warning("Parse a master profile in the first tab before tailoring a CV.")

    source_mode = st.radio("Job posting source", ["Paste text", "URL", "Upload PDF"], horizontal=True)
    posting_text = posting_url = None
    posting_file = None

    if source_mode == "Paste text":
        posting_text = st.text_area("Job posting text", height=200)
    elif source_mode == "URL":
        posting_url = st.text_input("Job posting URL", placeholder="https://…")
    else:
        posting_file = st.file_uploader("Job posting PDF", type=["pdf"])

    can_submit = st.session_state.profile_id and (posting_text or posting_url or posting_file)
    if st.button("Generate Tailored CV", type="primary", disabled=not can_submit):
        try:
            data = {"profile_id": st.session_state.profile_id}
            files = None
            if posting_text:
                data["posting_text"] = posting_text
            elif posting_url:
                data["posting_url"] = posting_url
            elif posting_file is not None:
                files = {"posting_file": (posting_file.name, posting_file.getvalue())}

            resp = requests.post(f"{API_BASE_URL}/api/tailor", data=data, files=files, timeout=30)
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            st.session_state.tailor_job_id = job_id

            result = poll_job(
                job_id,
                "Tailoring your CV — extracting requirements, selecting evidence, "
                "and running coverage/fabrication critics (this can take a minute)…",
            )
            st.session_state.tailor_result = result
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to tailor CV: {e}")

    result = st.session_state.tailor_result
    if result:
        st.divider()
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.write("**Decision**")
            decision_badge(result["decision"])
        c2.metric("Revisions", f"{result['revision_count']}/{result['max_revisions']}")
        c3.metric("Critic cycles", result["critic_cycle_count"])

        if result.get("security_flagged"):
            st.error(f"⚠️ Posting flagged during extraction: {result.get('flag_reason')}")

        if result.get("docx_available") or result.get("pdf_available"):
            dl1, dl2 = st.columns(2)
            job_id = st.session_state.tailor_job_id
            if result.get("docx_available"):
                docx_bytes = requests.get(f"{API_BASE_URL}/api/download/{job_id}/docx", timeout=30).content
                dl1.download_button("⬇️ Download .docx", docx_bytes, file_name="tailored_cv.docx")
            if result.get("pdf_available"):
                pdf_bytes = requests.get(f"{API_BASE_URL}/api/download/{job_id}/pdf", timeout=30).content
                dl2.download_button("⬇️ Download .pdf", pdf_bytes, file_name="tailored_cv.pdf")

        cv = result.get("tailored_cv")
        if cv:
            st.markdown("### Professional Summary")
            st.write(cv["professional_summary"])

            st.markdown("### Sections")
            by_section: dict[str, list] = {}
            for item in cv["sections"]:
                by_section.setdefault(item["section"], []).append(item)
            for section, items in by_section.items():
                st.markdown(f"**{section}**")
                for item in sorted(items, key=lambda x: x["relevance_score"], reverse=True):
                    title = item["title"] + (f"  |  {item['dates']}" if item.get("dates") else "")
                    st.write(f"**{title}**  (relevance {item['relevance_score']}/100)")
                    for bullet in item["tailored_bullets"]:
                        st.write(f"- {bullet['text']}")

            if cv.get("missing_skills"):
                st.markdown("### Missing Skills")
                st.write(", ".join(cv["missing_skills"]))

        cov = result.get("coverage_result")
        fab = result.get("fabrication_result")
        if cov or fab:
            with st.expander("Critic feedback"):
                if cov:
                    st.markdown(f"**Coverage Critic — {cov['status']}**")
                    st.write(cov["evidence"])
                    if cov.get("suggested_missing_items"):
                        st.write("Missing items:", ", ".join(cov["suggested_missing_items"]))
                if fab:
                    st.markdown(f"**Fabrication Critic — {fab['status']}**")
                    st.write(fab["evidence"])