"""
ProfileFit Streamlit UI.

Flow:
    1. Parse master profile (GitHub link / instruction, or upload a CV PDF).
    2. Submit a job posting (paste text, paste a URL, or upload a PDF).
    3. Poll the background /tailor job until done, showing live progress.
    4. Render the result: poisoned-posting alert, match score, missing
       skills, section breakdown, decision summary, and download buttons
       named "<candidate_name>_CV.docx/.pdf".

Run with the FastAPI backend already running (default http://localhost:8000):
    uvicorn api.FastAPI_app:app --reload
    streamlit run app/streamlit_app.py
"""

import os
import time

import requests
import streamlit as st

API_BASE_URL = os.getenv("PROFILEFIT_API_URL", "http://localhost:8000")
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 180

st.set_page_config(page_title="ProfileFit", page_icon="📄", layout="wide")
st.title("📄 ProfileFit — Job-Tailored CV Generator")

if "profile_id" not in st.session_state:
    st.session_state.profile_id = None
    st.session_state.profile_summary = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None


# ==================== Step 1: Master Profile ====================

st.header("1. Your Master Profile")

profile_source = st.radio(
    "How do you want to provide your profile?",
    ["GitHub link / instruction", "Upload CV (PDF)"],
    horizontal=True,
)

with st.form("profile_form"):
    if profile_source == "GitHub link / instruction":
        instruction = st.text_input(
            "GitHub link or instruction",
            placeholder="github link is https://github.com/yourusername",
        )
        profile_file = None
    else:
        instruction = None
        profile_file = st.file_uploader("Upload your CV", type=["pdf"])

    submitted_profile = st.form_submit_button("Parse Profile")

if submitted_profile:
    if not instruction and not profile_file:
        st.error("Please provide a GitHub link/instruction or upload a CV PDF.")
    else:
        with st.spinner("Parsing your profile..."):
            try:
                if profile_file is not None:
                    resp = requests.post(
                        f"{API_BASE_URL}/profile",
                        files={"file": (profile_file.name, profile_file.getvalue(), "application/pdf")},
                        timeout=120,
                    )
                else:
                    resp = requests.post(
                        f"{API_BASE_URL}/profile",
                        data={"instruction": instruction},
                        timeout=120,
                    )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.profile_id = data["profile_id"]
                st.session_state.profile_summary = data
                st.session_state.job_id = None  # reset any previous tailoring run
            except requests.RequestException as e:
                st.error(f"Failed to parse profile: {e}")

if st.session_state.profile_summary:
    summary = st.session_state.profile_summary
    st.success(
        f"✅ Profile parsed — {summary['items_count']} items found."
        + (f" Candidate: **{summary['contact_name']}**" if summary.get("contact_name") else "")
    )


# ==================== Step 2: Job Posting ====================

if st.session_state.profile_id:
    st.header("2. Job Posting")

    job_source = st.radio(
        "How do you want to provide the job posting?",
        ["Paste text", "Paste URL", "Upload PDF"],
        horizontal=True,
    )

    with st.form("job_form"):
        job_text = job_url = None
        job_file = None
        if job_source == "Paste text":
            job_text = st.text_area("Job posting text", height=200)
        elif job_source == "Paste URL":
            job_url = st.text_input("Job posting URL", placeholder="https://...")
        else:
            job_file = st.file_uploader("Upload job posting PDF", type=["pdf"], key="job_pdf")

        submitted_job = st.form_submit_button("Tailor My CV")

    def _looks_like_bare_url(text: str) -> bool:
        """Catch the common mistake of pasting a URL into the text box
        instead of switching to 'Paste URL' mode — a bare URL has nothing
        for the extractor to read as job requirements, and would otherwise
        just burn through retries and fail with a confusing error."""
        stripped = text.strip()
        return (
            stripped.startswith(("http://", "https://"))
            and " " not in stripped
            and "\n" not in stripped
        )

    if submitted_job:
        if job_text and _looks_like_bare_url(job_text):
            st.error(
                "That looks like a URL, not job posting text. Select **'Paste URL'** "
                "above instead — pasting a link into the text box won't work, since "
                "nothing will be fetched from it."
            )
        elif not job_text and not job_url and not job_file:
            st.error("Please provide job posting text, a URL, or a PDF.")
        else:
            data = {"profile_id": st.session_state.profile_id}
            files = None
            if job_file is not None:
                files = {"file": (job_file.name, job_file.getvalue(), "application/pdf")}
            elif job_url:
                data["url"] = job_url
            elif job_text:
                data["text"] = job_text

            try:
                resp = requests.post(f"{API_BASE_URL}/tailor", data=data, files=files, timeout=30)
                resp.raise_for_status()
                st.session_state.job_id = resp.json()["job_id"]
            except requests.RequestException as e:
                st.error(f"Failed to start tailoring job: {e}")


# ==================== Step 3: Poll + Render Result ====================

if st.session_state.job_id:
    st.header("3. Result")

    status_placeholder = st.empty()
    result = None
    elapsed = 0

    with st.spinner("Tailoring your CV — this can take a minute (parsing, extraction, "
                     "tailoring, and two quality-check agents running)..."):
        while elapsed < POLL_TIMEOUT_SECONDS:
            try:
                resp = requests.get(f"{API_BASE_URL}/tailor/{st.session_state.job_id}", timeout=15)
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as e:
                status_placeholder.error(f"Error polling job status: {e}")
                break

            if result["status"] == "done":
                break
            if result["status"] == "error":
                status_placeholder.error(f"Tailoring failed: {result.get('error')}")
                break

            status_placeholder.info(f"Still working... ({elapsed}s elapsed)")
            time.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS

    if result and result["status"] == "done":
        status_placeholder.empty()
    elif result and result["status"] == "error":
        status_placeholder.error(f"Tailoring failed: {result.get('error')}")
    elif result is None or result.get("status") == "running":
        status_placeholder.warning(
            f"Timed out after {POLL_TIMEOUT_SECONDS}s without finishing. The job "
            f"may still be running in the background — try refreshing in a bit, "
            f"or check the FastAPI server logs for what's taking long."
        )

    if result and result["status"] == "done":
        # --- Poisoned/spam posting alert (button-triggered reveal) ---
        if result.get("security_flagged"):
            st.error("🚨 **This job posting looks suspicious.**")
            with st.expander("Show flagged reason"):
                st.write(
                    result.get("flag_reason")
                    or "The posting contained content that did not match a normal job description."
                )
            st.warning("No CV was generated for this posting.")
        else:
            # --- Job info echo ---
            st.subheader(f"Tailored for: {result.get('job_title') or 'this role'}")
            if result.get("company"):
                st.caption(f"Company: {result['company']}")

            # --- Match score ---
            match_score = result.get("match_score")
            if match_score is not None:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Match Score", f"{match_score}%")
                with col2:
                    st.progress(match_score / 100)

            # --- Decision + revision transparency ---
            decision = result.get("decision")
            if decision == "APPROVED":
                st.success("✅ CV approved and ready to download.")
            elif decision == "REJECTED_MAX_REVISIONS":
                st.warning(
                    "This CV went through several revision attempts and is the best "
                    "version produced — consider adding more detail to your profile."
                )
            with st.expander(
                f"Why this result? (revision {result.get('revision_count', 0)}/"
                f"{result.get('max_revisions')}, {result.get('critic_cycle_count', 0)} "
                f"quality-check cycles)"
            ):
                st.write(result.get("decision_summary") or "No summary available.")

            # --- Missing skills ---
            if result.get("missing_skills"):
                st.subheader("Skills not found in your profile")
                for skill in result["missing_skills"]:
                    st.write(f"- {skill}")

            # --- Professional summary ---
            if result.get("professional_summary"):
                st.subheader("Professional Summary")
                st.write(result["professional_summary"])

            # --- Section breakdown ---
            if result.get("sections"):
                st.subheader("CV Sections")
                by_section: dict[str, list[dict]] = {}
                for item in result["sections"]:
                    by_section.setdefault(item["section"], []).append(item)
                for section_name, items in by_section.items():
                    with st.expander(f"{section_name.title()} ({len(items)})", expanded=True):
                        for item in sorted(items, key=lambda x: x["relevance_score"], reverse=True):
                            title = item["title"]
                            if item.get("dates"):
                                title += f"  |  {item['dates']}"
                            st.markdown(f"**{title}** — relevance {item['relevance_score']}/100")
                            for bullet in item["bullets"]:
                                st.write(f"  - {bullet}")

            if result.get("omitted_experience_ids"):
                with st.expander(f"Omitted items ({len(result['omitted_experience_ids'])})"):
                    st.write(
                        "These profile items weren't included in this version of the CV "
                        "(usually because they were less relevant to this specific role)."
                    )

            # --- Downloads, named candidate_name_CV ---
            st.subheader("Download")
            candidate_name = result.get("candidate_name") or "candidate"
            col1, col2 = st.columns(2)
            if result.get("docx_filename"):
                docx_resp = requests.get(
                    f"{API_BASE_URL}/tailor/{st.session_state.job_id}/file/docx", timeout=30
                )
                with col1:
                    st.download_button(
                        "⬇️ Download .docx",
                        docx_resp.content,
                        file_name=f"{candidate_name}_CV.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
            if result.get("pdf_filename"):
                pdf_resp = requests.get(
                    f"{API_BASE_URL}/tailor/{st.session_state.job_id}/file/pdf", timeout=30
                )
                with col2:
                    st.download_button(
                        "⬇️ Download .pdf",
                        pdf_resp.content,
                        file_name=f"{candidate_name}_CV.pdf",
                        mime="application/pdf",
                    )
            elif result.get("docx_filename"):
                st.caption("PDF not available (LibreOffice not installed on the server) — .docx only.")