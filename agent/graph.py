"""LangGraph orchestration for the ProfileFit tailoring pipeline."""

import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from coverage_critic import run_coverage_critic
from decision import build_fallback_output, run_decision_agent
from exporter import default_output_path, export_cv
from fabrication_critic import run_fabrication_critic
from graph_state import DecisionStatus, GraphState, MAX_REVISIONS
from profile_parser import get_contact_items, parse_profile_data
from rag import rag_search
from requirement_extractor import extract_requirements
from selector_tailor import generate_tailored_cv, job_requirements_from_extraction


def _log(message: str) -> None:
    print(f"[ProfileFit] {message}")


def _as_state(state: GraphState | dict) -> GraphState:
    return state if isinstance(state, GraphState) else GraphState.model_validate(state)


def parse_profile_node(state: GraphState) -> dict:
    state = _as_state(state)
    if state.profile_items:
        _log(f"Profile ready: {len(state.profile_items)} items")
        return {"contact_items": get_contact_items(state.profile_items)}
    if not state.profile_instruction:
        raise ValueError("profile_instruction or profile_items is required")

    items = parse_profile_data(state.profile_instruction, save=False)
    _log(f"Profile parsed: {len(items)} items")
    return {"profile_items": items, "contact_items": get_contact_items(items)}


def extract_job_node(state: GraphState) -> dict:
    state = _as_state(state)
    extraction = extract_requirements(
        state.posting_source,
        fetch_first=state.fetch_first,
        save=False,
    )
    status = "FLAGGED" if extraction.flagged_for_review else "ready"
    _log(f"Job extraction {status}: {extraction.job_title}")
    return {
        "job_extraction": extraction,
        "job_reqs": job_requirements_from_extraction(extraction),
        "security_flagged": extraction.flagged_for_review,
    }


def retrieve_evidence_node(state: GraphState) -> dict:
    state = _as_state(state)
    if state.security_flagged:
        _log("Evidence retrieval skipped: posting is flagged")
        return {"rag_sources": []}
    if state.profile_items:
        _log("Evidence retrieval skipped: using parsed profile items")
        return {"rag_sources": []}
    if state.job_reqs is None:
        raise ValueError("job_reqs must be populated before evidence retrieval")

    query = (
        "Candidate profile experiences, projects, education, and skills relevant to "
        f"{state.job_reqs.job_title}. Required skills: "
        f"{', '.join(state.job_reqs.required_skills)}. Preferred skills: "
        f"{', '.join(state.job_reqs.preferred_skills)}."
    )
    result = json.loads(
        rag_search(query, top_k=10, source_filter="master_profiles")
    )
    if result.get("message") and not result.get("sources"):
        raise RuntimeError(f"RAG retrieval failed: {result['message']}")
    _log(f"Evidence retrieved: {len(result.get('sources', []))} source chunks")
    return {"rag_sources": result.get("sources", [])}


def select_and_tailor_node(state: GraphState) -> dict:
    state = _as_state(state)
    if state.security_flagged:
        _log("Tailoring skipped: posting is flagged")
        return {}
    if state.job_reqs is None:
        raise ValueError("job_reqs must be populated before tailoring")

    cv = generate_tailored_cv(
        state.job_reqs,
        candidate_items=state.profile_items or None,
        rag_sources=state.rag_sources or None,
        revision_feedback=state.revision_feedback,
    )
    cycle = state.critic_cycle_count + 1
    _log(
        f"Tailoring complete: revision #{state.revision_count}, "
        f"critic cycle #{cycle}, {len(cv.sections)} sections"
    )
    return {"tailored_cv": cv, "critic_cycle_count": cycle}


def coverage_critic_node(state: GraphState) -> dict:
    state = _as_state(state)
    if state.security_flagged or state.tailored_cv is None or state.job_reqs is None:
        return {}
    result = run_coverage_critic(
        state.tailored_cv,
        state.job_reqs,
        source_items=state.profile_items,
    )
    _log(
        f"Coverage critic: {result.status.value} | match_score={result.match_score}/100 "
        f"(cycle #{state.critic_cycle_count})"
    )
    return {
        "coverage_result": result,
        "match_score": result.match_score,
    }


def fabrication_critic_node(state: GraphState) -> dict:
    state = _as_state(state)
    if state.security_flagged or state.tailored_cv is None:
        return {}
    result = run_fabrication_critic(
        state.tailored_cv,
        candidate_items=state.profile_items or None,
        rag_sources=state.rag_sources or None,
    )
    _log(f"Fabrication critic: {result.status.value} (cycle #{state.critic_cycle_count})")
    return {
        "fabrication_result": result
    }


def decision_node(state: GraphState) -> dict:
    """
    Routing logic here is deliberately plain, deterministic Python — NOT an
    LLM call.
    """
    state = _as_state(state)
    if state.security_flagged:
        _log("Decision: REJECTED_FLAGGED_POSTING")
        return {"decision": DecisionStatus.REJECTED_FLAGGED_POSTING}

    coverage = state.coverage_result
    fabrication = state.fabrication_result
    if coverage is None or fabrication is None:
        return {"decision": DecisionStatus.NEEDS_REVISION}

    overall_pass = coverage.status.value == "PASS" and fabrication.status.value == "PASS"

    try:
        agent_output = run_decision_agent(
            coverage,
            fabrication,
            overall_pass=overall_pass,
            revision_count=state.revision_count,
            max_revisions=MAX_REVISIONS,
        )
    except Exception as e:  # noqa: BLE001 - synthesis failing must not block routing
        _log(f"Decision Agent synthesis failed ({e}); using deterministic fallback text")
        agent_output = build_fallback_output(coverage, fabrication, overall_pass)

    if overall_pass:
        _log(
            f"Decision: APPROVED after revision #{state.revision_count}, "
            f"critic cycle #{state.critic_cycle_count}"
        )
        return {
            "decision": DecisionStatus.APPROVED,
            "revision_feedback": None,
            "decision_summary": agent_output.summary_for_human,
        }

    next_revision = state.revision_count + 1
    decision = (
        DecisionStatus.REJECTED_MAX_REVISIONS
        if next_revision >= MAX_REVISIONS
        else DecisionStatus.NEEDS_REVISION
    )
    _log(
        f"Decision: {decision.value} | revision #{next_revision} of {MAX_REVISIONS} | "
        f"critic cycle #{state.critic_cycle_count} | "
        f"coverage={coverage.status.value}, fabrication={fabrication.status.value}"
    )
    return {
        "decision": decision,
        "revision_count": next_revision,
        "revision_feedback": agent_output.prioritized_feedback,
        "decision_summary": agent_output.summary_for_human,
    }


def decision_router(state: GraphState) -> str:
    state = _as_state(state)
    if state.decision == DecisionStatus.APPROVED:
        return "export"
    if state.decision == DecisionStatus.NEEDS_REVISION:
        return "revise"
    return "stop"


def export_node(state: GraphState) -> dict:
    state = _as_state(state)
    if state.tailored_cv is None:
        raise ValueError("tailored_cv must be populated before export")
    result = export_cv(
        state.tailored_cv,
        contact_items=state.contact_items,
        output_path=default_output_path(state.contact_items),
    )
    _log(f"Export complete: DOCX={result['docx_path']}, PDF={result['pdf_path']}")
    return {
        "docx_path": str(result["docx_path"]) if result["docx_path"] else None,
        "pdf_path": str(result["pdf_path"]) if result["pdf_path"] else None,
    }


builder = StateGraph(GraphState)
builder.add_node("parse_profile", parse_profile_node)
builder.add_node("extract_job", extract_job_node)
builder.add_node("retrieve_evidence", retrieve_evidence_node)
builder.add_node("select_and_tailor", select_and_tailor_node)
builder.add_node("coverage_critic", coverage_critic_node)
builder.add_node("fabrication_critic", fabrication_critic_node)
builder.add_node("decision", decision_node)
builder.add_node("export", export_node)

builder.add_edge(START, "parse_profile")
builder.add_edge(START, "extract_job")
builder.add_edge(["parse_profile", "extract_job"], "retrieve_evidence")
builder.add_edge("retrieve_evidence", "select_and_tailor")
builder.add_edge("select_and_tailor", "coverage_critic")
builder.add_edge("select_and_tailor", "fabrication_critic")
builder.add_edge(["coverage_critic", "fabrication_critic"], "decision")
builder.add_conditional_edges(
    "decision",
    decision_router,
    {"revise": "select_and_tailor", "export": "export", "stop": END},
)
builder.add_edge("export", END)

graph = builder.compile()


if __name__ == "__main__":
    from graph_state import build_initial_state

    initial_state = build_initial_state(
        posting_source="https://www.linkedin.com/jobs/view/4446392638/",
        fetch_first=True,
        profile_instruction=(
            "github link is https://github.com/georgebassem111, and read the CV "
            "file at data/George_Bassem_Backend_AI_Engineer_CV.pdf."
        ),
    )
    final_state = graph.invoke(initial_state.model_dump())
    print("\n[ProfileFit] Run complete")
    print(f"  Decision: {final_state['decision'].value}")
    print(f"  Match score: {final_state.get('match_score')}/100")
    print(f"  Security flagged: {final_state.get('security_flagged')}")
    if final_state.get('security_flagged'):
        flag_reason = None
        job_extraction = final_state.get('job_extraction')
        if job_extraction is not None:
            flag_reason = getattr(job_extraction, 'flag_reason', None) or (
                job_extraction.get('flag_reason') if isinstance(job_extraction, dict) else None
            )
        print(f"  Flag reason: {flag_reason}")
    print(f"  Summary: {final_state.get('decision_summary') or 'n/a'}")
    print(f"  Revisions: {final_state['revision_count']}/{MAX_REVISIONS}")
    print(f"  Critic cycles: {final_state['critic_cycle_count']}")
    print(f"  DOCX: {final_state.get('docx_path') or 'not exported'}")
    print(f"  PDF: {final_state.get('pdf_path') or 'not exported'}")