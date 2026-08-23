"""
Shared graph state for the CV-tailoring pipeline.

This is the ONE object every LangGraph node reads from and writes back to.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from schemas import JobRequirements, ProfileItem, TailoredCV
from requirement_extractor import JobPostingExtraction
from coverage_critic import CoverageCriticResult
from fabrication_critic import FabricationCriticResult


class DecisionStatus(str, Enum):
    PENDING = "PENDING"  # critics haven't both reported in yet
    APPROVED = "APPROVED"  # both critics PASS (and human approved, once HITL exists) -> export
    NEEDS_REVISION = "NEEDS_REVISION"  # at least one critic FAILed -> back to Tailor
    REJECTED_MAX_REVISIONS = "REJECTED_MAX_REVISIONS"  # exhausted the bounded retry budget
    REJECTED_FLAGGED_POSTING = "REJECTED_FLAGGED_POSTING"  # posting failed injection/security check


# Bounded revision loop.
MAX_REVISIONS = 3


class GraphState(BaseModel):
    #  Run input (set once, at the start of a run) 
    posting_source: str = Field(
        description="Raw job posting text, or an instruction naming a file/URL to "
        "fetch (see requirement_extractor.extract_requirements's posting_source arg)."
    )
    fetch_first: bool = Field(
        default=False,
        description="Whether posting_source needs fetching via extract_pdf_text/"
        "extract_url_text before it's usable text (see extract_requirements).",
    )
    profile_instruction: Optional[str] = Field(
        default=None,
        description="Instruction for profile_parser.parse_profile_data, e.g. a "
        "GitHub link or CV file path. If None, profile_items must already be "
        "populated by the caller (e.g. loaded from a prior parse in the session).",
    )

    #  Requirement Extractor node output 
    job_extraction: Optional[JobPostingExtraction] = Field(
        default=None,
        description="Full structured extraction, including flagged_for_review/"
        "flag_reason. Kept alongside job_reqs (below) since it carries the "
        "security-flag fields that JobRequirements does not.",
    )
    job_reqs: Optional[JobRequirements] = Field(
        default=None,
        description="Simplified shape consumed by Selector/Tailor's prompt "
        "(selector_tailor.job_requirements_from_extraction(job_extraction)).",
    )

    #  Profile Parser node output 
    profile_items: list[ProfileItem] = Field(
        default_factory=list,
        description="ALL parsed items, including CONTACT. Selector/Tailor and "
        "Fabrication Critic filter out CONTACT themselves when needed "
        "(see selector_tailor.filter_tailorable_items) — this field is the "
        "single source of truth, not pre-filtered.",
    )
    contact_items: list[ProfileItem] = Field(
        default_factory=list,
        description="Convenience cache of profile_parser.get_contact_items(profile_items), "
        "set by the same node that populates profile_items. Used only by the "
        "Exporter node at the very end.",
    )
    rag_sources: list[dict] = Field(
        default_factory=list,
        description="Candidate evidence chunks retrieved from the master-profile "
        "knowledge base when profile_items were not supplied directly.",
    )

    #  Selector/Tailor node output (overwritten on each revision attempt) 
    tailored_cv: Optional[TailoredCV] = None
    revision_count: int = Field(
        default=0,
        description="How many times Selector/Tailor has been asked to revise "
        "based on critic feedback THIS run. Incremented by the Decision node "
        "when it routes back for revision. Compared against MAX_REVISIONS.",
    )
    critic_cycle_count: int = Field(
        default=0,
        description="How many complete coverage + fabrication critic cycles "
        "have run during this graph execution.",
    )
    revision_feedback: Optional[str] = Field(
        default=None,
        description="Combined, human-readable feedback from the last critic run "
        "(coverage + fabrication findings), passed back into the next "
        "generate_tailored_cv() call's prompt so the retry is informed, not blind.",
    )

    #  Critic node outputs (run in parallel, both populated before Decision runs) 
    coverage_result: Optional[CoverageCriticResult] = None
    fabrication_result: Optional[FabricationCriticResult] = None

    #  Decision node output 
    decision: DecisionStatus = DecisionStatus.PENDING
    human_approved: Optional[bool] = Field(
        default=None,
        description="Set by the human-in-the-loop review step once it exists. "
        "None means 'not yet reviewed'. The Decision node should not move to "
        "APPROVED purely on critic PASS status alone once HITL is wired in — "
        "both critic PASS AND human_approved=True should be required.",
    )

    # Exporter node output 
    docx_path: Optional[str] = None
    pdf_path: Optional[str] = None

    # Security / observability 
    security_flagged: bool = Field(
        default=False,
        description="Mirrors job_extraction.flagged_for_review once set, for "
        "quick access without unwrapping job_extraction. A flagged posting "
        "should short-circuit the graph to DecisionStatus.REJECTED_FLAGGED_POSTING "
        "rather than proceeding to tailoring at all.",
    )

    class Config:
        arbitrary_types_allowed = True


def build_initial_state(
    posting_source: str,
    fetch_first: bool = False,
    profile_instruction: Optional[str] = None,
    profile_items: Optional[list[ProfileItem]] = None,
    contact_items: Optional[list[ProfileItem]] = None,
) -> GraphState:
    """
    Convenience constructor for the state a run starts with. Exists mainly so
    graph.py (and any test harness) has one obvious place to build a valid
    starting GraphState rather than each caller hand-assembling one.
    """
    return GraphState(
        posting_source=posting_source,
        fetch_first=fetch_first,
        profile_instruction=profile_instruction,
        profile_items=profile_items or [],
        contact_items=contact_items or [],
    )