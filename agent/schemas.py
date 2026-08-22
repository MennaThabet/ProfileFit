"""
Shared Pydantic schemas for the CV-tailoring pipeline.

Every node (profile_parser, requirement_extractor, selector_tailor,
coverage_critic, fabrication_critic, ...) imports ProfileItem / ItemType /
TailoredCV from here instead of redefining them locally. 
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# Master profile

class ItemType(str, Enum):
    EXPERIENCE = "EXPERIENCE"
    EDUCATION = "EDUCATION"
    PROJECT = "PROJECT"
    CERTIFICATION = "CERTIFICATION"
    SKILL = "SKILL"
    ACHIEVEMENT = "ACHIEVEMENT"
    CONTACT = "CONTACT"


# Section types that the Tailor Agent is allowed to select from. CONTACT is
# deliberately excluded here — contact info (name, email, phone, links) must
# never be selected, rephrased, or emitted by the Tailor Agent
TAILORABLE_TYPES = [
    ItemType.EXPERIENCE,
    ItemType.EDUCATION,
    ItemType.PROJECT,
    ItemType.CERTIFICATION,
    ItemType.SKILL,
    ItemType.ACHIEVEMENT,
]


class ProfileItem(BaseModel):
    id: str
    type: ItemType
    title: str
    description: str
    skills_used: list[str] = Field(default_factory=list)
    dates: Optional[str] = None
    quantified_results: list[str] = Field(default_factory=list)


class ProfileExtraction(BaseModel):
    items: list[ProfileItem]


# Job requirements (simplified shape consumed by the Selector/Tailor prompt)

# NOTE: this is intentionally distinct from requirement_extractor.py's JobPostingExtraction/JobRequirement.


class JobRequirements(BaseModel):
    job_title: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_years_experience: int


# Selector / Tailor output

# TailoredCV is fully structured — there is no free-form `cv: str` field.
class TailoredBullet(BaseModel):
    text: str = Field(description="A single tailored, job-language bullet point.")


class TailoredSectionItem(BaseModel):
    original_id: str = Field(
        description=(
            "The ProfileItem.id this entry was tailored from. Must match an "
            "ID actually present in the supplied candidate context (either "
            "an explicit ProfileItem.id, or an [ID: ...] tag found in a "
            "retrieved RAG chunk's text) — never invented. Required for "
            "traceability — the Fabrication Critic verifies every entry "
            "against this id."
        )
    )
    section: ItemType = Field(
        description="Which master-profile section this item belongs to. "
        "Must be one of TAILORABLE_TYPES — never CONTACT."
    )
    title: str
    dates: Optional[str] = None
    relevance_score: int = Field(
        ge=0, le=100,
        description="0-100 score against the job requirements. Used to rank "
        "items within their section and across sections when trimming for length.",
    )
    tailored_bullets: list[TailoredBullet] = Field(
        default_factory=list,
        description="Action-oriented bullets rephrased to the job's language. "
        "Must not introduce facts, tools, or metrics absent from the source item.",
    )


# Per-section minimum coverage: at least this many items per section must be represented in the final CV
MIN_ITEMS_PER_SECTION = 2


class TailoredCV(BaseModel):
    professional_summary: str = Field(
        description="2-4 sentence summary tailored to the job. No contact "
        "information, names, emails, or phone numbers — summary content only."
    )
    sections: list[TailoredSectionItem] = Field(
        default_factory=list,
        description=(
            "The selected, tailored items across all master-profile sections "
            "(never CONTACT). Must satisfy MIN_ITEMS_PER_SECTION coverage "
            "whenever the master profile has that many items available in a "
            "given section."
        ),
    )
    omitted_experience_ids: list[str] = Field(
        default_factory=list,
        description="IDs of profile items that were provided but NOT selected.",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Important skills from the job description not found in "
        "the candidate context. Never rendered inside a bullet or the "
        "summary — this field only.",
    )