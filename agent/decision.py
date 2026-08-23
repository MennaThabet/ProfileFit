"""
Decision Agent.

Synthesizes the Coverage Critic's and Fabrication Critic's findings into:
  1. A short, plain-English summary a human reviewer can read at a glance.
  2. Prioritized, de-duplicated, actionable feedback for the Tailor Agent's
     next revision attempt.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from coverage_critic import CoverageCriticResult
from fabrication_critic import FabricationCriticResult


class DecisionAgentOutput(BaseModel):
    summary_for_human: str = Field(
        description="2-4 sentence plain-English summary of the outcome, for a "
        "human reviewer. States what passed, what failed, and why — no jargon."
    )
    prioritized_feedback: str = Field(
        description="Actionable feedback for the Tailor Agent's next revision, "
        "ordered by severity: fabrication/traceability issues FIRST (these are "
        "honesty problems, non-negotiable), then coverage gaps. De-duplicated "
        "and rewritten as clear instructions, not just concatenated raw fields."
    )


SYS_PROMPT = """
You are a Decision Agent. You do NOT decide whether a CV is approved or rejected —
that decision has ALREADY been made by deterministic code before you were called,
based strictly on whether both critics returned PASS. You are given the critics'
findings as FIXED, ALREADY-TRUE FACTS. Your only job is to:

1. Write a short summary_for_human explaining the outcome in plain English —
   what was checked, what passed, what failed, and the practical reason why.
2. Write prioritized_feedback for the Tailor Agent's next attempt: take the
   critics' raw findings and turn them into clear, ordered, actionable
   instructions.

CRITICAL RULES:
- NEVER soften, dispute, downplay, or second-guess a FAIL verdict, an
  unresolved_id, or an unsupported_claim you are given. Treat every finding
  handed to you as ground truth — your job is to communicate it clearly, not
  to re-evaluate whether it's really a problem.
- Fabrication/traceability issues ALWAYS come first in prioritized_feedback,
  before coverage issues. An honesty problem (invented id, exaggerated claim)
  is more serious than a missing skill mention, and revision should fix it first.
- Do not invent new findings, new missing skills, or new claims not present in
  the critic results you were given. You are synthesizing, not investigating.
- Never include or reference contact information (name, email, phone) in
  either output field, even if it appears somewhere in the critic findings —
  it is out of scope regardless of source.
- Keep prioritized_feedback specific and quote the exact problematic phrases
  where the critics provided them, so the Tailor Agent knows exactly what to
  change — vague guidance like "improve accuracy" is not acceptable.
"""

def run_decision_agent(
    coverage: CoverageCriticResult,
    fabrication: FabricationCriticResult,
    overall_pass: bool,
    revision_count: int,
    max_revisions: int,
) -> DecisionAgentOutput:
    """
    Synthesize both critics' results into a human summary and (if not
    overall_pass) prioritized revision feedback.
    """
    client = genai.Client()

    findings_text = (
        f"OVERALL RESULT (already decided, do not change): "
        f"{'APPROVED' if overall_pass else 'NOT APPROVED'}\n"
        f"Revision attempt: {revision_count} of {max_revisions}\n\n"
        f"--- Coverage Critic ---\n"
        f"Status: {coverage.status.value}\n"
        f"Evidence: {coverage.evidence}\n"
        f"Suggested missing items: {coverage.suggested_missing_items}\n"
        f"Missing experiences: {coverage.missing_experiences}\n"
        f"Weak or unproven claims: {coverage.weak_or_unproven_claims}\n\n"
        f"--- Fabrication Critic ---\n"
        f"Status: {fabrication.status.value}\n"
        f"Evidence: {fabrication.evidence}\n"
        f"Findings:\n"
        + "\n".join(
            f"  [{f.verdict.value}] {f.title} (id={f.original_id})"
            + (f" UNRESOLVED_ID" if f.unresolved_id else "")
            + (f" | unsupported: {f.unsupported_claims}" if f.unsupported_claims else "")
            + (f" | notes: {f.notes}" if f.notes else "")
            for f in fabrication.findings
        )
    )

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        contents=findings_text,
        config=types.GenerateContentConfig(
            system_instruction=SYS_PROMPT,
            response_mime_type="application/json",
            response_schema=DecisionAgentOutput,
            temperature=0.0,
        ),
    )
    return response.parsed


def build_fallback_output(
    coverage: CoverageCriticResult,
    fabrication: FabricationCriticResult,
    overall_pass: bool,
) -> DecisionAgentOutput:
    """
    Deterministic, no-LLM fallback in case the Decision Agent call itself
    fails (network error, etc.), so we can still return a structured output to the graph.
    """
    summary = (
        "CV approved: both coverage and fabrication checks passed."
        if overall_pass
        else "CV not approved: see feedback below for what needs to change."
    )
    feedback = (
        f"Coverage critic ({coverage.status.value}): {coverage.evidence}\n"
        f"Suggested missing items: {coverage.suggested_missing_items}\n"
        f"Missing experiences: {coverage.missing_experiences}\n"
        f"Weak claims: {coverage.weak_or_unproven_claims}\n\n"
        f"Fabrication critic ({fabrication.status.value}): {fabrication.evidence}\n"
        f"Findings: {[f.model_dump() for f in fabrication.findings]}"
    )
    return DecisionAgentOutput(summary_for_human=summary, prioritized_feedback=feedback)