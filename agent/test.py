"""
Full-pipeline smoke test: extract -> tailor -> coverage critic ->
fabrication critic -> export. Run this after any schema or node change to
catch integration breaks fast, before wiring anything into LangGraph.

Usage:
    python agent\\smoke_test.py --text "..."
    python agent\\smoke_test.py --file path/to/posting.pdf
    python agent\\smoke_test.py                      # uses built-in demo posting
"""

import argparse
import json
import sys

from requirement_extractor import extract_requirements
from selector_tailor import generate_tailored_cv, job_requirements_from_extraction
from coverage_critic import run_coverage_critic
from fabrication_critic import run_fabrication_critic
from exporter import export_cv
from rag import rag_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline smoke test.")
    parser.add_argument("--text", help="Raw job posting text.")
    parser.add_argument("--file", help="Path to a PDF containing the job posting.")
    parser.add_argument("--url", help="URL of the job posting to fetch.")
    args = parser.parse_args()

    if args.url:
        posting_source, fetch_first = f"job posting is: {args.url}", True
    elif args.file:
        posting_source, fetch_first = f"pdf path is {args.file}", True
    elif args.text:
        posting_source, fetch_first = args.text, False
    else:
        posting_source, fetch_first = (
            """
            Data Analyst Intern — Acme Corp
            Requirements:
            - Bachelor's degree in progress (CS, Statistics, or related field)
            - Required: proficiency in SQL and Python (pandas)
            - Preferred: experience with Tableau or Power BI
            - Nice to have: exposure to A/B testing concepts
            """,
            False,
        )

    print("=" * 60)
    print("STEP 1: Requirement Extraction")
    print("=" * 60)
    extraction = extract_requirements(posting_source, fetch_first=fetch_first, save=False)
    if extraction.flagged_for_review:
        print(f"[FLAGGED] {extraction.flag_reason}")
        sys.exit(1)
    job_reqs = job_requirements_from_extraction(extraction)
    print(f"Job: {job_reqs.job_title}")
    print(f"Required: {job_reqs.required_skills}")
    print(f"Preferred: {job_reqs.preferred_skills}\n")

    print("=" * 60)
    print("STEP 2: Retrieval (master profile only)")
    print("=" * 60)
    rag_result = json.loads(
        rag_search(
            f"Candidate profile experiences, projects, education, and skills "
            f"relevant to {job_reqs.job_title}. Required: "
            f"{', '.join(job_reqs.required_skills)}.",
            top_k=10,
            source_filter="master_profiles",
        )
    )
    rag_sources = rag_result.get("sources", [])
    print(f"Retrieved {len(rag_sources)} chunks.\n")
    if not rag_sources:
        print("WARNING: no master profile chunks found — later steps will be weak/empty.\n")

    print("=" * 60)
    print("STEP 3: Selector & Tailor")
    print("=" * 60)
    tailored_cv = generate_tailored_cv(job_reqs, rag_sources=rag_sources)
    print(f"Sections produced: {len(tailored_cv.sections)}")
    by_section = {}
    for item in tailored_cv.sections:
        by_section.setdefault(item.section.value, 0)
        by_section[item.section.value] += 1
    print(f"Per-section counts: {by_section}")
    print(f"Missing skills: {tailored_cv.missing_skills}\n")

    print("=" * 60)
    print("STEP 4a: Coverage Critic")
    print("=" * 60)
    coverage_result = run_coverage_critic(tailored_cv, job_reqs)
    print(f"Status: {coverage_result.status.value}")
    print(f"Evidence: {coverage_result.evidence}\n")

    print("=" * 60)
    print("STEP 4b: Fabrication Critic")
    print("=" * 60)
    fabrication_result = run_fabrication_critic(tailored_cv, rag_sources=rag_sources)
    print(f"Status: {fabrication_result.status.value}")
    print(f"Evidence: {fabrication_result.evidence}")
    for finding in fabrication_result.findings:
        if finding.verdict.value == "FAIL":
            print(f"  FAIL [{finding.original_id}] {finding.title}: {finding.unsupported_claims}")
    print()

    print("=" * 60)
    print("STEP 5: Decision")
    print("=" * 60)
    both_pass = coverage_result.status.value == "PASS" and fabrication_result.status.value == "PASS"
    print(f"Coverage: {coverage_result.status.value} | Fabrication: {fabrication_result.status.value}")
    print(f"Decision: {'APPROVE — export' if both_pass else 'REJECT — needs revision, not exported'}\n")

    if not both_pass:
        print("Skipping export since a critic failed. (In the real graph, this would")
        print("trigger a revision loop back to the Tailor Agent instead of stopping.)")
        sys.exit(0)

    print("=" * 60)
    print("STEP 6: Export")
    print("=" * 60)
    result = export_cv(tailored_cv, contact_items=[], output_path="output/smoke_test_cv.docx")
    print(f"docx: {result['docx_path']}")
    print(f"pdf:  {result['pdf_path']}")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()