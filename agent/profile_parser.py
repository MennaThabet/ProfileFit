import os
import uuid
import requests
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

# 1. Define the Schema ->>>> will be moved to schema file later
class ItemType(str, Enum):
    EXPERIENCE = "EXPERIENCE"
    EDUCATION = "EDUCATION"
    PROJECT = "PROJECT"
    CERTIFICATION = "CERTIFICATION"

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

# TOOLS
def extract_pdf_text(file_path: str) -> str:
    """Extracts all text from a local PDF file given its file path."""
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_url_text(url: str) -> str:
    """Scrapes and extracts visible text from a given web URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    return " ".join(soup.stripped_strings)


SYS_PROMPT = """
You are an expert profile parser. Read the extracted text from resumes, LinkedIn
profiles, or GitHub pages. Extract all distinct roles, projects, and educational
experiences into a structured list. Extract metrics for 'quantified_results' where possible.
"""

MODEL_NAME = "gemini-3.6-flash"


def _gather_raw_text(client: genai.Client, instruction: str) -> str:
    """
    Step 1: tool-enabled call, NO response_schema.
    Keeping this step schema-free lets the model freely call tools and return
    the raw extracted text, which we then structure in step 2
    """
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYS_PROMPT,
            tools=[extract_url_text, extract_pdf_text],
            temperature=0.1,
        ),
    )
    response = chat.send_message(instruction)
    return response.text


def _structure_text(client: genai.Client, raw_text: str, feedback: Optional[str] = None) -> ProfileExtraction:
    """
    Step 2: schema-enforced call, NO tools.
    Takes the raw text gathered in step 1 and forces it into the ProfileExtraction
    schema. If `feedback` is provided (a prior validation error), it's included
    so the model can self-correct on retry
    """
    prompt = f"Structure the following profile text into ProfileItems:\n\n{raw_text}"
    if feedback:
        prompt += (
            f"\n\nYour previous attempt failed schema validation with this error:\n{feedback}\n"
            f"Fix the output so it strictly matches the required schema."
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYS_PROMPT,
            response_mime_type="application/json",
            response_schema=ProfileExtraction,
            temperature=0.1,
        ),
    )
    return response.parsed


def parse_profile_data(instruction: str, max_retries: int = 3) -> list[ProfileItem]:
    """
    Bounded repair loop if structured parsing fails validation,
    feed the error back to the model and retry, up to max_retries times
    """
    client = genai.Client()

    raw_text = _gather_raw_text(client, instruction)

    last_error: Optional[str] = None
    for attempt in range(max_retries):
        try:
            extraction = _structure_text(client, raw_text, feedback=last_error)
            items = extraction.items

            # Generate stable, unique IDs ourselves rather than trusting the model
            for item in items:
                item.id = f"item_{uuid.uuid4().hex[:8]}"

            return items
        except Exception as e:
            last_error = str(e)
            continue

    raise ValueError(f"Failed to parse profile after {max_retries} attempts: {last_error}")


# Example Usage:
if __name__ == "__main__":
    instruction = f"github link is https://github.com/georgebassem111"
    try:
        extracted_items = parse_profile_data(instruction)
        for item in extracted_items:
            print(item.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error parsing profile: {e}")
