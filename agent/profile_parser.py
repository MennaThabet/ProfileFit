import os
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


def parse_profile_data(instruction: str) -> list[ProfileItem]:
    client = genai.Client()
    
    sys_prompt = """
    You are an expert profile parser. Read the extracted text from resumes, LinkedIn 
    profiles, or GitHub pages. Extract all distinct roles, projects, and educational 
    experiences into a structured list. Extract metrics for 'quantified_results' where possible.
    """
    
    chat = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,  
            tools=[extract_url_text, extract_pdf_text],
            response_mime_type="application/json",
            response_schema=ProfileExtraction,
            temperature=0.1, 
        )
    )
    
    response = chat.send_message(instruction)
    return response.parsed.items


# Example Usage:
if __name__ == "__main__":
    # text = extract_pdf_text("")
    instruction = f"github link is https://github.com/georgebassem111"
    try:
        extracted_items = parse_profile_data(instruction)
        for item in extracted_items:
            print(item.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error parsing profile: {e}")