import json
import os
import re
from typing import Any
from fastapi import File
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv('.env.local')

from huggingface_hub import login

login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

def extract_pdf_text(fileObj: File) -> str:
    reader = PdfReader(fileObj)
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    return "<next-page>".join(parts).strip()


def init_huggingface_llm():
    """Initialize HuggingFace LLM with token from Colab secrets"""
    
    # Using Mistral-7B for better instruction following
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=4012,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",
    )

    # Wrap the LLM with ChatHuggingFace to enable tool calling
    chat_model = ChatHuggingFace(llm=llm)

    print("✓ HuggingFace ChatModel initialized")
    return chat_model

def parse_jsonish(text: str) -> dict[str, Any]:
    """
    Best-effort JSON object parsing for LLM outputs.
    Always returns a dict (either parsed JSON or a wrapper with the raw output).
    """
    candidate = text.strip()
    _JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    fence_match = _JSON_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            pass

    return {"llm_output": text}