import json
import os
import re
from typing import Any
from fastapi import File
from litellm import completion
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv

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


def get_model(model: str | None) -> str:
    if os.environ.get('GEMINI_API_KEY'):
        print('gemini found')
        return "gemini/gemini-2.5-flash"
    
    # elif model:
    #     return model
    # elif os.environ.get("MISTRAL_API_KEY"):
    #     return "mistral/mistral-medium-latest"
        # return "huggingface/HuggingFaceTB/SmolLM3-3B"


def litellm_chat(messages: list[dict[str, str]], model: str | None = None, schema: BaseModel = None) -> str:
    
    # model_str = model if model else ''
    response = completion(
        model=get_model(model),
        messages=messages,
        temperature=0.4,
        response_format=schema,
        reasoning_effort="low"
        # api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    try:
        return response["choices"][0]["message"]["content"]
    except Exception:  # pragma: no cover
        return str(response)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_jsonish(text: str) -> dict[str, Any]:
    """
    Best-effort JSON object parsing for LLM outputs.
    Always returns a dict (either parsed JSON or a wrapper with the raw output).
    """
    candidate = text.strip()

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