import base64
import os
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

from .utils import safe_json_extract


def _image_to_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = "image/png"
    return f"data:{mime};base64,{b64}"


@dataclass
class ModelResponse:
    raw_text: str
    parsed: Optional[Dict[str, Any]]


def call_openai(
    model: str,
    prompt: str,
    image_path: Path,
    timeout: Optional[float] = 60,
    reasoning_effort: Optional[str] = None,
) -> ModelResponse:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    data_url = _image_to_data_url(image_path)

    # If reasoning effort is requested, use the Responses API (supports reasoning)
    if reasoning_effort:
        try:
            _kwargs = {}
            if timeout:
                _kwargs["timeout"] = timeout
            resp = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                response_format={"type": "json_object"},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                **_kwargs,
            )
            txt = getattr(resp, "output_text", None) or ""
            return ModelResponse(raw_text=txt, parsed=safe_json_extract(txt))
        except Exception:
            # Fallback: try without response_format if provider rejects it
            _kwargs = {}
            if timeout:
                _kwargs["timeout"] = timeout
            resp = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                **_kwargs,
            )
            txt = getattr(resp, "output_text", None) or ""
            return ModelResponse(raw_text=txt, parsed=safe_json_extract(txt))

    # Default path: Chat Completions API
    _kwargs = {}
    if timeout:
        _kwargs["timeout"] = timeout
    completion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        **_kwargs,
    )
    txt = completion.choices[0].message.content or ""
    return ModelResponse(raw_text=txt, parsed=safe_json_extract(txt))


def call_openrouter(model: str, prompt: str, image_path: Path, timeout: Optional[float] = 60) -> ModelResponse:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    data_url = _image_to_data_url(image_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Optional ranking headers
    referer = os.environ.get("OPENROUTER_REFERER")
    title = os.environ.get("OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    want_response_format = not os.environ.get("OPENROUTER_NO_RESPONSE_FORMAT")
    body = {
        "model": model,
        **({"response_format": {"type": "json_object"}} if want_response_format else {}),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    url = f"{base_url.rstrip('/')}/chat/completions"

    def _post(b):
        return requests.post(url, headers=headers, json=b, timeout=timeout)

    # Attempt with optional response_format once; if 400, drop it.
    resp = _post(body)
    if resp.status_code == 400 and want_response_format:
        body.pop("response_format", None)
        resp = _post(body)

    # Basic retry for rate limits or transient errors
    max_retries = int(os.environ.get("OPENROUTER_MAX_RETRIES", "2"))
    base_sleep = float(os.environ.get("OPENROUTER_RETRY_BASE", "2.0"))
    attempt = 0
    while resp.status_code in (429, 502, 503, 504) and attempt < max_retries:
        attempt += 1
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                delay = float(retry_after)
            except Exception:
                delay = base_sleep * (2 ** (attempt - 1))
        else:
            delay = base_sleep * (2 ** (attempt - 1))
        time.sleep(delay)
        resp = _post(body)
    resp.raise_for_status()
    data = resp.json()
    txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return ModelResponse(raw_text=txt, parsed=safe_json_extract(txt))
