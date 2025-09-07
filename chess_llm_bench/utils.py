import json
import os
import re
from typing import Any, Dict, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def only_fen64(full_fen: str) -> str:
    return full_fen.split(" ")[0]


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse strict JSON; if that fails, attempt to extract the first JSON object."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object in the text
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    snippet = m.group(0)
    try:
        return json.loads(snippet)
    except Exception:
        return None


def parse_eval_value_to_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float):
            return int(round(value))
        if isinstance(value, str):
            # Accept numbers embedded in strings
            m = re.search(r"-?\d+", value)
            if m:
                return int(m.group(0))
            return None
    except Exception:
        return None
    return None


def safe_filename(text: str, max_len: int = 120) -> str:
    """Return a filesystem-safe filename fragment.

    - Replace path separators and illegal chars with '-'
    - Allow only [A-Za-z0-9._-]
    - Collapse multiple '-'
    - Trim to max_len
    """
    # Replace common path separators first
    t = text.replace(os.sep, "-").replace("/", "-")
    # Replace Windows-forbidden characters
    t = re.sub(r"[\\/:*?\"<>|]", "-", t)
    # Keep only safe chars
    t = re.sub(r"[^A-Za-z0-9._-]", "-", t)
    # Collapse dashes
    t = re.sub(r"-+", "-", t).strip("-._")
    if len(t) > max_len:
        t = t[:max_len].rstrip("-._")
    return t or "run"
