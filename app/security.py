"""Security: prompt-injection defenses.

Three layers:
  1. an input length limit (max_input_chars) -> 400;
  2. injection-pattern detection (case-insensitive) -> 400 + log to suspicious_requests.log;
  3. an output filter AFTER streaming: system-prompt fragments in the answer -> output_filtered + log.

The log files (`*.log`) are gitignored (they may contain raw user input).
"""
from __future__ import annotations
import logging
import re

from fastapi import HTTPException
from app.config import settings

# Common prompt-injection marker patterns
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)",
    r"forget\s+(?:everything|all|your\s+instructions)",
    r"you\s+are\s+now\b",
    r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
    r"system\s*:",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"</s>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Distinctive fragments of OUR system prompt — their presence in an answer signals a leak
_SYSTEM_FRAGMENTS = [
    "you are a q&a assistant for the fastapi documentation",
    "use only the provided context snippets",
    "do not use outside knowledge",
]


def _file_logger(name: str, path: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.FileHandler(path, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.WARNING)
        lg.propagate = False
    return lg


_req_log = _file_logger("suspicious_requests", "suspicious_requests.log")
_resp_log = _file_logger("suspicious_responses", "suspicious_responses.log")


def check_input(message: str, api_key: str = "") -> None:
    """Validate input BEFORE streaming. Raises 400 on length overflow or a matched injection pattern."""
    if len(message) > settings.max_input_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long: {len(message)} chars (max {settings.max_input_chars})",
        )
    for rx in _INJECTION_RE:
        if rx.search(message):
            _req_log.warning("BLOCKED key=%s pattern=%r input=%r", api_key, rx.pattern, message[:300])
            raise HTTPException(status_code=400, detail="Input rejected: suspicious content detected")


def filter_output(text: str, api_key: str = "") -> bool:
    """Return True if the answer contains system-prompt fragments (a leak) and log it."""
    low = text.lower()
    for frag in _SYSTEM_FRAGMENTS:
        if frag in low:
            _resp_log.warning("LEAK key=%s fragment=%r", api_key, frag)
            return True
    return False