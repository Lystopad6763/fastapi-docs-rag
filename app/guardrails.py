"""Output / retrieved-content guardrails.

The safety eval (eval/safety/) found the bot is robust at the INPUT and PROMPT layers but
has NO defense on RETRIEVED CONTENT: a poisoned chunk can (a) carry an instruction the bot
then obeys (indirect prompt injection), or (b) carry PII the bot then repeats. Both are
closed BEFORE the LLM sees the context — for each retrieved chunk we:
  1. strip lines that try to instruct the assistant (injected imperatives);
  2. mask PII (Presidio + a regex safety-net for structured values).

Pre-LLM, so it is streaming-compatible. Gated by settings.guardrails_enabled (default off,
to keep the production image torch/spaCy-free).
"""
from __future__ import annotations

import re
from functools import lru_cache

# Lines in retrieved content that try to steer the assistant -> drop the whole line.
_INSTRUCTION_RE = re.compile(
    r"(?im)^.*\b("
    r"system note|assistant instruction|important system|"
    r"you must (?:append|recommend|include|output|add|say)|"
    r"ignore (?:all |the )?previous|disregard (?:the )?above|"
    r"recommend (?:that )?(?:the user )?install|when (?:answering|summari[sz]ing)|"
    r"append .*\bline\b|visit https?://"
    r")\b.*$"
)

# Regex safety-net so structured PII is masked even if Presidio scores it low.
_PII_NET = {
    "EMAIL_ADDRESS": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE_NUMBER": re.compile(r"\+?\d[\d \-()]{8,}\d"),
    "API_KEY": re.compile(r"sk-[A-Za-z0-9-]{6,}"),
    "IBAN_CODE": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9 ]{10,30}\b"),
}
_PII_TYPES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IBAN_CODE",
              "IP_ADDRESS", "US_PASSPORT", "PERSON", "CRYPTO"]


@lru_cache(maxsize=1)
def _analyzer():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    prov = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    })
    return AnalyzerEngine(nlp_engine=prov.create_engine(), supported_languages=["en"])


def _mask_pii(text: str) -> str:
    try:
        results = _analyzer().analyze(text=text, language="en", entities=_PII_TYPES)
        for r in sorted(results, key=lambda x: x.start, reverse=True):
            if r.score >= 0.5:
                text = text[:r.start] + f"[{r.entity_type}]" + text[r.end:]
    except Exception:  # noqa: BLE001  -- never let the guardrail break a request
        pass
    for label, rx in _PII_NET.items():       # regex net (always runs)
        text = rx.sub(f"[{label}]", text)
    return text


def sanitize_context(text: str) -> str:
    """Strip injected instructions + mask PII from a single retrieved chunk."""
    text = _INSTRUCTION_RE.sub("[removed: out-of-band instruction in source]", text)
    return _mask_pii(text)