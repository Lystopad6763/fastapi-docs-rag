"""Langfuse v4 tracing.

Enabled ONLY when keys are present (settings.langfuse_enabled). Without keys, `start_trace`
returns a no-op object with the same interface — so main.py needs no branching (tracing is
transparently disabled). Each request -> one trace with spans: embed -> cache_lookup -> retrieval -> llm.
"""
from __future__ import annotations

from app.config import settings

_client = None
_init = False


def _get_client():
    global _client, _init
    if not _init:
        _init = True
        if settings.langfuse_enabled:
            from langfuse import Langfuse
            _client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
    return _client


class _NoopSpan:
    """Stand-in used when Langfuse is disabled — same interface, does nothing."""
    trace_id = None

    def start_observation(self, *a, **k):
        return self

    def update(self, *a, **k):
        return self

    def set_trace_io(self, *a, **k):
        return self

    def end(self, *a, **k):
        return self


def start_trace(name: str, **kwargs):
    """Root span (i.e. the trace). Real when Langfuse is enabled; otherwise a no-op."""
    lf = _get_client()
    if lf is None:
        return _NoopSpan()
    return lf.start_observation(name=name, as_type="span", **kwargs)


def flush() -> None:
    lf = _get_client()
    if lf is not None:
        lf.flush()