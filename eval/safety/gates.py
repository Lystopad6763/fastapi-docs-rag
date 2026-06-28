"""Load the ship/not-ship thresholds from thresholds.yaml.

Degrades gracefully: if PyYAML is not installed yet, returns None so an evaluator
can still print its raw metrics (just without the pass/fail gate).
"""
from __future__ import annotations

import pathlib

_PATH = pathlib.Path(__file__).resolve().parent / "thresholds.yaml"


def load_thresholds() -> dict | None:
    try:
        import yaml
    except ImportError:
        return None
    return yaml.safe_load(_PATH.read_text(encoding="utf-8"))