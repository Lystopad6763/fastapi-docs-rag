"""CI gate for the safety eval — eval as CI/CD.

Fails the build if the SHIPPING configuration (guardrails on) regresses below any class gate,
or if the overall verdict is no longer SHIP. Runs OFFLINE against the committed
results/summary.json, so it needs no API keys — the live eval (which does) runs locally /
nightly and refreshes summary.json. Regenerate with: python eval/safety/run_eval.py

    pytest eval/safety/test_safety_gates.py -v
"""
import json
import pathlib

import pytest

SUMMARY = pathlib.Path(__file__).resolve().parent / "results" / "summary.json"


def _summary() -> dict:
    assert SUMMARY.exists(), "results/summary.json missing — run: python eval/safety/run_eval.py"
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cls", ["refusal", "injection", "pii", "faithfulness"])
def test_shipping_class_within_gate(cls):
    after = _summary()["after_guardrail"]
    assert cls in after, f"{cls} missing from after_guardrail (run run_eval.py)"
    assert after[cls]["pass"], f"{cls} regressed below its gate: {after[cls]['detail']}"


def test_overall_shipping_verdict():
    assert _summary()["verdict_after"] == "SHIP", "shipping verdict is no longer SHIP"
