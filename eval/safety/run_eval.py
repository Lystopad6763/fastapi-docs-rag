"""Aggregate the per-class safety results into ONE production-readiness verdict.

Reads results/*.json (the unguarded "before" runs) and the guardrail "after" runs, applies
the gates in thresholds.yaml, and writes results/summary.json + prints the ship/not-ship
matrix that REPORT.md cites.

First run each class (each writes results/<class>.json):
    python eval/safety/evaluators/refusal.py
    python eval/safety/evaluators/injection.py
    python eval/safety/evaluators/pii.py
    python eval/safety/evaluators/faithfulness.py
The "after" (guardrail on) runs are produced against a server started with
GUARDRAILS_ENABLED=true, saved to results/*_after.json / *_guarded.json. Then:
    python eval/safety/run_eval.py
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gates import load_thresholds                       # noqa: E402

RES = HERE / "results"
TH = load_thresholds() or {}


def _load(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("metrics")


def _refusal(m: dict) -> tuple[bool, str]:
    g = TH["refusal"]
    ok = m["correct_refusal_rate"] >= g["correct_refusal_min"] and m["over_refusal_rate"] <= g["over_refusal_max"]
    return ok, f"correct={m['correct_refusal_rate']} over={m['over_refusal_rate']}"


def _injection(m: dict) -> tuple[bool, str]:
    g = TH["injection"]
    return m["asr_overall"] <= g["asr_max"], f"ASR={m['asr_overall']} (by_family={m['asr_by_family']})"


def _pii(m: dict) -> tuple[bool, str]:
    g = TH["pii"]
    return m["leak_rate"] <= g["leak_rate_max"], f"leak_rate={m['leak_rate']} ({m['leaks']}/{m['n']})"


def _faith(m: dict) -> tuple[bool, str]:
    g = TH["faithfulness"]
    ok = ((m["faithfulness_mean"] or 0) >= g["faithfulness_min"]
          and (m["answer_relevancy_mean"] or 0) >= g["answer_relevancy_min"]
          and m["fabrication_rate"] <= g["fabrication_rate_max"])
    return ok, f"faith={m['faithfulness_mean']} rel={m['answer_relevancy_mean']} fabr={m['fabrication_rate']}"


CLASSES = [
    ("refusal",      "refusal.json",      "refusal_guarded.json",  _refusal),
    ("injection",    "injection.json",    "injection_after.json",  _injection),
    ("pii",          "pii.json",          "pii_after.json",        _pii),
    ("faithfulness", "faithfulness.json", "faithfulness_guarded.json", _faith),
]


def _phase(file: str, fn) -> dict | None:
    m = _load(file)
    if m is None:
        return None
    ok, detail = fn(m)
    return {"pass": ok, "detail": detail, "metrics": m}


def main() -> None:
    before, after = {}, {}
    for name, bfile, afile, fn in CLASSES:
        b = _phase(bfile, fn)
        a = _phase(afile, fn)
        if b:
            before[name] = b
        if a:
            after[name] = a

    def verdict(phase: dict) -> str:
        if len(phase) < len(CLASSES):
            return "INCOMPLETE"
        return "SHIP" if all(v["pass"] for v in phase.values()) else "NOT SHIP"

    summary = {
        "thresholds": TH,
        "before_guardrail": before,
        "after_guardrail": after,
        "verdict_before": verdict(before),
        "verdict_after": verdict(after),
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def show(title: str, phase: dict):
        print(f"\n=== {title} ===")
        for name, _, _, _ in CLASSES:
            v = phase.get(name)
            if not v:
                print(f"  {name:14} (missing)")
                continue
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {name:14} | {v['detail']}")

    print("############ SAFETY EVAL — PRODUCTION READINESS ############")
    show("BEFORE guardrail (current default)", before)
    print(f"\n  VERDICT (before): {summary['verdict_before']}")
    show("AFTER guardrail (guardrails_enabled=true)", after)
    print(f"\n  VERDICT (after):  {summary['verdict_after']}")
    print(f"\nsaved -> {(RES / 'summary.json')}")


if __name__ == "__main__":
    main()