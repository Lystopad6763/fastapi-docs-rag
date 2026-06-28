"""Validate the LLM judge against a human-labeled set.

Validating a judge against human labels (a periodic correlation check) is essential — we already
saw the judge can err (the pwdlib case), so we don't take it on faith. This runs the *refusal*
judge rubric (the same one used in scoring, via the 3-vote ensemble) on a fixed set of
hand-labeled (question, answer, gold) pairs — including the pwdlib trap, a false-premise
abstention, and a case where the bot wrongly answered an off-topic question. It measures the
JUDGE, not the bot (the answers are fixed).

Run:  python eval/safety/validate_judge.py
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE / "evaluators"))

from judge import judge_vote                             # noqa: E402
from refusal import JUDGE_SYSTEM                          # noqa: E402  (reuse the scoring rubric)

DATA = HERE / "datasets" / "judge_calibration.jsonl"
OUT = HERE / "results" / "judge_calibration.json"


def main() -> None:
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, agree = [], 0
    for r in rows:
        user = f"QUESTION:\n{r['question']}\n\nANSWER:\n{r['answer']}\n\nClassify."
        label, votes, _ = judge_vote(JUDGE_SYSTEM, user, lambda v: v.get("label", "?"))
        ok = label == r["gold"]
        agree += ok
        out.append({**r, "judge": label, "votes": votes, "match": ok})
        print(f"  {'OK ' if ok else '!! '}{r['id']} gold={r['gold']:8} judge={label:8} | {r['note'][:48]}")
    acc = round(agree / len(rows), 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"judge_human_agreement": acc, "n": len(rows), "results": out},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJudge <-> human agreement: {acc}  ({agree}/{len(rows)})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
