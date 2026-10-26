"""Scores the rule engine against the labeled test set."""
import json
import sys
from collections import defaultdict
from pathlib import Path

from app.rules.label import LabelData
from app.rules.run_checks import overall_verdict, run_all_checks

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "testset" / "cases.json"


def run_cases(cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        findings = run_all_checks(LabelData(**case["label"]))
        results.append({
            "id": case["id"],
            "violation_type": case["violation_type"],
            "expected": case["expected_overall"],
            "predicted": overall_verdict(findings).value,
        })
    return results


def summarize(results: list[dict]) -> dict:
    by_type = defaultdict(lambda: {"n": 0, "correct": 0})
    false_passes, false_alarms = [], []

    for r in results:
        stats = by_type[r["violation_type"]]
        stats["n"] += 1
        stats["correct"] += int(r["expected"] == r["predicted"])
        if r["expected"] != "pass" and r["predicted"] == "pass":
            false_passes.append(r["id"])
        if r["expected"] == "pass" and r["predicted"] != "pass":
            false_alarms.append(r["id"])

    total = len(results)
    correct = sum(s["correct"] for s in by_type.values())
    must_catch = sum(1 for r in results if r["expected"] != "pass")
    clean = sum(1 for r in results if r["expected"] == "pass")

    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "by_type": dict(by_type),
        "false_passes": false_passes,
        "false_pass_rate": len(false_passes) / must_catch if must_catch else 0.0,
        "false_alarms": false_alarms,
        "false_alarm_rate": len(false_alarms) / clean if clean else 0.0,
    }


def main(path: Path = DEFAULT_PATH) -> int:
    cases = json.loads(path.read_text())
    summary = summarize(run_cases(cases))

    print(f"Cases: {summary['total']}   Accuracy: {summary['accuracy']:.1%}")
    print(f"False-pass rate:  {summary['false_pass_rate']:.1%}  ({len(summary['false_passes'])} missed violations)")
    print(f"False-alarm rate: {summary['false_alarm_rate']:.1%}  ({len(summary['false_alarms'])} clean labels flagged)")
    print("\nBy violation type:")
    for vtype, s in sorted(summary["by_type"].items()):
        print(f"  {vtype:30s} {s['correct']}/{s['n']}")
    if summary["false_passes"]:
        print("\nMissed:", ", ".join(summary["false_passes"]))

    return 1 if summary["false_passes"] else 0


if __name__ == "__main__":
    sys.exit(main())