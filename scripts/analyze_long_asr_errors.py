"""Describe paired long-test errors without changing the frozen evaluation."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/whisper_long_rescored/utterances.csv"
OUTPUT = ROOT / "data/results/whisper_long_rescored"


def key(row):
    return row["speaker"], row["session"], row["utterance_id"]


def errors(row):
    return sum(int(row[field]) for field in ("substitutions", "deletions", "insertions"))


def main():
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    before = {key(r): r for r in source if r["stage"] == "before_lora"}
    after = {key(r): r for r in source if r["stage"] == "after_lora"}
    if len(before) != len(after) or before.keys() != after.keys():
        raise ValueError("Before and after IDs differ")
    rows = []
    by_speaker = defaultdict(list)
    for clip in sorted(after):
        a, b = after[clip], before[clip]
        item = {"speaker": a["speaker"], "session": a["session"],
                "utterance_id": a["utterance_id"], "duration_seconds": a["duration_seconds"],
                "prompt": a["scoring_target"], "after_hypothesis": a["hypothesis"],
                "reference_words": a["reference_words"],
                "before_errors": errors(b), "after_errors": errors(a),
                "errors_saved": errors(b) - errors(a),
                "after_substitutions": a["substitutions"],
                "after_deletions": a["deletions"], "after_insertions": a["insertions"]}
        rows.append(item)
        by_speaker[a["speaker"]].append(item)
    with (OUTPUT / "error_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r["after_errors"]), reverse=True))
    totals = Counter({field: sum(int(r[field]) for r in after.values())
                      for field in ("substitutions", "deletions", "insertions")})
    summary = {"clips": len(rows), "after_error_types": dict(totals),
               "after_total_errors": sum(totals.values()),
               "clips_with_more_than_10_insertions": sum(int(r["after_insertions"]) > 10 for r in rows),
               "by_speaker": {speaker: {"clips": len(group),
                                        "after_errors": sum(int(r["after_errors"]) for r in group),
                                        "errors_saved": sum(int(r["errors_saved"]) for r in group),
                                        "after_insertions": sum(int(r["after_insertions"]) for r in group)}
                              for speaker, group in sorted(by_speaker.items())},
               "worst_clips": [{k: row[k] for k in ("speaker", "session", "utterance_id", "after_errors",
                                                    "after_insertions", "errors_saved")}
                               for row in sorted(rows, key=lambda r: int(r["after_errors"]), reverse=True)[:10]]}
    (OUTPUT / "error_analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
