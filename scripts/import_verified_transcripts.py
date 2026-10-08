"""Validate clip-linked human review decisions and build scoring targets.

Never infer 'verified' from a prompt or a model transcript. Only explicit
review_status decisions with a reviewer identity are accepted.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "reports/long_test_transcript_review.csv"
SPLITS = ROOT / "data/processed/torgo_long_asr_splits.csv"
OUTPUT = ROOT / "data/processed/manual_verified_transcripts.csv"
TARGETS = ROOT / "data/processed/torgo_long_scoring_targets.csv"
TARGET_EXPORT = ROOT / "data/exports/torgo_long_test_targets.csv"
STATUSES = {"unreviewed", "prompt_matches", "corrected", "unintelligible", "audio_issue"}


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row["speaker"], row["session"], row["utterance_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, default=STARTER,
                        help="Exported CSV after a person listened to and marked each clip")
    args = parser.parse_args()
    starter = {key(r): r for r in read(STARTER)}
    reviewed = read(args.review_csv)
    if len({key(r) for r in reviewed}) != len(reviewed):
        raise ValueError("Duplicate clip IDs in submitted review")
    unknown = [key(r) for r in reviewed if key(r) not in starter]
    if unknown:
        raise ValueError(f"Unknown clip IDs: {unknown[:5]}")
    manual = []
    for row in reviewed:
        status = row["review_status"].strip().lower() or "unreviewed"
        if status not in STATUSES:
            raise ValueError(f"Unknown review status for {key(row)}: {status}")
        if row["reference_prompt"] != starter[key(row)]["reference_prompt"] or row["audio_path"] != starter[key(row)]["audio_path"]:
            raise ValueError(f"Prompt/audio identity changed for {key(row)}")
        if status == "unreviewed":
            continue
        reviewer = row["reviewer"].strip()
        if not reviewer:
            raise ValueError(f"Reviewer is required for an explicit decision: {key(row)}")
        transcript = row["verified_transcript"].strip()
        if status == "prompt_matches":
            if transcript and transcript != row["reference_prompt"]:
                raise ValueError(f"Prompt-match text conflicts with prompt: {key(row)}")
            transcript = row["reference_prompt"]
        elif status == "corrected" and not transcript:
            raise ValueError(f"Corrected transcript is empty: {key(row)}")
        elif status in ("unintelligible", "audio_issue"):
            transcript = ""
        manual.append({"speaker": row["speaker"], "session": row["session"],
                       "utterance_id": row["utterance_id"], "audio_path": row["audio_path"],
                       "review_status": status, "verified_transcript": transcript,
                       "reviewer": reviewer, "review_notes": row["review_notes"]})
    write(OUTPUT, manual, ["speaker", "session", "utterance_id", "audio_path", "review_status",
                           "verified_transcript", "reviewer", "review_notes"])
    decisions = {key(r): r for r in manual}
    scoring = []
    for row in read(SPLITS):
        if row["role"] != "test":
            continue
        decision = decisions.get(key(row))
        if decision and decision["review_status"] in ("prompt_matches", "corrected"):
            target, status = decision["verified_transcript"], "human_verified"
        elif decision and decision["review_status"] in ("unintelligible", "audio_issue"):
            target, status = "", decision["review_status"]
        else:
            target, status = row["working_target"], "project_prompt_target"
        scoring.append({"fold": row["fold"], "speaker": row["speaker"],
                        "session": row["session"], "utterance_id": row["utterance_id"],
                        "audio_path": row["audio_path"], "duration_seconds": row["duration_seconds"],
                        "written_prompt": row["written_prompt"],
                        "scoring_target": target, "target_status": status})
    write(TARGETS, scoring, ["fold", "speaker", "session", "utterance_id", "audio_path",
                             "duration_seconds", "written_prompt", "scoring_target", "target_status"])
    write(TARGET_EXPORT, scoring, ["fold", "speaker", "session", "utterance_id", "audio_path",
                                   "duration_seconds", "written_prompt", "scoring_target", "target_status"])
    print(f"Imported {len(manual)} explicit review decisions; "
          f"{sum(r['target_status']=='human_verified' for r in scoring)} human-verified test targets; "
          f"{sum(r['target_status']=='project_prompt_target' for r in scoring)} project prompt targets", flush=True)


if __name__ == "__main__":
    main()
