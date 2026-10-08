"""Freeze longer speaker-held-out ASR clips and clip-linked transcript review."""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from build_transcript_review import FIELDS as REVIEW_FIELDS, html_page

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
SOURCE = ROOT / "data/exports/torgo_asr_targets.csv"
SPLITS = PROCESSED / "torgo_speaker_splits.csv"
AUDIT = PROCESSED / "torgo_audio_audit.csv"
BASELINE = ROOT / "data/results/whisper_small_en_zero_shot/utterances.csv"
OUTPUT = PROCESSED / "torgo_long_asr_splits.csv"
REVIEW = REPORTS / "long_test_transcript_review.csv"
HTML = REPORTS / "long_test_transcript_review.html"
SEED = 2027


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clip_key(row):
    return row["speaker"], row["session"], row["utterance_id"]


def long_review_page(rows):
    page = html_page(rows)
    page = page.replace("<title>TORGO transcript review</title>",
                        "<title>TORGO longer test transcript review</title>")
    page = page.replace("<h1>TORGO transcript review</h1>",
                        "<h1>TORGO longer test transcript review</h1>")
    page = page.replace(
        "This queue mixes randomly chosen and high-disagreement clips; the CSV records which is which.",
        "This page contains the 187 frozen longer held-out test clips. Each card shows its speaker, session, and recording ID."
    )
    page = page.replace("Review flags on sensor motion are for inspection only.",
                        "Enter a reviewer name for every completed decision.")
    page = page.replace("const key='torgo-review-v1';", "const key='torgo-long-review-v1';")
    page = page.replace("a.download='torgo_transcript_review_completed.csv';",
                        "a.download='torgo_long_transcript_review_completed.csv';")
    return page


def main():
    source = read(SOURCE)
    by_speaker = defaultdict(list)
    for row in source:
        duration = float(row["duration_seconds"])
        if not row["audio_review_flag"] and 0.3 <= duration <= 30:
            by_speaker[row["speaker"]].append(row)
    splits = read(SPLITS)
    rng = random.Random(SEED)
    selected = []
    for fold in range(4):
        for split in [r for r in splits if int(r["fold"]) == fold]:
            role, condition, speaker = split["role"], split["condition"], split["speaker"]
            if role in ("validation", "test") and condition == "control":
                continue
            pool = by_speaker[speaker]
            if role == "test":
                # At most 35 per speaker, while retaining all available for
                # speakers with fewer long recordings.
                candidates = [r for r in pool if float(r["duration_seconds"]) > 6]
                chosen = rng.sample(candidates, min(35, len(candidates)))
            elif role == "validation":
                long = [r for r in pool if 6 < float(r["duration_seconds"]) <= 12]
                short = [r for r in pool if float(r["duration_seconds"]) <= 6]
                chosen = rng.sample(long, min(10, len(long))) + rng.sample(short, min(20, len(short)))
            elif condition == "dysarthric":
                long = [r for r in pool if 6 < float(r["duration_seconds"]) <= 12]
                short = [r for r in pool if float(r["duration_seconds"]) <= 6]
                chosen = rng.sample(long, min(40, len(long))) + rng.sample(short, min(100, len(short)))
            else:
                long = [r for r in pool if 6 < float(r["duration_seconds"]) <= 12]
                short = [r for r in pool if float(r["duration_seconds"]) <= 6]
                chosen = rng.sample(long, min(15, len(long))) + rng.sample(short, min(35, len(short)))
            if role == "test" and len(chosen) < 5:
                raise ValueError(f"Insufficient long held-out clips: {speaker}")
            for row in chosen:
                selected.append({"fold": fold, "role": role, "speaker": speaker,
                                 "condition": condition, "session": row["session"],
                                 "utterance_id": row["utterance_id"], "audio_path": row["audio_path"],
                                 "duration_seconds": row["duration_seconds"],
                                 "written_prompt": row["written_prompt"],
                                 "working_target": row["target_text"],
                                 "target_status": row["target_status"]})
    selected.sort(key=lambda r: (int(r["fold"]), r["role"], r["speaker"], r["session"], r["utterance_id"]))
    write(OUTPUT, selected, list(selected[0]))
    test = [r for r in selected if r["role"] == "test"]
    if len({clip_key(r) for r in test}) != len(test):
        raise ValueError("Repeated test clip")
    audit = {clip_key(r): r for r in read(AUDIT)}
    baseline = {clip_key(r): r for r in read(BASELINE)}
    review = []
    for row in test:
        key = clip_key(row)
        original = baseline[key]
        audio = audit[key]
        review.append({
            "speaker": row["speaker"], "condition": row["condition"],
            "session": row["session"], "utterance_id": row["utterance_id"],
            "audio_path": row["audio_path"], "microphone": original["microphone"],
            "duration_seconds": row["duration_seconds"],
            "reference_prompt": row["working_target"],
            "whisper_hypothesis": original["hypothesis"],
            "prompt_reference_wer": original["wer"],
            "selection_reason": f"fold_{row['fold']}_long_held_out_test",
            "audio_rms_dbfs": audio["rms_dbfs"],
            "audio_clipped_sample_fraction": audio["clipped_sample_fraction"],
            "audio_review_flag": audio["review_flag"], "sensor_review_flag": "",
            "review_status": "unreviewed", "verified_transcript": "",
            "review_notes": "", "reviewer": "",
        })
    write(REVIEW, review, REVIEW_FIELDS)
    HTML.write_text(long_review_page(review), encoding="utf-8")
    print(f"Prepared {len(selected)} split rows and {len(test)} unique long test clips", flush=True)
    for speaker in sorted({r["speaker"] for r in test}):
        print(speaker, sum(r["speaker"] == speaker for r in test), flush=True)


if __name__ == "__main__":
    main()
