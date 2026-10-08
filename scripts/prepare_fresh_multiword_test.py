"""Freeze an unused, balanced multiword TORGO test before the next model run."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from whisper_baseline import normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/exports/torgo_asr_targets.csv"
OLD_LONG = ROOT / "data/exports/torgo_long_test_targets.csv"
OLD_REVIEW = ROOT / "reports/transcript_review_queue.csv"
OLD_SHORT = ROOT / "data/results"
OUTPUT = ROOT / "data/processed/torgo_fresh_multiword_test_40.csv"
META = ROOT / "data/processed/torgo_fresh_multiword_test_40.json"
SEED = 2040


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["speaker"], row["session"], row["utterance_id"]


def main():
    if OUTPUT.exists() or META.exists():
        raise FileExistsError("Fresh test is already frozen; preserve its existing IDs")
    source = read(SOURCE)
    excluded = {key(r) for r in read(OLD_LONG)} | {key(r) for r in read(OLD_REVIEW)}
    for fold in range(4):
        path = OLD_SHORT / f"whisper_small_lora_fold{fold}_pilot/selected_utterances.csv"
        excluded |= {key(r) for r in read(path) if r["role"] == "test"}
    by_speaker = defaultdict(list)
    for row in source:
        duration = float(row["duration_seconds"])
        if (row["condition"] == "dysarthric" and not row["audio_review_flag"]
                and 0.3 <= duration <= 30 and len(normalize(row["target_text"])) >= 4
                and key(row) not in excluded):
            by_speaker[row["speaker"]].append(row)
    rng = random.Random(SEED)
    selected = []
    test_fold_by_speaker = {}
    for speaker in sorted(by_speaker):
        pool = by_speaker[speaker]
        if len(pool) < 5:
            raise ValueError(f"Insufficient unused multiword clips for {speaker}: {len(pool)}")
        chosen = rng.sample(pool, 5)
        for row in chosen:
            folds = [i for i in range(4) if row[f"fold_{i}_role"] == "test"]
            if len(folds) != 1:
                raise ValueError(f"Expected one held-out fold for {speaker}")
            test_fold_by_speaker[speaker] = folds[0]
            selected.append({"fold": folds[0], "speaker": speaker, "session": row["session"],
                             "utterance_id": row["utterance_id"], "audio_path": row["audio_path"],
                             "duration_seconds": row["duration_seconds"],
                             "written_prompt": row["written_prompt"],
                             "scoring_target": row["target_text"],
                             "target_status": row["target_status"]})
    if len(selected) != 40 or len({key(r) for r in selected}) != 40:
        raise ValueError("Expected 40 unique frozen clips")
    selected.sort(key=lambda r: (r["fold"], r["speaker"], r["session"], r["utterance_id"]))
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    metadata = {"selection_seed": SEED, "clips": len(selected),
                "per_speaker": {speaker: 5 for speaker in sorted(by_speaker)},
                "held_out_fold_by_speaker": test_fold_by_speaker,
                "exclusions": [str(OLD_LONG.relative_to(ROOT)), str(OLD_REVIEW.relative_to(ROOT)),
                               "prior short pilot test IDs"],
                "selection": "five per dysarthric speaker, four or more prompt words, no audio audit flag, 0.3-30 seconds",
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "csv_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                "status": "frozen_before_targeted_training"}
    META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
