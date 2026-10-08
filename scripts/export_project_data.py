"""Export project prompt targets and a portable TORGO listening-review pack.

The full datasets remain at their original project paths. Only the 180 selected
review recordings are copied into a ZIP. Written prompts are the user's chosen
ASR targets; optional clip-specific corrections are tracked separately.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
EXPORT = ROOT / "data/exports"
REVIEW = ROOT / "reports/transcript_review_queue.csv"
REVIEW_HTML = ROOT / "reports/transcript_review.html"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict) -> tuple[str, str, str]:
    return row["speaker"], row["session"], row["utterance_id"]


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(PROCESSED / "utterance_manifest.csv")
    fold_roles = {(int(r["fold"]), r["speaker"]): r["role"] for r in read_csv(PROCESSED / "torgo_speaker_splits.csv")}
    audio_audit = {key(r): r for r in read_csv(PROCESSED / "torgo_audio_audit.csv")}
    sensor_audit = {key(r): r for r in read_csv(PROCESSED / "torgo_sensor_audit.csv")}
    fold_fields = [f"fold_{i}_role" for i in range(4)]

    asr = []
    for row in manifest:
        if row["dataset"] != "TORGO" or row["asr_eligible"] != "1":
            continue
        audit = audio_audit[key(row)]
        asr.append({
            "speaker": row["speaker"], "condition": row["condition"],
            "session": row["session"], "utterance_id": row["utterance_id"],
            "audio_path": row["audio_path"], "sample_rate_hz": row["sample_rate_hz"],
            "duration_seconds": row["duration_seconds"],
            "written_prompt": row["reference_text"],
            "target_text": row["scoring_reference"],
            "target_source": "TORGO_elicitation_prompt",
            "target_status": "project_prompt_target",
            "audio_review_flag": audit["review_flag"],
            **{f"fold_{i}_role": fold_roles[(i, row["speaker"])] for i in range(4)},
        })
    asr_fields = ["speaker", "condition", "session", "utterance_id", "audio_path",
                  "sample_rate_hz", "duration_seconds", "written_prompt", "target_text", "target_source",
                  "target_status", "audio_review_flag", *fold_fields]
    write_csv(EXPORT / "torgo_asr_targets.csv", asr, asr_fields)
    # Keep the original export path for older experiment scripts and notebooks.
    write_csv(EXPORT / "torgo_asr_working_targets.csv", asr, asr_fields)

    aai = []
    for row in read_csv(PROCESSED / "torgo_aai_verified_pairs.csv"):
        sensor = sensor_audit[key(row)]
        aai.append({
            "speaker": row["speaker"], "condition": row["condition"],
            "session": row["session"], "utterance_id": row["utterance_id"],
            "audio_path": row["head_audio_path"], "ema_pos_path": row["pos_path"],
            "pos_frames": row["pos_frames"],
            "sensor_review_flag": sensor["review_flag"],
            "sensor_review_priority": sensor["review_priority"],
            **{f"fold_{i}_role": fold_roles[(i, row["speaker"])] for i in range(4)},
        })
    aai_fields = ["speaker", "condition", "session", "utterance_id", "audio_path",
                  "ema_pos_path", "pos_frames", "sensor_review_flag", "sensor_review_priority", *fold_fields]
    write_csv(EXPORT / "torgo_aai_candidates.csv", aai, aai_fields)

    mocha = read_csv(PROCESSED / "mocha_aai_pairs.csv")
    write_csv(EXPORT / "mocha_aai_pairs.csv", mocha, list(mocha[0]))

    review = read_csv(REVIEW)
    audio_paths = sorted({r["audio_path"] for r in review})
    if len(review) != 180 or len(audio_paths) != 180:
        raise ValueError("Expected 180 distinct review recordings")
    zip_path = EXPORT / "torgo_transcript_review_180.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(REVIEW_HTML, "reports/transcript_review.html")
        archive.write(REVIEW, "reports/transcript_review_queue.csv")
        archive.writestr("README.txt", "Extract this ZIP, then open reports/transcript_review.html in a browser. The 180 included WAV files retain the relative project paths expected by the page. The CSV is an unreviewed starter queue; use Export review CSV after listening. Keep corpus audio for permitted local academic use only.\n")
        for path in audio_paths:
            source = ROOT / path
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, path)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Review ZIP integrity check failed")
        if len(archive.namelist()) != 183:
            raise ValueError("Review ZIP has an unexpected file count")

    summary = {
        "torgo_asr_prompt_targets": len(asr),
        "torgo_aai_format_timing_candidates": len(aai),
        "mocha_aai_pairs": len(mocha),
        "packaged_review_recordings": len(audio_paths),
        "review_zip_bytes": zip_path.stat().st_size,
        "review_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "label_note": "The user designated written TORGO prompts as the project ASR targets. No prior listening checks are linked to individual recording IDs; any future clip-specific corrections are recorded separately.",
    }
    (EXPORT / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
