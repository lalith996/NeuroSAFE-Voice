"""Validate TORGO AG500 .pos tracks against the synchronous head microphone.

The AG500 position stream is 200 Hz, with 12 sensors and seven float32 values
per sensor: x, y, z, phi, theta, RMS, and an extra field. See the TORGO source
documentation and the TAPADM manual linked in data/processed/README.md.
"""

from __future__ import annotations

import csv
import wave
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "torgo_aai_pair_candidates.csv"
OUTPUT = ROOT / "data" / "processed" / "torgo_aai_validation.csv"
VERIFIED = ROOT / "data" / "processed" / "torgo_aai_verified_pairs.csv"
FRAME_RATE = 200
SENSORS = 12
VALUES_PER_SENSOR = 7
FRAME_BYTES = SENSORS * VALUES_PER_SENSOR * 4
FIELDS = [
    "speaker", "condition", "session", "utterance_id", "head_audio_path",
    "array_audio_path", "pos_path", "pos_frames", "pos_duration_seconds",
    "head_duration_seconds", "timing_gap_seconds", "nonfinite_position_fraction",
    "p95_xyz_step", "status",
]


def head_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise ValueError("unexpected head microphone format")
        return handle.getnframes() / handle.getframerate()


def validate(row: dict) -> dict:
    result = {
        "speaker": row["speaker"], "condition": row["condition"],
        "session": row["session"], "utterance_id": row["utterance_id"],
        "head_audio_path": row["audio_head_path"],
        "array_audio_path": row["audio_array_path"], "pos_path": row["ema_path"],
        "pos_frames": "", "pos_duration_seconds": "", "head_duration_seconds": "",
        "timing_gap_seconds": "", "nonfinite_position_fraction": "",
        "p95_xyz_step": "", "status": "",
    }
    path = ROOT / row["ema_path"]
    size = path.stat().st_size
    if not size or size % FRAME_BYTES:
        result["status"] = "invalid_pos_size"
        return result
    frames = size // FRAME_BYTES
    track = np.memmap(path, dtype="<f4", mode="r", shape=(frames, SENSORS, VALUES_PER_SENSOR))
    xyz = track[:, :, :3]
    nonfinite = float(np.mean(~np.isfinite(xyz)))
    result["pos_frames"] = frames
    result["pos_duration_seconds"] = round(frames / FRAME_RATE, 5)
    result["nonfinite_position_fraction"] = round(nonfinite, 8)
    if nonfinite:
        result["status"] = "nonfinite_positions"
        return result
    if frames > 1:
        # Diagnostic only: abrupt motion may reflect sensor faults, not format errors.
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=2)
        result["p95_xyz_step"] = round(float(np.quantile(steps, 0.95)), 4)
    if not row["audio_head_path"]:
        result["status"] = "array_only_alignment_needed"
        return result
    try:
        duration = head_duration(ROOT / row["audio_head_path"])
    except (ValueError, wave.Error, EOFError):
        result["status"] = "invalid_head_audio"
        return result
    gap = duration - frames / FRAME_RATE
    result["head_duration_seconds"] = round(duration, 5)
    result["timing_gap_seconds"] = round(gap, 5)
    result["status"] = "verified_head_aligned" if abs(gap) <= 0.02 else "head_timing_mismatch"
    return result


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    rows = []
    for index, row in enumerate(candidates, 1):
        rows.append(validate(row))
        if index % 1000 == 0 or index == len(candidates):
            print(f"Validated {index}/{len(candidates)} TORGO position tracks", flush=True)
    write(OUTPUT, rows)
    write(VERIFIED, [row for row in rows if row["status"] == "verified_head_aligned"])
    print("Status counts:", dict(Counter(row["status"] for row in rows)), flush=True)
    print("Verified by condition:", dict(Counter(row["condition"] for row in rows if row["status"] == "verified_head_aligned")), flush=True)


if __name__ == "__main__":
    main()
