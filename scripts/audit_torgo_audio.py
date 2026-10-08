"""Screen TORGO baseline audio for level and clipping review candidates."""

from __future__ import annotations

import csv
import json
import math
import wave
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/whisper_small_en_zero_shot/utterances.csv"
OUTPUT = ROOT / "data/processed/torgo_audio_audit.csv"
SUMMARY = ROOT / "data/processed/torgo_audio_audit_summary.json"
FIELDS = [
    "speaker", "session", "utterance_id", "audio_path", "microphone",
    "duration_seconds", "rms_dbfs", "peak_dbfs", "clipped_sample_fraction",
    "exact_zero_sample_fraction", "review_flag",
]


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for index, row in enumerate(source, 1):
        path = ROOT / row["audio_path"]
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
                raise ValueError(f"Unexpected WAV format: {path}")
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        signal = pcm.astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(signal * signal)))
        peak = float(np.max(np.abs(signal)))
        clipped = float(np.mean((pcm == 32767) | (pcm == -32768)))
        zero = float(np.mean(pcm == 0))
        rms_dbfs = 20 * math.log10(max(rms, 1e-12))
        peak_dbfs = 20 * math.log10(max(peak, 1e-12))
        flags = []
        if rms_dbfs < -45:
            flags.append("very_low_average_level")
        if clipped >= 0.001:
            flags.append("clipped_samples_at_least_0.1_percent")
        rows.append({
            "speaker": row["speaker"], "session": row["session"],
            "utterance_id": row["utterance_id"], "audio_path": row["audio_path"],
            "microphone": row["microphone"], "duration_seconds": row["duration_seconds"],
            "rms_dbfs": round(rms_dbfs, 2), "peak_dbfs": round(peak_dbfs, 2),
            "clipped_sample_fraction": round(clipped, 6),
            "exact_zero_sample_fraction": round(zero, 6),
            "review_flag": "|".join(flags),
        })
        if index % 1000 == 0 or index == len(source):
            print(f"Screened {index}/{len(source)} TORGO audio files", flush=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(flag for row in rows for flag in row["review_flag"].split("|") if flag)
    summary = {
        "screened_recordings": len(rows),
        "recordings_with_review_flags": sum(bool(row["review_flag"]) for row in rows),
        "flag_counts": dict(counts),
        "rule_note": "Very low average level: full-recording RMS below -45 dBFS. Clipping: at least 0.1% of samples equal the 16-bit maximum or minimum. These are triage rules, not automatic exclusions; quiet speech, background sound, and recording gain require listening review.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
