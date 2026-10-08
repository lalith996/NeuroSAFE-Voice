"""Prepare 100 Hz TORGO audio/EMA pairs with per-articulator quality masks."""
from __future__ import annotations

import csv
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_mocha_features import audio_features, mel_filterbank

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
OUTPUT = PROCESSED / "torgo_aai_features"
INDEX = PROCESSED / "torgo_aai_feature_index.csv"
CONFIG = PROCESSED / "torgo_aai_feature_config.json"
SENSORS = [8, 6, 7, 3, 2, 1]  # jaw, upper/lower lip, tip, middle, back
NAMES = ["li", "ul", "ll", "tt", "tb", "td"]
TARGETS = [f"{name}_{axis}" for axis in ("x", "vertical") for name in NAMES]


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sensor_set(value):
    return {int(token) for token in value.split("|") if token}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs = read(PROCESSED / "torgo_aai_verified_pairs.csv")
    audit = {(r["speaker"], r["session"], r["utterance_id"]): r
             for r in read(PROCESSED / "torgo_sensor_audit.csv")}
    unavailable = defaultdict(set)
    for row in read(PROCESSED / "torgo_sensor_availability.csv"):
        if row["usually_unavailable"] == "1":
            unavailable[row["speaker"], row["session"]].add(int(row["sensor_index_1based"]))
    publisher = defaultdict(list)
    for row in read(PROCESSED / "torgo_publisher_sensor_notes.csv"):
        if row["usable_as_record_specific_note"] == "1":
            publisher[row["speaker"], row["session"], row["utterance_id"]].append(row)
    filters = mel_filterbank()
    rows = []
    mask_reasons = Counter()
    for number, pair in enumerate(pairs, 1):
        key = pair["speaker"], pair["session"], pair["utterance_id"]
        with wave.open(str(ROOT / pair["head_audio_path"]), "rb") as handle:
            if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
                raise ValueError(f"Unexpected audio format: {key}")
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        track = np.memmap(ROOT / pair["pos_path"], dtype="<f4", mode="r").reshape(-1, 12, 7)
        # The released .pos stream is already head corrected and sampled at 200 Hz.
        # x and z are the sagittal coordinates. Use every second frame at 100 Hz.
        positions = np.asarray(track[::2, :, :3], dtype=np.float32)
        count = min(len(positions), int(np.floor((len(audio) - 1) / 160)) + 1)
        if count < 10:
            raise ValueError(f"Too few frames: {key}")
        times = np.arange(count, dtype=np.float64) / 100
        x = audio_features(audio, times, filters)
        y = np.empty((count, 12), dtype=np.float32)
        for i, sensor in enumerate(SENSORS):
            y[:, i] = positions[:count, sensor - 1, 0]
            y[:, i + 6] = positions[:count, sensor - 1, 2]
        # Per-utterance centering removes session-specific coordinate origin.
        y -= y.mean(axis=0, keepdims=True)
        flags = audit[key]
        invalid = set(unavailable[pair["speaker"], pair["session"]])
        invalid |= sensor_set(flags["flat_sensors"])
        invalid |= sensor_set(flags["large_step_sensors"])
        if invalid:
            mask_reasons["automatic_sensor_flags"] += 1
        for note in publisher[key]:
            comment = note["comment"].lower()
            if any(term in comment for term in ("flat", "turned off", "no movement", "large spike", "check spike", "step", "bad", "broken")):
                invalid |= sensor_set(note["channels_1based"])
                mask_reasons["publisher_fault_notes"] += 1
        mask = np.array([sensor not in invalid for sensor in SENSORS] * 2, dtype=np.uint8)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError(f"Nonfinite data: {key}")
        destination = OUTPUT / pair["speaker"] / pair["session"] / f"{pair['utterance_id']}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            np.savez_compressed(handle, log_mel=x, ema_target=y, target_mask=mask,
                                frame_times_seconds=times.astype(np.float32))
        rows.append({
            "speaker": pair["speaker"], "condition": pair["condition"],
            "session": pair["session"], "utterance_id": pair["utterance_id"],
            "feature_path": str(destination.relative_to(ROOT)), "frames": count,
            "valid_target_dimensions": int(mask.sum()),
            "valid_target_mask": "|".join(map(str, mask)),
            "sensor_flags": "|".join(map(str, sorted(invalid))),
            "alignment": "head_audio_pos_same_start_100hz",
        })
        if number % 1000 == 0 or number == len(pairs):
            print(f"Prepared {number}/{len(pairs)} TORGO pairs", flush=True)
    with INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    CONFIG.write_text(json.dumps({
        "audio_features": "same 40 band log mel, 25 ms window, 10 ms hop as MOCHA",
        "target_channels": TARGETS, "target_sensor_indices_1based": SENSORS * 2,
        "target_coordinate_axes": "TORGO x and z sagittal, head corrected .pos",
        "target_centering": "subtract each utterance's own target mean",
        "mask": "session-unavailable, flat, large-step, and unambiguous publisher sensor-fault notes",
        "speaker_splits": "torgo_speaker_splits.csv; no test speaker training data",
        "source": "https://catalog.ldc.upenn.edu/docs/LDC2012S02/README.txt",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} pairs; {sum(int(r['frames']) for r in rows)} frames; "
          f"masked counts {dict(mask_reasons)}", flush=True)


if __name__ == "__main__":
    main()
