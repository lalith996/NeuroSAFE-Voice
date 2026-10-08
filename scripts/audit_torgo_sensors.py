"""Screen TORGO position tracks for review candidates without excluding data.

The AG500 streams are already format/timing checked by validate_torgo_ema.py.
Here, exact flat channels are treated as unavailable for a speaker/session when
they are flat in at least 90% of its recordings. Large frame steps are only
review flags; their physical cause cannot be established from this script.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/torgo_aai_verified_pairs.csv"
OUTPUT = ROOT / "data/processed/torgo_sensor_audit.csv"
MASKS = ROOT / "data/processed/torgo_sensor_availability.csv"
QUEUE = ROOT / "reports/torgo_sensor_review_queue.csv"
SUMMARY = ROOT / "data/processed/torgo_sensor_audit_summary.json"
SENSORS = 12
FIELDS = [
    "speaker", "condition", "session", "utterance_id", "pos_path",
    "frames", "flat_sensors", "normally_active_flat_sensors",
    "large_step_sensors", "steps_over_20", "steps_over_10",
    "max_step_units", "p99_step_units", "review_flag", "review_priority",
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    measurements = []
    by_session: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for index, row in enumerate(source, 1):
        path = ROOT / row["pos_path"]
        track = np.memmap(path, dtype="<f4", mode="r").reshape(-1, SENSORS, 7)
        xyz = track[:, :, :3]
        # Exact flat traces are a reliable unavailable-channel clue in TORGO;
        # tiny real motion is not called flat.
        flat = np.max(np.ptp(xyz, axis=0), axis=1) < 0.001
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=2)
        per_sensor_large = np.sum(steps > 20.0, axis=0)
        item = {
            "speaker": row["speaker"], "condition": row["condition"],
            "session": row["session"], "utterance_id": row["utterance_id"],
            "pos_path": row["pos_path"], "frames": len(track),
            "flat": flat, "large": per_sensor_large,
            "steps_over_20": int(np.sum(steps > 20.0)),
            "steps_over_10": int(np.sum(steps > 10.0)),
            "max_step_mm": float(np.max(steps)) if steps.size else 0.0,
            "p99_step_mm": float(np.quantile(steps, 0.99)) if steps.size else 0.0,
        }
        measurements.append(item)
        by_session[(row["speaker"], row["session"])].append(flat)
        if index % 1000 == 0 or index == len(source):
            print(f"Screened {index}/{len(source)} TORGO tracks", flush=True)

    unavailable = {}
    mask_rows = []
    for key, traces in sorted(by_session.items()):
        flat_rate = np.mean(np.stack(traces), axis=0)
        mask = flat_rate >= 0.9
        unavailable[key] = mask
        for sensor in range(SENSORS):
            mask_rows.append({
                "speaker": key[0], "session": key[1], "sensor_index_1based": sensor + 1,
                "recordings": len(traces), "flat_recording_fraction": round(float(flat_rate[sensor]), 4),
                "usually_unavailable": int(mask[sensor]),
            })

    rows = []
    for item in measurements:
        mask = unavailable[(item["speaker"], item["session"])]
        active_flat = np.flatnonzero(item["flat"] & ~mask) + 1
        large = np.flatnonzero((item["large"] > 0) & ~mask) + 1
        flags = []
        if len(active_flat):
            flags.append("normally_active_channel_flat")
        if len(large):
            flags.append("large_position_step")
        rows.append({
            "speaker": item["speaker"], "condition": item["condition"],
            "session": item["session"], "utterance_id": item["utterance_id"],
            "pos_path": item["pos_path"], "frames": item["frames"],
            "flat_sensors": "|".join(map(str, np.flatnonzero(item["flat"]) + 1)),
            "normally_active_flat_sensors": "|".join(map(str, active_flat)),
            "large_step_sensors": "|".join(map(str, large)),
            "steps_over_20": item["steps_over_20"],
            "steps_over_10": item["steps_over_10"],
            "max_step_units": round(item["max_step_mm"], 3),
            "p99_step_units": round(item["p99_step_mm"], 3),
            "review_flag": "|".join(flags),
            "review_priority": "high" if len(active_flat) or item["steps_over_20"] >= 10 else "medium" if flags else "none",
        })
    write_csv(OUTPUT, rows, FIELDS)
    write_csv(MASKS, mask_rows, list(mask_rows[0]))
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    flagged = [row for row in rows if row["review_flag"]]
    flagged.sort(key=lambda row: (row["review_priority"] != "high", -int(row["steps_over_20"]), row["speaker"], row["session"], row["utterance_id"]))
    write_csv(QUEUE, flagged, FIELDS)
    summary = {
        "screened_tracks": len(rows),
        "tracks_with_review_flags": sum(bool(row["review_flag"]) for row in rows),
        "normally_active_channel_flat": sum("normally_active_channel_flat" in row["review_flag"] for row in rows),
        "large_position_step": sum("large_position_step" in row["review_flag"] for row in rows),
        "high_priority_tracks": sum(row["review_priority"] == "high" for row in rows),
        "medium_priority_tracks": sum(row["review_priority"] == "medium" for row in rows),
        "usually_unavailable_speaker_session_channels": sum(row["usually_unavailable"] for row in mask_rows),
        "rule_note": "Flat: <0.001 position-unit total variation; usually unavailable: flat in >=90% of speaker/session recordings; large step: >20 position units in one 5 ms frame on a normally active channel. Flags require review and are not automatic exclusions.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
