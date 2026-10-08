"""Decode the two primary MOCHA-TIMIT speakers into paired audio/EMA NPZ files."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "utterance_manifest.csv"
SPLITS = ROOT / "data" / "processed" / "mocha_utterance_splits.csv"
OUTPUT = ROOT / "data" / "processed" / "mocha_aai_decoded"
PAIR_CSV = ROOT / "data" / "processed" / "mocha_aai_pairs.csv"
FIELDS = [
    "speaker", "utterance_id", "role", "audio_path", "ema_path", "decoded_path",
    "audio_samples", "sample_rate_hz", "audio_duration_seconds", "ema_frames",
    "ema_duration_seconds", "duration_gap_seconds", "present_fraction",
    "nonfinite_coordinate_fraction", "preprocessing_flags",
]


def read_nist_audio(path: Path) -> tuple[np.ndarray, int]:
    with path.open("rb") as handle:
        header = handle.read(1024).decode("ascii", errors="strict")
        pcm = handle.read()
    if not header.startswith("NIST_1A") or "end_head" not in header:
        raise ValueError("invalid NIST audio header")

    def integer(field: str) -> int:
        match = re.search(rf"^{field} -i (\d+)$", header, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing NIST field {field}")
        return int(match.group(1))

    count = integer("sample_count")
    rate = integer("sample_rate")
    channels = integer("channel_count")
    bytes_per_sample = integer("sample_n_bytes")
    byte_order = re.search(r"^sample_byte_format -s2 (\d{2})$", header, re.MULTILINE)
    coding = re.search(r"^sample_coding -s3 (\w+)$", header, re.MULTILINE)
    if rate != 16000 or channels != 1 or bytes_per_sample != 2 or coding is None or coding.group(1) != "pcm":
        raise ValueError("unsupported NIST audio format")
    if byte_order is None or byte_order.group(1) not in ("01", "10"):
        raise ValueError("unsupported NIST byte order")
    if len(pcm) != count * bytes_per_sample:
        raise ValueError("NIST sample count does not match payload")
    dtype = "<i2" if byte_order.group(1) == "01" else ">i2"
    return np.frombuffer(pcm, dtype=dtype).astype(np.int16), rate


def read_ema(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    content = path.read_bytes()
    marker = b"EST_Header_End\n"
    offset = content.find(marker)
    if not content.startswith(b"EST_File Track\n") or offset < 0:
        raise ValueError("invalid EST track header")
    offset += len(marker)
    header = content[:offset].decode("ascii", errors="replace")

    def integer(field: str) -> int:
        match = re.search(rf"^{field} (\d+)$", header, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing EST field {field}")
        return int(match.group(1))

    frames = integer("NumFrames")
    channels = integer("NumChannels")
    order = re.search(r"^ByteOrder (\d{2})$", header, re.MULTILINE)
    if order is None or order.group(1) not in ("01", "10"):
        raise ValueError("unsupported EST byte order")
    dtype = "<f4" if order.group(1) == "01" else ">f4"
    data = np.frombuffer(content, dtype=dtype, count=frames * (channels + 2), offset=offset)
    if data.size != frames * (channels + 2) or len(content) - offset != data.nbytes:
        raise ValueError("EST frame count does not match payload")
    matrix = data.reshape(frames, channels + 2)
    times = matrix[:, 0].astype(np.float32)
    present = matrix[:, 1].astype(np.float32)
    coordinates = matrix[:, 2:].astype(np.float32)
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("invalid EMA timestamps")
    names = []
    for index in range(channels):
        match = re.search(rf"^Channel_{index} (.+)$", header, re.MULTILINE)
        names.append(match.group(1).strip() if match else f"channel_{index}")
    return times, present, coordinates, names


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == "MOCHA-TIMIT" and row["speaker"] in ("fsew0", "msak0") and row["aai_pair_candidate"] == "1"]
    with SPLITS.open(newline="", encoding="utf-8") as handle:
        roles = {(row["speaker"], row["utterance_id"]): row["role"] for row in csv.DictReader(handle)}
    output_rows = []
    errors = []
    for index, row in enumerate(rows, 1):
        speaker, uid = row["speaker"], row["utterance_id"]
        destination = OUTPUT / speaker / f"{uid}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            audio, rate = read_nist_audio(ROOT / row["audio_path"])
            times, present, coordinates, channel_names = read_ema(ROOT / row["ema_path"])
            audio_duration = len(audio) / rate
            ema_duration = float(times[-1] - times[0])
            gap = audio_duration - ema_duration
            flags = []
            if abs(gap) > 0.5:
                flags.append("audio_ema_duration_gap_gt_0_5s")
            if np.mean(present == 0) > 0:
                flags.append("ema_missing_frames")
            if not np.all(np.isfinite(coordinates)):
                flags.append("nonfinite_ema_coordinates")
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, audio_pcm16=audio, sample_rate_hz=rate,
                                    ema_times_seconds=times, ema_present=present,
                                    ema_coordinates=coordinates,
                                    ema_channel_names=np.array(channel_names))
            temporary.replace(destination)
            output_rows.append({
                "speaker": speaker, "utterance_id": uid,
                "role": roles[(speaker, uid)],
                "audio_path": row["audio_path"], "ema_path": row["ema_path"],
                "decoded_path": str(destination.relative_to(ROOT)),
                "audio_samples": len(audio), "sample_rate_hz": rate,
                "audio_duration_seconds": round(audio_duration, 5),
                "ema_frames": len(times), "ema_duration_seconds": round(ema_duration, 5),
                "duration_gap_seconds": round(gap, 5),
                "present_fraction": round(float(np.mean(present != 0)), 6),
                "nonfinite_coordinate_fraction": round(float(np.mean(~np.isfinite(coordinates))), 6),
                "preprocessing_flags": "|".join(flags),
            })
        except Exception as exc:
            errors.append((speaker, uid, type(exc).__name__, str(exc)))
        if index % 100 == 0 or index == len(rows):
            print(f"Decoded {index}/{len(rows)} candidate pairs; errors {len(errors)}", flush=True)
    with PAIR_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    with (ROOT / "data" / "processed" / "mocha_aai_decode_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["speaker", "utterance_id", "error_type", "error"])
        writer.writerows(errors)
    print(f"Saved {len(output_rows)} decoded audio/EMA pairs; {len(errors)} errors", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
