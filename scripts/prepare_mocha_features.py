"""Prepare synchronized 100 Hz acoustic and articulatory features for MOCHA AAI.

MOCHA source documentation says the audio and EMA were recorded synchronously.
EMA ends before the audio files here, so only their timestamp overlap is used;
no estimated lag or forced alignment is applied. Targets are 2D positions of
seven speech articulators relative to the upper incisor. Unknown ``****``
channels, the upper incisor, and bridge-nose reference channels are excluded
from prediction targets. Normalization is deliberately left for training.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/mocha_aai_pairs.csv"
OUTPUT = ROOT / "data/processed/mocha_aai_features"
INDEX = ROOT / "data/processed/mocha_aai_feature_index.csv"
CONFIG = ROOT / "data/processed/mocha_aai_feature_config.json"
RATE = 16000
HOP_HZ = 100
WINDOW_SAMPLES = 400  # 25 ms
FFT_SIZE = 512
MEL_BANDS = 40
TARGETS = ["li_x", "ul_x", "ll_x", "tt_x", "tb_x", "td_x", "v_x",
           "li_y", "ul_y", "ll_y", "tt_y", "tb_y", "td_y", "v_y"]
FIELDS = ["speaker", "utterance_id", "role", "feature_path", "frames",
          "target_dimensions", "ema_last_time_seconds", "audio_duration_seconds",
          "audio_minus_ema_end_seconds", "status"]


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    centers_hz = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(7600), MEL_BANDS + 2))
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1 / RATE)
    filters = np.zeros((MEL_BANDS, len(frequencies)), dtype=np.float32)
    for i in range(MEL_BANDS):
        left, middle, right = centers_hz[i : i + 3]
        filters[i] = np.maximum(0, np.minimum(
            (frequencies - left) / (middle - left),
            (right - frequencies) / (right - middle),
        ))
        filters[i] /= max(float(np.sum(filters[i])), 1e-12)
    return filters


def audio_features(audio: np.ndarray, times: np.ndarray, filters: np.ndarray) -> np.ndarray:
    signal = audio.astype(np.float32) / 32768.0
    centers = np.rint(times * RATE).astype(np.int64)
    offsets = np.arange(WINDOW_SAMPLES, dtype=np.int64) - WINDOW_SAMPLES // 2
    indices = centers[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < len(signal))
    frames = np.zeros(indices.shape, dtype=np.float32)
    frames[valid] = signal[indices[valid]]
    frames *= np.hanning(WINDOW_SAMPLES).astype(np.float32)[None, :]
    power = np.abs(np.fft.rfft(frames, n=FFT_SIZE, axis=1)) ** 2
    return np.log(np.maximum(power @ filters.T, 1e-8)).astype(np.float32)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    filters = mel_filterbank()
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for index, row in enumerate(source, 1):
        with np.load(ROOT / row["decoded_path"]) as data:
            audio = data["audio_pcm16"]
            sample_rate = int(data["sample_rate_hz"])
            ema_times = data["ema_times_seconds"].astype(np.float64)
            ema = data["ema_coordinates"]
            names = data["ema_channel_names"].tolist()
            present = data["ema_present"]
        if sample_rate != RATE or not np.all(present == 1) or not np.all(np.isfinite(ema)):
            raise ValueError(f"Unexpected source quality for {row['speaker']} {row['utterance_id']}")
        if names.count("****") != 2 or len(names) != 20 or not all(name in names for name in TARGETS):
            raise ValueError(f"Unexpected EMA channel layout for {row['speaker']} {row['utterance_id']}")
        if float(ema_times[0]) > 0.0021 or ema_times[-1] >= len(audio) / RATE:
            raise ValueError(f"Unexpected audio/EMA time coverage for {row['speaker']} {row['utterance_id']}")
        first_frame = math.ceil((float(ema_times[0]) - 1e-7) * HOP_HZ)
        last_frame = math.floor((float(ema_times[-1]) + 1e-7) * HOP_HZ)
        target_times = np.arange(first_frame, last_frame + 1, dtype=np.float64) / HOP_HZ
        acoustics = audio_features(audio, target_times, filters)
        target = np.empty((len(target_times), len(TARGETS)), dtype=np.float32)
        for i, name in enumerate(TARGETS):
            anchor = "ui_x" if name.endswith("_x") else "ui_y"
            relative = ema[:, names.index(name)] - ema[:, names.index(anchor)]
            target[:, i] = np.interp(target_times, ema_times, relative)
        if not np.all(np.isfinite(acoustics)) or not np.all(np.isfinite(target)):
            raise ValueError(f"Nonfinite feature for {row['speaker']} {row['utterance_id']}")
        destination = OUTPUT / row["speaker"] / f"{row['utterance_id']}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, log_mel=acoustics, ema_target=target,
                                frame_times_seconds=target_times.astype(np.float32))
        temporary.replace(destination)
        rows.append({
            "speaker": row["speaker"], "utterance_id": row["utterance_id"],
            "role": row["role"], "feature_path": str(destination.relative_to(ROOT)),
            "frames": len(target_times), "target_dimensions": len(TARGETS),
            "ema_last_time_seconds": round(float(ema_times[-1]), 4),
            "audio_duration_seconds": row["audio_duration_seconds"],
            "audio_minus_ema_end_seconds": round(len(audio) / RATE - float(ema_times[-1]), 4),
            "status": "synchronized_overlap_no_lag_estimation",
        })
        if index % 100 == 0 or index == len(source):
            print(f"Prepared {index}/{len(source)} MOCHA feature pairs", flush=True)
    with INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    CONFIG.write_text(json.dumps({
        "audio_rate_hz": RATE, "output_rate_hz": HOP_HZ,
        "window_samples": WINDOW_SAMPLES, "fft_size": FFT_SIZE,
        "mel_bands": MEL_BANDS, "mel_upper_hz": 7600,
        "target_channels": TARGETS, "target_reference": "upper_incisor_same_axis",
        "alignment": "Use supplied synchronous timestamps, zero lag, truncate trailing audio to EMA coverage",
        "normalization": "Not applied; fit from train examples only in the model run",
        "source": "https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} feature pairs; total frames {sum(int(r['frames']) for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
