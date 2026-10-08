"""Export sensor-free AAI estimates for all TORGO ASR clips.

Each speaker is inferred with the TORGO AAI model for the fold where that
speaker was held out. Thus the model never saw that speaker's EMA targets.
"""
from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np

from prepare_mocha_features import audio_features, mel_filterbank
from train_mocha_ridge_aai import context

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "data/results/torgo_mocha_transfer"
OUTPUT = PROCESSED / "torgo_oof_aai_features"
INDEX = PROCESSED / "torgo_oof_aai_index.csv"


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance-cmvn", action="store_true")
    args = parser.parse_args()
    results = (ROOT / "data/results/torgo_mocha_transfer_cmvn") if args.utterance_cmvn else RESULTS
    output = (PROCESSED / "torgo_oof_aai_features_cmvn") if args.utterance_cmvn else OUTPUT
    index_path = (PROCESSED / "torgo_oof_aai_cmvn_index.csv") if args.utterance_cmvn else INDEX
    config_path = (PROCESSED / "torgo_oof_aai_cmvn_config.json") if args.utterance_cmvn else (PROCESSED / "torgo_oof_aai_config.json")
    output.mkdir(parents=True, exist_ok=True)
    rows = read(ROOT / "data/exports/torgo_asr_targets.csv")
    splits = read(PROCESSED / "torgo_speaker_splits.csv")
    fold_by_speaker = {r["speaker"]: int(r["fold"]) for r in splits if r["role"] == "test"}
    if len(fold_by_speaker) != 15:
        raise ValueError("Expected every speaker in exactly one test fold")
    models = {}
    for fold in range(4):
        with np.load(results / f"fold{fold}_torgo_only.npz") as item:
            models[fold] = {key: item[key] for key in item.files}
    filters = mel_filterbank()
    index = []
    for number, row in enumerate(rows, 1):
        with wave.open(str(ROOT / row["audio_path"]), "rb") as handle:
            if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
                raise ValueError(f"Unexpected audio format: {row['audio_path']}")
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        count = int(np.floor((len(audio) - 1) / 160)) + 1
        times = np.arange(count, dtype=np.float64) / 100
        mel = audio_features(audio, times, filters)
        model_mel = mel.astype(np.float64)
        if args.utterance_cmvn:
            model_mel = (model_mel - model_mel.mean(axis=0, keepdims=True)) / np.maximum(model_mel.std(axis=0, keepdims=True), 1e-3)
        x = context(model_mel).astype(np.float64)
        fold = fold_by_speaker[row["speaker"]]
        model = models[fold]
        predicted = np.empty((count, 12), dtype=np.float32)
        for j in range(12):
            predicted[:, j] = ((((x - model["x_mean"][j]) / model["x_std"][j])
                                @ model["weights"][j]) * model["y_std"][j] + model["y_mean"][j])
        predicted -= predicted.mean(axis=0, keepdims=True)
        destination = output / row["speaker"] / row["session"] / f"{row['utterance_id']}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            np.savez_compressed(handle, log_mel=mel.astype(np.float16),
                                inferred_aai=predicted.astype(np.float16),
                                frame_times_seconds=times.astype(np.float32))
        index.append({"speaker": row["speaker"], "condition": row["condition"],
                      "session": row["session"], "utterance_id": row["utterance_id"],
                      "fold_with_speaker_held_out": fold,
                      "audio_path": row["audio_path"],
                      "feature_path": str(destination.relative_to(ROOT)),
                      "frames": count, "status": "inferred_not_measured"})
        if number % 1000 == 0 or number == len(rows):
            print(f"Exported {number}/{len(rows)} sensor-free AAI clips", flush=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    config_path.write_text(json.dumps({
        "model": "speaker-held-out TORGO-only ridge AAI with utterance CMVN" if args.utterance_cmvn else "speaker-held-out TORGO-only ridge AAI",
        "utterance_cmvn": args.utterance_cmvn,
        "units": "estimated utterance-centered TORGO positional units",
        "target_channels": json.loads((PROCESSED / "torgo_aai_feature_config.json").read_text())["target_channels"],
        "intended_use": "exploratory M2/M3 ASR fusion inputs; AAI validation correlation is modest",
        "leakage_rule": "each speaker uses only the fold in which that speaker is test",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(index)} AAI estimate files", flush=True)


if __name__ == "__main__":
    main()
