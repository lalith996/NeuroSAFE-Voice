"""Speaker-disjoint TORGO AAI ridge baseline and MOCHA weight-prior transfer.

The transfer experiment is deliberately exploratory: positions come from two
different EMA systems and only their utterance-centered directions correspond.
Both experiments use identical TORGO train/validation/test data and select
regularization on validation speakers alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_mocha_ridge_aai import context

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "data/results/torgo_mocha_transfer"
UTTERANCE_CMVN = False
TARGETS = [f"{name}_{axis}" for axis in ("x", "vertical") for name in ("li", "ul", "ll", "tt", "tb", "td")]
MOCHA_COLUMNS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
ALPHAS = [10.0, 100.0, 1000.0]
PRIOR_STRENGTHS = [10.0, 100.0, 1000.0, 10000.0]
STRIDE = 3  # Train every third 100 Hz frame; evaluate all frames.


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def arrays(row, mocha=False, stride=1):
    with np.load(ROOT / row["feature_path"]) as item:
        mel = item["log_mel"].astype(np.float64)
        if UTTERANCE_CMVN:
            mel = (mel - mel.mean(axis=0, keepdims=True)) / np.maximum(mel.std(axis=0, keepdims=True), 1e-3)
        x = context(mel)[::stride].astype(np.float64)
        y = item["ema_target"][::stride].astype(np.float64)
        if mocha:
            y = y[:, MOCHA_COLUMNS]
            y -= y.mean(axis=0, keepdims=True)
            mask = np.ones(12, dtype=bool)
        else:
            mask = item["target_mask"].astype(bool)
    return x, y, mask


def empty_stats():
    return {
        "n": np.zeros(12), "sx": np.zeros((12, 280)), "sxx": np.zeros((12, 280, 280)),
        "sy": np.zeros(12), "syy": np.zeros(12), "sxy": np.zeros((12, 280)),
    }


def accumulate(stats, x, y, mask):
    if not np.any(mask):
        return
    n = len(x)
    sx = x.sum(axis=0)
    sxx = x.T @ x
    sy, syy, sxy = y.sum(axis=0), (y * y).sum(axis=0), x.T @ y
    for channel in np.flatnonzero(mask):
        stats["n"][channel] += n
        stats["sx"][channel] += sx
        stats["sxx"][channel] += sxx
        stats["sy"][channel] += sy[channel]
        stats["syy"][channel] += syy[channel]
        stats["sxy"][channel] += sxy[:, channel]


def sum_stats(items):
    result = empty_stats()
    for item in items:
        for key in result:
            result[key] += item[key]
    return result


def normalized(stats):
    n = stats["n"]
    xm = stats["sx"] / n[:, None]
    xs = np.sqrt(np.maximum(np.diagonal(stats["sxx"], axis1=1, axis2=2) / n[:, None] - xm * xm, 1e-8))
    ym = stats["sy"] / n
    ys = np.sqrt(np.maximum(stats["syy"] / n - ym * ym, 1e-8))
    gram = np.empty_like(stats["sxx"])
    cross = np.empty_like(stats["sxy"])
    for j in range(12):
        centered_gram = stats["sxx"][j] - n[j] * np.outer(xm[j], xm[j])
        gram[j] = centered_gram / np.outer(xs[j], xs[j])
        cross[j] = (stats["sxy"][j] - n[j] * xm[j] * ym[j]) / (xs[j] * ys[j])
    return {"x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
            "gram": gram, "cross": cross}


def solve(norm, alpha, prior=None, strength=0):
    weights = np.empty((12, 280))
    for j in range(12):
        matrix = norm["gram"][j].copy()
        matrix.flat[::281] += alpha + strength
        rhs = norm["cross"][j] + (strength * prior[j] if prior is not None else 0)
        weights[j] = np.linalg.solve(matrix, rhs)
    return {key: norm[key] for key in ("x_mean", "x_std", "y_mean", "y_std")} | {"weights": weights}


def evaluate_stats(stats, model):
    """Exact pooled Pearson r on the sampled frames, from sufficient statistics."""
    scores = np.full(12, np.nan)
    for j in range(12):
        n = stats["n"][j]
        if n < 2:
            continue
        weight = model["weights"][j] * model["y_std"][j] / model["x_std"][j]
        covariance_xy = stats["sxy"][j] - stats["sx"][j] * stats["sy"][j] / n
        covariance_xx = stats["sxx"][j] - np.outer(stats["sx"][j], stats["sx"][j]) / n
        variance_y = stats["syy"][j] - stats["sy"][j] ** 2 / n
        variance_pred = weight @ covariance_xx @ weight
        if variance_y > 0 and variance_pred > 0:
            scores[j] = (weight @ covariance_xy) / np.sqrt(variance_y * variance_pred)
    return scores


def main():
    global RESULTS, UTTERANCE_CMVN
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance-cmvn", action="store_true")
    args = parser.parse_args()
    UTTERANCE_CMVN = args.utterance_cmvn
    if UTTERANCE_CMVN:
        RESULTS = ROOT / "data/results/torgo_mocha_transfer_cmvn"
    RESULTS.mkdir(parents=True, exist_ok=True)
    torgo = read(PROCESSED / "torgo_aai_feature_index.csv")
    mocha = read(PROCESSED / "mocha_aai_feature_index.csv")
    splits = read(PROCESSED / "torgo_speaker_splits.csv")
    roles = {(int(r["fold"]), r["speaker"]): r["role"] for r in splits}
    by_speaker = defaultdict(list)
    for row in torgo:
        by_speaker[row["speaker"]].append(row)
    print("Gathering MOCHA source statistics", flush=True)
    mocha_stats = empty_stats()
    for i, row in enumerate(mocha, 1):
        if row["role"] != "train":
            continue
        accumulate(mocha_stats, *arrays(row, mocha=True, stride=STRIDE))
    mocha_norm = normalized(mocha_stats)
    mocha_prior = solve(mocha_norm, 1000)["weights"]
    print("Gathering TORGO speaker statistics", flush=True)
    speaker_stats = {}
    for speaker, rows in sorted(by_speaker.items()):
        stats = empty_stats()
        for row in rows:
            accumulate(stats, *arrays(row, stride=STRIDE))
        speaker_stats[speaker] = stats
        print(f"  {speaker}: {len(rows)} recordings", flush=True)
    outputs = []
    configs = []
    for fold in range(4):
        partitions = {role: [r for r in torgo if roles[fold, r["speaker"]] == role]
                      for role in ("train", "validation", "test")}
        train_speakers = [s for s in by_speaker if roles[fold, s] == "train"]
        train_norm = normalized(sum_stats(speaker_stats[s] for s in train_speakers))
        candidates = []
        for method in ("torgo_only", "mocha_prior"):
            search = [(a, 0.0) for a in ALPHAS] if method == "torgo_only" else [
                (a, b) for a in ALPHAS for b in PRIOR_STRENGTHS]
            for alpha, strength in search:
                model = solve(train_norm, alpha, mocha_prior if strength else None, strength)
                validation_speakers = {r["speaker"] for r in partitions["validation"]}
                per_speaker = {s: evaluate_stats(speaker_stats[s], model) for s in validation_speakers}
                score = float(np.nanmean([np.nanmean(v) for v in per_speaker.values()]))
                candidates.append((score, method, alpha, strength))
            selected = max((x for x in candidates if x[1] == method), key=lambda item: item[0])
            _, _, alpha, strength = selected
            if method == "torgo_only":
                train_only_model = solve(train_norm, alpha)
                np.savez_compressed(RESULTS / f"fold{fold}_torgo_only_train_only.npz", **train_only_model)
            # Validation speakers are folded into final training only after selecting hyperparameters.
            final_speakers = [s for s in by_speaker if roles[fold, s] in ("train", "validation")]
            final_norm = normalized(sum_stats(speaker_stats[s] for s in final_speakers))
            model = solve(final_norm, alpha, mocha_prior if strength else None, strength)
            np.savez_compressed(RESULTS / f"fold{fold}_{method}.npz", **model)
            test_speakers = {r["speaker"] for r in partitions["test"]}
            per_speaker = {s: evaluate_stats(speaker_stats[s], model) for s in test_speakers}
            values = evaluate_stats(sum_stats(speaker_stats[s] for s in test_speakers), model)
            for speaker, channel_values in per_speaker.items():
                rows = [r for r in partitions["test"] if r["speaker"] == speaker]
                for j, target in enumerate(TARGETS):
                    outputs.append({"fold": fold, "method": method, "test_speaker": speaker,
                                    "condition": rows[0]["condition"], "target": target,
                                    "pearson_r": channel_values[j],
                                    "valid_utterances": sum(r["valid_target_mask"].split("|")[j] == "1" for r in rows),
                                    "alpha": alpha, "prior_strength": strength})
            configs.append({"fold": fold, "method": method, "validation_score": selected[0],
                            "alpha": alpha, "prior_strength": strength,
                            "train_speakers": train_speakers,
                            "validation_speakers": sorted({r["speaker"] for r in partitions["validation"]}),
                            "test_speakers": sorted(per_speaker),
                            "test_mean_channel_r": float(np.nanmean(values))})
            print(f"fold {fold} {method}: validation {selected[0]:.3f}, test {np.nanmean(values):.3f}", flush=True)
    with (RESULTS / "channel_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)
    (RESULTS / "run_config.json").write_text(json.dumps({
        "train_and_evaluation_stride": STRIDE, "validation_criterion": "mean of per-speaker mean valid-channel Pearson r",
        "utterance_cepstral_mean_variance_normalization": UTTERANCE_CMVN,
        "target": "utterance-centered sagittal coordinates; target scale fitted on TORGO training speakers",
        "transfer": "MOCHA train-only standardized ridge coefficients as a shrinkage prior",
        "limitations": "sensor conventions and coordinate systems differ; this is an exploratory transfer pilot",
        "folds": configs,
    }, indent=2) + "\n", encoding="utf-8")
    print("Wrote", len(outputs), "channel scores", flush=True)


if __name__ == "__main__":
    main()
