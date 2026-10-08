"""Train a reproducible MOCHA acoustic-to-articulatory ridge baseline.

This is an initial AAI reference model, not the planned neural AAI head. It
predicts 14 upper-incisor-relative EMA coordinates from 40-band log-mel audio
with a seven-frame acoustic context. Each speaker has fixed train/validation/
test utterances. The other speaker is also evaluated as an unseen-speaker swap.
All acoustic and target normalization is fitted within the training partition.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/processed/mocha_aai_feature_index.csv"
FEATURE_CONFIG = ROOT / "data/processed/mocha_aai_feature_config.json"
RESULTS = ROOT / "data/results/mocha_ridge_aai"
RADIUS = 3
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
BOOTSTRAPS = 1000


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def context(features: np.ndarray) -> np.ndarray:
    padded = np.pad(features, ((RADIUS, RADIUS), (0, 0)), mode="edge")
    return np.concatenate([padded[RADIUS + offset:RADIUS + offset + len(features)]
                           for offset in range(-RADIUS, RADIUS + 1)], axis=1)


def load_arrays(row: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / row["feature_path"]) as data:
        x = context(data["log_mel"])
        y = data["ema_target"]
    if len(x) != len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Bad feature pair: {row['feature_path']}")
    return x, y


def training_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    parts = [load_arrays(row) for row in rows]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def moments(rows: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Per-utterance sufficient statistics for pooled Pearson correlation."""
    result = []
    for truth, predicted in rows:
        a, b = truth.astype(np.float64), predicted.astype(np.float64)
        result.append(np.stack([
            np.full(a.shape[1], len(a), dtype=np.float64),
            a.sum(axis=0), b.sum(axis=0), (a * a).sum(axis=0),
            (b * b).sum(axis=0), (a * b).sum(axis=0),
        ]))
    return np.stack(result)


def pearson(stats: np.ndarray) -> np.ndarray:
    n, sy, sp, syy, spp, syp = np.sum(stats, axis=0)
    covariance = syp - sy * sp / n
    variance_y = syy - sy * sy / n
    variance_p = spp - sp * sp / n
    denominator = np.sqrt(np.maximum(variance_y, 0) * np.maximum(variance_p, 0))
    return np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)


def evaluate(rows: list[dict], model: dict, save_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    pairs = []
    total_frames = 0
    for row in rows:
        x, y = load_arrays(row)
        normalized = (x.astype(np.float64) - model["x_mean"]) / model["x_std"]
        prediction = (normalized @ model["weights"]) * model["y_std"] + model["y_mean"]
        prediction = prediction.astype(np.float32)
        pairs.append((y, prediction))
        total_frames += len(y)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_dir / f"{row['utterance_id']}.npz",
                                prediction=prediction, target=y)
    stats = moments(pairs)
    return pearson(stats), stats, total_frames


def fit(rows: list[dict], alpha: float) -> dict:
    x, y = training_arrays(rows)
    x_mean = x.mean(axis=0, dtype=np.float64)
    x_std = x.std(axis=0, dtype=np.float64)
    y_mean = y.mean(axis=0, dtype=np.float64)
    y_std = y.std(axis=0, dtype=np.float64)
    x_std = np.maximum(x_std, 1e-6)
    y_std = np.maximum(y_std, 1e-6)
    normalized_x = (x.astype(np.float64) - x_mean) / x_std
    normalized_y = (y.astype(np.float64) - y_mean) / y_std
    gram = normalized_x.T @ normalized_x
    cross = normalized_x.T @ normalized_y
    gram.flat[::len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, cross)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean,
            "y_std": y_std, "weights": weights, "training_frames": len(x)}


def ci(stats: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    estimates = np.empty((BOOTSTRAPS, stats.shape[2]), dtype=np.float64)
    for i in range(BOOTSTRAPS):
        sample = rng.integers(0, len(stats), size=len(stats))
        estimates[i] = pearson(stats[sample])
    low, high = np.nanquantile(estimates, [0.025, 0.975], axis=0)
    return low, high


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    index = read_csv(INDEX)
    config = json.loads(FEATURE_CONFIG.read_text(encoding="utf-8"))
    targets = config["target_channels"]
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in index:
        by_speaker[row["speaker"]].append(row)
    speakers = sorted(by_speaker)
    results = []
    run_config = {"model": "ridge_regression", "audio_context_radius_frames": RADIUS,
                  "alpha_candidates": ALPHAS, "alpha_selection": "maximum_mean_validation_pearson_r",
                  "train_only_normalization": True, "bootstrap_unit": "utterance",
                  "bootstrap_replicates": BOOTSTRAPS, "feature_config": config}
    for source in speakers:
        train = [row for row in by_speaker[source] if row["role"] == "train"]
        validation = [row for row in by_speaker[source] if row["role"] == "validation"]
        test = [row for row in by_speaker[source] if row["role"] == "test"]
        if len(validation) != 30 or len(test) != 30 or len(train) < 399:
            raise ValueError(f"Unexpected MOCHA split for {source}")
        print(f"Training {source}: {len(train)} train / {len(validation)} validation / {len(test)} test", flush=True)
        x, y = training_arrays(train)
        x_mean, x_std = x.mean(axis=0, dtype=np.float64), np.maximum(x.std(axis=0, dtype=np.float64), 1e-6)
        y_mean, y_std = y.mean(axis=0, dtype=np.float64), np.maximum(y.std(axis=0, dtype=np.float64), 1e-6)
        nx = (x.astype(np.float64) - x_mean) / x_std
        ny = (y.astype(np.float64) - y_mean) / y_std
        gram, cross = nx.T @ nx, nx.T @ ny
        del x, y, nx, ny
        candidates = []
        for alpha in ALPHAS:
            regularized = gram.copy()
            regularized.flat[::len(regularized) + 1] += alpha
            weights = np.linalg.solve(regularized, cross)
            model = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean,
                     "y_std": y_std, "weights": weights}
            r, _, _ = evaluate(validation, model)
            candidates.append((float(np.nanmean(r)), alpha))
            print(f"  alpha {alpha:g}: validation mean r {np.nanmean(r):.3f}", flush=True)
        _, chosen = max(candidates)
        run_config[source + "_chosen_alpha"] = chosen
        final_model = fit(train + validation, chosen)
        np.savez_compressed(RESULTS / f"model_{source}.npz", **final_model)
        evaluations = [(source, "within_speaker_test", test)]
        evaluations.extend((other, "unseen_speaker_swap", by_speaker[other])
                           for other in speakers if other != source)
        for target_speaker, scope, rows in evaluations:
            prediction_dir = RESULTS / "predictions" / f"{source}_to_{target_speaker}_{scope}"
            r, stats, frames = evaluate(rows, final_model, prediction_dir)
            low, high = ci(stats, 2026 + sum(map(ord, source + target_speaker)))
            for j, target_name in enumerate(targets):
                results.append({
                    "train_speaker": source, "test_speaker": target_speaker,
                    "evaluation_scope": scope, "utterances": len(rows), "frames": frames,
                    "target_channel": target_name, "pearson_r": float(r[j]),
                    "ci_low": float(low[j]), "ci_high": float(high[j]),
                    "chosen_alpha": chosen,
                })
            print(f"  {scope} {target_speaker}: {len(rows)} utterances, mean r {np.nanmean(r):.3f}", flush=True)
    with (RESULTS / "channel_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summaries = []
    for source in speakers:
        for target_speaker, scope in [(source, "within_speaker_test"),
                                      (next(s for s in speakers if s != source), "unseen_speaker_swap")]:
            group = [r for r in results if r["train_speaker"] == source and r["test_speaker"] == target_speaker and r["evaluation_scope"] == scope]
            summaries.append({"train_speaker": source, "test_speaker": target_speaker,
                              "scope": scope, "utterances": group[0]["utterances"],
                              "frames": group[0]["frames"],
                              "mean_channel_r": float(np.mean([r["pearson_r"] for r in group])),
                              "median_channel_r": float(np.median([r["pearson_r"] for r in group]))})
    (RESULTS / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
