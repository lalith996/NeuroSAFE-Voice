"""Temporal convolution AAI: TORGO scratch versus MOCHA pretraining.

Uses the fixed speaker-disjoint TORGO folds, per-channel sensor validity masks,
and normalization estimated only from training speakers. MOCHA source training
uses only its designated train utterances. Scores use every third 100 Hz frame,
matching the earlier ridge comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "data/results/temporal_aai"
MOCHA_COLUMNS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
TARGETS = [f"{name}_{axis}" for axis in ("x", "vertical") for name in ("li", "ul", "ll", "tt", "tb", "td")]
WINDOW = 256
BATCH = 12
SEED = 2028


class Block(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                                   dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        value = self.pointwise(self.depthwise(x)).transpose(1, 2)
        return x + self.activation(self.norm(value)).transpose(1, 2)


class TemporalAAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = nn.Conv1d(40, 96, 7, padding=3)
        self.blocks = nn.Sequential(*(Block(96, dilation) for dilation in (1, 2, 4, 8)))
        self.output = nn.Conv1d(96, 12, 1)

    def forward(self, x):
        return self.output(self.blocks(torch.nn.functional.gelu(self.front(x))))


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_data(path, mocha=False):
    rows = read(path)
    data = []
    for number, row in enumerate(rows, 1):
        with np.load(ROOT / row["feature_path"]) as item:
            x = item["log_mel"].astype(np.float32)
            y = item["ema_target"].astype(np.float32)
            if mocha:
                y = y[:, MOCHA_COLUMNS]
                y -= y.mean(axis=0, keepdims=True)
                mask = np.ones(12, dtype=bool)
            else:
                mask = item["target_mask"].astype(bool)
        if x.shape[0] != y.shape[0] or x.shape[1] != 40 or y.shape[1] != 12:
            raise ValueError(f"Bad feature shape: {row['feature_path']}")
        data.append({"row": row, "x": x, "y": y, "mask": mask})
        if number % 1000 == 0 or number == len(rows):
            print(f"  loaded {number}/{len(rows)} {'MOCHA' if mocha else 'TORGO'} pairs", flush=True)
    return data


def normalization(items):
    nx = 0
    sx = np.zeros(40, dtype=np.float64)
    sxx = np.zeros(40, dtype=np.float64)
    ny = np.zeros(12, dtype=np.float64)
    sy = np.zeros(12, dtype=np.float64)
    syy = np.zeros(12, dtype=np.float64)
    for item in items:
        x, y, mask = item["x"][::3], item["y"][::3], item["mask"]
        nx += len(x)
        sx += x.sum(axis=0)
        sxx += np.square(x.astype(np.float64)).sum(axis=0)
        for j in np.flatnonzero(mask):
            ny[j] += len(y)
            sy[j] += y[:, j].sum()
            syy[j] += np.square(y[:, j].astype(np.float64)).sum()
    if nx == 0 or np.any(ny == 0):
        raise ValueError("No valid training frames for a target channel")
    xm, ym = sx / nx, sy / ny
    xs = np.sqrt(np.maximum(sxx / nx - xm * xm, 1e-6))
    ys = np.sqrt(np.maximum(syy / ny - ym * ym, 1e-6))
    return {"xm": xm.astype(np.float32), "xs": xs.astype(np.float32),
            "ym": ym.astype(np.float32), "ys": ys.astype(np.float32)}


def standardized(items, stats):
    result = []
    for item in items:
        x = np.clip((item["x"] - stats["xm"]) / stats["xs"], -6, 6).astype(np.float32)
        y = np.clip((item["y"] - stats["ym"]) / stats["ys"], -8, 8).astype(np.float32)
        result.append({**item, "nx": x, "ny": y})
    return result


def batches(items, rng, speakers=None):
    if speakers is None:
        groups = defaultdict(list)
        for item in items:
            groups[item["row"]["speaker"]].append(item)
    else:
        groups = {name: [item for item in items if item["row"]["speaker"] == name] for name in speakers}
    names = [name for name, group in groups.items() if group]
    if not names:
        raise ValueError("Empty training pool")
    while True:
        xb, yb, mb = [], [], []
        for _ in range(BATCH):
            item = rng.choice(groups[rng.choice(names)])
            count = len(item["nx"])
            start = rng.randrange(max(1, count - WINDOW + 1))
            x = item["nx"][start:start + WINDOW]
            y = item["ny"][start:start + WINDOW]
            valid = np.broadcast_to(item["mask"], (len(x), 12)).copy()
            if len(x) < WINDOW:
                pad = WINDOW - len(x)
                x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
                y = np.pad(y, ((0, pad), (0, 0)), mode="edge")
                valid = np.pad(valid, ((0, pad), (0, 0)), constant_values=False)
            xb.append(x.T)
            yb.append(y.T)
            mb.append(valid.T)
        yield (torch.from_numpy(np.stack(xb)), torch.from_numpy(np.stack(yb)),
               torch.from_numpy(np.stack(mb)))


def fit(model, items, steps, device, seed):
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    iterator = batches(items, random.Random(seed))
    losses = []
    for step in range(1, steps + 1):
        x, y, mask = next(iterator)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        element = torch.nn.functional.smooth_l1_loss(prediction, y, reduction="none", beta=0.5)
        loss = (element * mask).sum() / mask.sum().clamp(min=1)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 50 == 0 or step == steps:
            print(f"    step {step}/{steps}: loss {np.mean(losses[-50:]):.4f}", flush=True)
    return float(np.mean(losses[-50:]))


def evaluate(model, items, stats, device):
    model.to(device).eval()
    accum = defaultdict(lambda: np.zeros((12, 6), dtype=np.float64))
    counts = defaultdict(lambda: np.zeros(12, dtype=np.int64))
    with torch.inference_mode():
        for item in items:
            speaker = item["row"]["speaker"]
            x = np.clip((item["x"] - stats["xm"]) / stats["xs"], -6, 6).astype(np.float32)
            tensor = torch.from_numpy(x.T[None]).to(device)
            prediction = model(tensor)[0].transpose(0, 1).cpu().numpy()[::3]
            prediction = prediction * stats["ys"] + stats["ym"]
            truth = item["y"][::3]
            for j in np.flatnonzero(item["mask"]):
                a, b = truth[:, j].astype(np.float64), prediction[:, j].astype(np.float64)
                accum[speaker][j] += [len(a), a.sum(), b.sum(), a @ a, b @ b, a @ b]
                counts[speaker][j] += 1
    scores = []
    for speaker, moments in sorted(accum.items()):
        for j, (n, sy, sp, syy, spp, syp) in enumerate(moments):
            vy = syy - sy * sy / n if n else 0
            vp = spp - sp * sp / n if n else 0
            covariance = syp - sy * sp / n if n else 0
            r = covariance / np.sqrt(vy * vp) if vy > 0 and vp > 0 else float("nan")
            scores.append({"speaker": speaker, "target": TARGETS[j],
                           "pearson_r": r, "valid_utterances": int(counts[speaker][j]),
                           "frames": int(n)})
    return scores


def macro(scores, condition_by_speaker):
    by_speaker = defaultdict(list)
    for row in scores:
        if row["valid_utterances"] >= 20 and np.isfinite(row["pearson_r"]):
            by_speaker[row["speaker"]].append(row["pearson_r"])
    result = {}
    for condition in ("dysarthric", "control"):
        values = [np.mean(group) for speaker, group in by_speaker.items()
                  if condition_by_speaker[speaker] == condition]
        result[condition] = float(np.mean(values)) if values else float("nan")
    return result


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--mocha-steps", type=int, default=300)
    parser.add_argument("--torgo-steps", type=int, default=300)
    parser.add_argument("--dysarthric-only", action="store_true",
                        help="Fit TORGO AAI using dysarthric training speakers only")
    args = parser.parse_args()
    if args.dysarthric_only:
        RESULTS = ROOT / "data/results/temporal_aai_dys_only"
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading MOCHA AAI features", flush=True)
    mocha = load_data(PROCESSED / "mocha_aai_feature_index.csv", mocha=True)
    mocha_train = [item for item in mocha if item["row"]["role"] == "train"]
    mocha_stats = normalization(mocha_train)
    mocha_model = TemporalAAI()
    print("MOCHA pretraining", flush=True)
    mocha_loss = fit(mocha_model, standardized(mocha_train, mocha_stats), args.mocha_steps, device, SEED)
    torch.save(mocha_model.cpu().state_dict(), RESULTS / "mocha_pretrained.pt")
    del mocha, mocha_train
    print("Loading TORGO AAI features", flush=True)
    torgo = load_data(PROCESSED / "torgo_aai_feature_index.csv")
    split = {(int(r["fold"]), r["speaker"]): r["role"] for r in read(PROCESSED / "torgo_speaker_splits.csv")}
    condition = {item["row"]["speaker"]: item["row"]["condition"] for item in torgo}
    results = []
    summaries = []
    for fold in args.folds:
        partitions = {role: [item for item in torgo if split[fold, item["row"]["speaker"]] == role]
                      for role in ("train", "validation", "test")}
        train_items = [item for item in partitions["train"]
                       if not args.dysarthric_only or item["row"]["condition"] == "dysarthric"]
        stats = normalization(train_items)
        training = standardized(train_items, stats)
        for method in ("scratch", "mocha_pretrained"):
            torch.manual_seed(SEED + fold)
            model = TemporalAAI()
            if method == "mocha_pretrained":
                model.load_state_dict(torch.load(RESULTS / "mocha_pretrained.pt", map_location="cpu", weights_only=True))
            print(f"Fold {fold} {method}: {len(training)} train, {len(partitions['validation'])} val, "
                  f"{len(partitions['test'])} test pairs", flush=True)
            loss = fit(model, training, args.torgo_steps, device, SEED + fold)
            val_scores = evaluate(model, partitions["validation"], stats, device)
            test_scores = evaluate(model, partitions["test"], stats, device)
            val_macro = macro(val_scores, condition)
            test_macro = macro(test_scores, condition)
            np.savez_compressed(RESULTS / f"fold{fold}_{method}_normalization.npz", **stats)
            torch.save(model.cpu().state_dict(), RESULTS / f"fold{fold}_{method}.pt")
            for role, scores in (("validation", val_scores), ("test", test_scores)):
                for row in scores:
                    results.append({"fold": fold, "method": method, "role": role,
                                    "condition": condition[row["speaker"]], **row})
            summaries.append({"fold": fold, "method": method, "train_last50_loss": loss,
                              "validation_macro_r": val_macro, "test_macro_r": test_macro})
            print(f"  val {val_macro}, test {test_macro}", flush=True)
    with (RESULTS / "channel_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (RESULTS / "summary.json").write_text(json.dumps({
        "architecture": "40 log-mel -> 96-channel dilated temporal convolution -> 12 articulator coordinates",
        "mocha_train_steps": args.mocha_steps, "torgo_train_steps_per_fold": args.torgo_steps,
        "batch_windows": BATCH, "window_frames": WINDOW,
        "torgo_training_condition": "dysarthric_only" if args.dysarthric_only else "dysarthric_and_control",
        "training_loss": "masked smooth L1, beta=0.5",
        "sampling": "speaker-balanced windows, one fold held out per speaker",
        "normalization": "audio and targets fit on each corpus/fold training items only",
        "evaluation": "pooled per-speaker/channel Pearson r on every third 100 Hz frame; macro speaker mean from channels with >=20 valid utterances",
        "mocha_last50_loss": mocha_loss, "folds": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    print("Saved temporal AAI results", flush=True)


if __name__ == "__main__":
    main()
