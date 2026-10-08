"""Train a stronger Whisper Small LoRA using the full prompt-target CSV.

Each fold uses only its training speakers from the 9,104-row export. Checkpoints
are selected on frozen validation speakers by teacher-forced loss. The exact
same frozen long test clips are decoded with the saved prior M1 and the new
adapter, so the comparison changes training only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_long_asr import ROOT, MODEL, TRAIN_SECONDS, decode, features, key, metrics, read

SOURCE = ROOT / "data/exports/torgo_asr_targets.csv"
SPLITS = ROOT / "data/processed/torgo_long_asr_splits.csv"
TARGETS = ROOT / "data/processed/torgo_long_scoring_targets.csv"
SEED = 2032
STEPS = 900
CHECKPOINTS = (300, 600, 900)


class BalancedSampler:
    def __init__(self, rows, seed):
        self.rng = random.Random(seed)
        self.groups = defaultdict(list)
        self.cursors = {}
        for row in rows:
            self.groups[(row["condition"], float(row["duration_seconds"]) > 6, row["speaker"])].append(row)
        for group in self.groups.values():
            self.rng.shuffle(group)
        self.speakers = {
            (condition, is_long): sorted({speaker for c, length, speaker in self.groups
                                          if c == condition and length == is_long})
            for condition in ("dysarthric", "control") for is_long in (False, True)
        }

    def next(self):
        condition = "dysarthric" if self.rng.random() < 0.75 else "control"
        is_long = self.rng.random() < 0.30
        speakers = self.speakers[(condition, is_long)]
        if not speakers:
            raise ValueError(f"No examples for {condition}, long={is_long}")
        speaker = self.rng.choice(speakers)
        group_key = condition, is_long, speaker
        group = self.groups[group_key]
        cursor = self.cursors.get(group_key, 0)
        if cursor >= len(group):
            self.rng.shuffle(group)
            cursor = 0
        self.cursors[group_key] = cursor + 1
        return group[cursor]


def load_rows(fold):
    all_targets = read(SOURCE)
    if len(all_targets) != 9104 or any(r["target_status"] != "project_prompt_target" for r in all_targets):
        raise ValueError("Expected the complete user-designated prompt-target export")
    train = [r for r in all_targets if r[f"fold_{fold}_role"] == "train"
             and not r["audio_review_flag"] and 0.3 <= float(r["duration_seconds"]) <= TRAIN_SECONDS]
    fixed = [r for r in read(SPLITS) if int(r["fold"]) == fold]
    validation = [r for r in fixed if r["role"] == "validation"]
    test = [r for r in fixed if r["role"] == "test"]
    train_speakers = {r["speaker"] for r in train}
    if train_speakers & {r["speaker"] for r in validation + test}:
        raise ValueError("Speaker leaked into train and validation/test")
    if len({key(r) for r in train}) != len(train):
        raise ValueError("Duplicate train clip IDs")
    targets = {key(r): r for r in read(TARGETS) if int(r["fold"]) == fold}
    if {key(r) for r in test} != set(targets):
        raise ValueError("Frozen test targets do not match test clips")
    return train, validation, test, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    fold = args.fold
    output = ROOT / f"data/results/whisper_full_prompt_fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    adapter_path = output / "adapter"
    train, validation, test, targets = load_rows(fold)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED + fold)
    sampler = BalancedSampler(train, SEED + fold)
    processor = WhisperProcessor.from_pretrained(MODEL)
    config = {"model": MODEL, "fold": fold, "seed": SEED + fold,
              "steps": STEPS, "checkpoints": CHECKPOINTS, "train_max_seconds": TRAIN_SECONDS,
              "train_pool_count": len(train), "validation_count": len(validation),
              "test_count": len(test), "sampling": "75% dysarthric, 30% >6 second, uniform speaker within stratum",
              "checkpoint_selection": "minimum mean teacher-forced validation loss; test unused",
              "learning_rate": "cosine 1e-4 to 1e-5", "target_file": str(SOURCE.relative_to(ROOT))}
    (output / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    @lru_cache(maxsize=256)
    def cached_input(audio_path, target):
        row = {"audio_path": audio_path}
        x = features(row, processor, TRAIN_SECONDS)
        labels = torch.tensor(processor.tokenizer(target).input_ids[1:], dtype=torch.long)
        return x, labels

    validation_inputs = [(r, cached_input(r["audio_path"], r["working_target"])) for r in validation]
    history = []
    if not (output / "training_history.json").exists():
        base = WhisperForConditionalGeneration.from_pretrained(MODEL)
        base.config.max_source_positions = TRAIN_SECONDS * 50
        base.model.encoder.embed_positions = torch.nn.Embedding.from_pretrained(
            base.model.encoder.embed_positions.weight[:TRAIN_SECONDS * 50].clone(), freeze=True)
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                                target_modules=["q_proj", "v_proj"], bias="none"))
        model.to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.1 + 0.9 * (1 + math.cos(math.pi * min(step, STEPS) / STEPS)) / 2)
        losses = []
        seen = set()
        best_val = float("inf")
        best_step = None
        started = time.monotonic()
        for step in range(1, STEPS + 1):
            model.train()
            row = sampler.next()
            seen.add(key(row))
            x, labels = cached_input(row["audio_path"], row["target_text"])
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_features=x.to(device).unsqueeze(0),
                         labels=labels.to(device).unsqueeze(0)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                print(f"fold {fold} step {step}/{STEPS}, recent train loss {np.mean(losses[-100:]):.3f}", flush=True)
            if step in CHECKPOINTS:
                model.eval()
                with torch.inference_mode():
                    val_losses = [float(model(input_features=x.to(device).unsqueeze(0),
                                              labels=labels.to(device).unsqueeze(0)).loss.cpu())
                                  for _, (x, labels) in validation_inputs]
                val_loss = float(np.mean(val_losses))
                history.append({"step": step, "train_loss_recent": float(np.mean(losses[-100:])),
                                "validation_mean_loss": val_loss, "elapsed_seconds": time.monotonic() - started})
                print(f"fold {fold} checkpoint {step}, validation loss {val_loss:.3f}", flush=True)
                if val_loss < best_val:
                    best_val, best_step = val_loss, step
                    model.save_pretrained(adapter_path)
        (output / "training_history.json").write_text(json.dumps({"checkpoints": history,
            "best_step": best_step, "best_validation_mean_loss": best_val,
            "unique_training_clips_sampled": len(seen), "training_seconds": time.monotonic() - started},
            indent=2) + "\n", encoding="utf-8")
        del model, base
        if device == "mps":
            torch.mps.empty_cache()
    else:
        print(f"Using existing full-prompt adapter for fold {fold}", flush=True)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
    adapted = PeftModel.from_pretrained(base, adapter_path).to(device)
    predictions_path = output / "test_predictions.csv"
    if predictions_path.exists():
        predictions = read(predictions_path)
        for row in predictions:
            for field in ("reference_words", "substitutions", "deletions", "insertions"):
                row[field] = int(row[field])
    else:
        predictions = decode(adapted, processor, test, targets, "full_prompt_lora", device, predictions_path)
    prior = read(ROOT / f"data/results/whisper_long_fold{fold}/after_lora_predictions.csv")
    if {key(r) for r in predictions} != {key(r) for r in prior}:
        raise ValueError("New and previous model test clips differ")
    for row in prior:
        for field in ("reference_words", "substitutions", "deletions", "insertions"):
            row[field] = int(row[field])
    summary = {"fold": fold, "train_pool_count": len(train), "validation_count": len(validation),
               "test_count": len(test), "prior_m1": metrics(prior), "full_prompt_m1": metrics(predictions),
               "by_speaker": {speaker: {"prior_m1": metrics([r for r in prior if r["speaker"] == speaker]),
                                       "full_prompt_m1": metrics([r for r in predictions if r["speaker"] == speaker])}
                              for speaker in sorted({r["speaker"] for r in test})}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
