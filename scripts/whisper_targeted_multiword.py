"""Continue Whisper LoRA on multiword targets; select using validation only.

The fresh 40-clip test file is intentionally never opened by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_baseline import edit_counts, normalize
from whisper_full_prompt_training import load_rows
from whisper_long_asr import ROOT, MODEL, TRAIN_SECONDS, features, key

STEPS = 600
CHECKPOINTS = (300, 600)
SEED = 2041


class MultiwordSampler:
    def __init__(self, rows, seed):
        self.rng = random.Random(seed)
        self.groups = defaultdict(list)
        self.cursor = {}
        for row in rows:
            self.groups[(row["condition"], len(normalize(row["target_text"])) >= 4,
                         row["speaker"])].append(row)
        for group in self.groups.values():
            self.rng.shuffle(group)
        self.speakers = {(condition, multiword): sorted({speaker for c, m, speaker in self.groups
                                                        if c == condition and m == multiword})
                         for condition in ("dysarthric", "control") for multiword in (False, True)}

    def next(self):
        condition = "dysarthric" if self.rng.random() < 0.75 else "control"
        multiword = self.rng.random() < 0.60
        speakers = self.speakers[(condition, multiword)]
        if not speakers:
            raise ValueError(f"No {condition} multiword={multiword} samples")
        speaker = self.rng.choice(speakers)
        group_key = condition, multiword, speaker
        group = self.groups[group_key]
        cursor = self.cursor.get(group_key, 0)
        if cursor == len(group):
            self.rng.shuffle(group)
            cursor = 0
        self.cursor[group_key] = cursor + 1
        return group[cursor]


def validation_scores(model, val_inputs, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for row, (x, labels) in val_inputs:
            tokens = model.generate(input_features=x.to(device).unsqueeze(0),
                                    max_new_tokens=64, do_sample=False)
            hypothesis = row["processor"].tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            reference = row["working_target"]
            ref, hyp = normalize(reference), normalize(hypothesis)
            sub, deletion, insertion = edit_counts(ref, hyp)
            rows.append({"speaker": row["speaker"], "session": row["session"],
                         "utterance_id": row["utterance_id"], "reference_words": len(ref),
                         "errors": sub + deletion + insertion, "multiword": len(ref) >= 4,
                         "hypothesis": hypothesis})
    def score(group):
        words = sum(r["reference_words"] for r in group)
        errors = sum(r["errors"] for r in group)
        return {"clips": len(group), "words": words, "errors": errors,
                "wer": errors / words if words else None}
    return {"all": score(rows), "multiword": score([r for r in rows if r["multiword"]])}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    fold = args.fold
    output = ROOT / f"data/results/whisper_targeted_fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    if selection_path.exists():
        print(selection_path.read_text(), flush=True)
        return
    train, validation, _, _ = load_rows(fold)
    train_keys = {key(r) for r in train}
    if train_keys & {key(r) for r in validation}:
        raise ValueError("Validation clip leaked into training")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED + fold)
    processor = WhisperProcessor.from_pretrained(MODEL)
    sampler = MultiwordSampler(train, SEED + fold)

    @lru_cache(maxsize=256)
    def input_for(audio_path, target):
        x = features({"audio_path": audio_path}, processor, TRAIN_SECONDS)
        labels = torch.tensor(processor.tokenizer(target).input_ids[1:], dtype=torch.long)
        return x, labels

    val_inputs = []
    for row in validation:
        val_inputs.append(({**row, "processor": processor}, input_for(row["audio_path"], row["working_target"])))
    base = WhisperForConditionalGeneration.from_pretrained(MODEL)
    base.config.max_source_positions = TRAIN_SECONDS * 50
    base.model.encoder.embed_positions = torch.nn.Embedding.from_pretrained(
        base.model.encoder.embed_positions.weight[:TRAIN_SECONDS * 50].clone(), freeze=True)
    original_adapter = ROOT / f"data/results/whisper_full_prompt_fold{fold}/adapter"
    model = PeftModel.from_pretrained(base, original_adapter, is_trainable=True).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    history = []
    best_step = 0
    started = time.monotonic()
    baseline, _ = validation_scores(model, val_inputs, device)
    best_errors, best_words = baseline["multiword"]["errors"], baseline["multiword"]["words"]
    history.append({"step": 0, "validation": baseline})
    print(f"fold {fold} baseline multiword validation {best_errors}/{best_words}", flush=True)
    losses = []
    seen = set()
    for step in range(1, STEPS + 1):
        model.train()
        row = sampler.next()
        seen.add(key(row))
        x, labels = input_for(row["audio_path"], row["target_text"])
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_features=x.to(device).unsqueeze(0),
                     labels=labels.to(device).unsqueeze(0)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 100 == 0:
            print(f"fold {fold} step {step}/{STEPS}, loss {np.mean(losses[-100:]):.3f}", flush=True)
        if step in CHECKPOINTS:
            scores, rows = validation_scores(model, val_inputs, device)
            history.append({"step": step, "validation": scores,
                            "train_loss_recent": float(np.mean(losses[-100:]))})
            errors, words = scores["multiword"]["errors"], scores["multiword"]["words"]
            print(f"fold {fold} checkpoint {step} multiword validation {errors}/{words}", flush=True)
            if errors < best_errors:
                best_errors, best_step = errors, step
                model.save_pretrained(output / "adapter")
            with (output / f"validation_step{step}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    result = {"fold": fold, "train_pool_count": len(train), "validation_count": len(validation),
              "steps": STEPS, "sampled_unique_training_clips": len(seen),
              "best_step": best_step, "best_multiword_errors": best_errors,
              "best_multiword_words": best_words,
              "selection_rule": "minimum multiword validation WER, prior adapter wins ties",
              "source_adapter": str(original_adapter.relative_to(ROOT)),
              "selected_adapter": str((output / "adapter" if best_step else original_adapter).relative_to(ROOT)),
              "sampling": "75% dysarthric; 60% targets with at least four words; speaker balanced",
              "training_seconds": time.monotonic() - started,
              "history": history, "fresh_test_used": False}
    selection_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
