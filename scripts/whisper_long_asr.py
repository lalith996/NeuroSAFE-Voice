"""Longer-recording Whisper Small LoRA evaluation on held-out speakers.

Adapters train on <=12 second clips, then load into the original full 30-second
Whisper model. The paired baseline and adapted model decode the exact same
6-30 second held-out test recordings. Written prompts are the user's designated
project targets; optional clip-linked corrections can be imported later.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_baseline import edit_counts, normalize

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/processed/torgo_long_asr_splits.csv"
TARGETS = ROOT / "data/processed/torgo_long_scoring_targets.csv"
MODEL = "openai/whisper-small.en"
TRAIN_SECONDS = 12
MAX_SECONDS = 30
SEED = 2029


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["speaker"], row["session"], row["utterance_id"]


def features(row, processor, seconds):
    audio, rate = sf.read(ROOT / row["audio_path"], dtype="float32")
    if rate != 16000 or audio.ndim != 1 or not (0 < len(audio) <= seconds * rate):
        raise ValueError(f"Invalid audio for {row['audio_path']}")
    return processor.feature_extractor(audio, sampling_rate=rate, return_tensors="pt",
                                       max_length=seconds * rate, padding="max_length").input_features[0]


def decode(model, processor, rows, targets, stage, device, output):
    model.eval()
    results = []
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            x = features(row, processor, MAX_SECONDS).to(device).unsqueeze(0)
            tokens = model.generate(input_features=x, max_new_tokens=160, do_sample=False)
            hypothesis = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            target = targets[key(row)]
            reference_words, hypothesis_words = normalize(target["scoring_target"]), normalize(hypothesis)
            substitutions, deletions, insertions = edit_counts(reference_words, hypothesis_words)
            results.append({"fold": row["fold"], "speaker": row["speaker"], "session": row["session"],
                            "utterance_id": row["utterance_id"], "audio_path": row["audio_path"],
                            "duration_seconds": row["duration_seconds"], "stage": stage,
                            "reference": target["scoring_target"], "target_status": target["target_status"],
                            "hypothesis": hypothesis, "reference_words": len(reference_words),
                            "substitutions": substitutions, "deletions": deletions, "insertions": insertions})
            if number % 10 == 0 or number == len(rows):
                print(f"  {stage} decoded {number}/{len(rows)}", flush=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def metrics(rows):
    words = sum(r["reference_words"] for r in rows)
    errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows)
    return {"utterances": len(rows), "reference_words": words, "errors": errors,
            "prompt_wer": errors / words if words else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    fold = args.fold
    torch.manual_seed(SEED + fold)
    rng = random.Random(SEED + fold)
    rows = [r for r in read(SPLITS) if int(r["fold"]) == fold]
    train = [r for r in rows if r["role"] == "train"]
    validation = [r for r in rows if r["role"] == "validation"]
    test = [r for r in rows if r["role"] == "test"]
    targets = {key(r): r for r in read(TARGETS) if int(r["fold"]) == fold}
    if len(targets) != len(test) or any(r["target_status"] in ("audio_issue", "unintelligible") for r in targets.values()):
        raise ValueError("Missing or unscoreable test targets")
    output = ROOT / f"data/results/whisper_long_fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL)
    adapter_path = output / "adapter"
    training_seconds = None
    validation_loss = None
    if not (adapter_path / "adapter_model.safetensors").exists():
        base = WhisperForConditionalGeneration.from_pretrained(MODEL)
        base.config.max_source_positions = TRAIN_SECONDS * 50
        base.model.encoder.embed_positions = torch.nn.Embedding.from_pretrained(
            base.model.encoder.embed_positions.weight[:TRAIN_SECONDS * 50].clone(), freeze=True)
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                                target_modules=["q_proj", "v_proj"], bias="none"))
        model.to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        train_cache = {r["audio_path"]: (features(r, processor, TRAIN_SECONDS),
                                         torch.tensor(processor.tokenizer(r["working_target"]).input_ids[1:], dtype=torch.long))
                       for r in train + validation}
        groups = defaultdict(list)
        for row in train:
            groups[row["speaker"]].append(row)
        speakers = sorted(groups)
        losses = []
        started = time.monotonic()
        model.train()
        for step in range(1, args.steps + 1):
            row = rng.choice(groups[rng.choice(speakers)])
            x, labels = train_cache[row["audio_path"]]
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_features=x.to(device).unsqueeze(0),
                         labels=labels.to(device).unsqueeze(0)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 40 == 0 or step == args.steps:
                print(f"  fold {fold} step {step}/{args.steps}, train loss {np.mean(losses[-40:]):.3f}", flush=True)
        model.eval()
        with torch.inference_mode():
            val_losses = []
            for row in validation:
                x, labels = train_cache[row["audio_path"]]
                val_losses.append(float(model(input_features=x.to(device).unsqueeze(0),
                                              labels=labels.to(device).unsqueeze(0)).loss.cpu()))
        validation_loss = float(np.mean(val_losses)) if val_losses else None
        training_seconds = time.monotonic() - started
        model.save_pretrained(adapter_path)
        del model, base, train_cache
        if device == "mps":
            torch.mps.empty_cache()
    else:
        print(f"Using existing fold {fold} adapter", flush=True)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
    baseline_path = output / "before_lora_predictions.csv"
    if baseline_path.exists():
        baseline = read(baseline_path)
        for row in baseline:
            for field in ("reference_words", "substitutions", "deletions", "insertions"):
                row[field] = int(row[field])
    else:
        baseline = decode(base, processor, test, targets, "before_lora", device, baseline_path)
    adapted = PeftModel.from_pretrained(base, adapter_path).to(device)
    adapted_path = output / "after_lora_predictions.csv"
    after = decode(adapted, processor, test, targets, "after_lora", device, adapted_path)
    by_speaker = {}
    for speaker in sorted({r["speaker"] for r in test}):
        by_speaker[speaker] = {"before_lora": metrics([r for r in baseline if r["speaker"] == speaker]),
                               "after_lora": metrics([r for r in after if r["speaker"] == speaker])}
    summary = {"fold": fold, "model": MODEL, "steps": args.steps,
               "train_max_seconds": TRAIN_SECONDS, "test_max_seconds": MAX_SECONDS,
               "train_utterances": len(train), "validation_utterances": len(validation),
               "test_utterances": len(test), "trainable_parameters": 884736,
               "training_seconds": training_seconds, "validation_mean_loss": validation_loss,
               "reference_note": "user-designated TORGO written prompt targets; optional clip-specific corrections can be rescored",
               "before_lora": metrics(baseline), "after_lora": metrics(after), "by_speaker": by_speaker}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fold": fold, "before": summary["before_lora"], "after": summary["after_lora"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
