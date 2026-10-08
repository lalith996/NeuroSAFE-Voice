"""Reproducible fold-0 speaker-held-out Whisper Tiny English LoRA pilot.

Targets are the user-designated TORGO written prompts. This pilot is a
feasibility ablation, separate from the existing
Whisper Small zero-shot benchmark.
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
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_baseline import edit_counts, normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/exports/torgo_asr_targets.csv"
RESULTS = ROOT / "data/results/whisper_tiny_lora_fold0_pilot"
MODEL = "openai/whisper-tiny.en"
SECONDS = 6
SEED = 2026


def select(fold):
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    by_speaker = defaultdict(list)
    for row in source:
        if float(row["duration_seconds"]) <= SECONDS and not row["audio_review_flag"]:
            by_speaker[row["speaker"]].append(row)
    rng = random.Random(SEED)
    selected = []
    with (ROOT / "data/processed/torgo_speaker_splits.csv").open(newline="", encoding="utf-8") as handle:
        split_rows = [row for row in csv.DictReader(handle) if int(row["fold"]) == fold]
    planned = []
    for split in split_rows:
        role, condition = split["role"], split["condition"]
        if role == "test" and condition == "control":
            continue  # The pilot evaluates dysarthric recognition only.
        if role == "validation" and condition == "control":
            continue
        limit = 80 if role == "train" and condition == "dysarthric" else 25 if role in ("train", "validation") else 35
        planned.append((role, split["speaker"], limit))
    for role, speaker, limit in planned:
        pool = by_speaker[speaker]
        if len(pool) < limit:
            raise ValueError(f"Insufficient examples for {speaker}")
        chosen = rng.sample(pool, limit)
        for row in chosen:
            selected.append({"role": role, "speaker": speaker, "session": row["session"],
                             "utterance_id": row["utterance_id"], "audio_path": row["audio_path"],
                             "reference": row["target_text"], "duration_seconds": row["duration_seconds"],
                             "target_status": row["target_status"]})
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "selected_utterances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return selected


def inputs(row, processor):
    audio, rate = sf.read(ROOT / row["audio_path"], dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) > SECONDS * rate:
        raise ValueError(f"Unexpected audio: {row['audio_path']}")
    features = processor.feature_extractor(audio, sampling_rate=rate, return_tensors="pt",
                                           max_length=SECONDS * rate, padding="max_length").input_features[0]
    labels = processor.tokenizer(row["reference"]).input_ids[1:]
    return features, torch.tensor(labels, dtype=torch.long)


def decode_rows(model, processor, rows, feature_cache, stage, device):
    model.eval()
    scores = []
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            x = feature_cache[row["audio_path"]][0].unsqueeze(0).to(device)
            output = model.generate(input_features=x, max_new_tokens=48, do_sample=False)
            text = processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
            ref, hyp = normalize(row["reference"]), normalize(text)
            sub, deletion, insertion = edit_counts(ref, hyp)
            scores.append({**{key: row[key] for key in ("speaker", "session", "utterance_id", "reference")},
                           "stage": stage, "hypothesis": text, "reference_words": len(ref),
                           "substitutions": sub, "deletions": deletion, "insertions": insertion})
            if i % 20 == 0 or i == len(rows):
                print(f"  {stage} decoded {i}/{len(rows)}", flush=True)
    return scores


def summary(scores):
    result = {}
    for stage in sorted({r["stage"] for r in scores}):
        for speaker in sorted({r["speaker"] for r in scores}):
            group = [r for r in scores if r["stage"] == stage and r["speaker"] == speaker]
            if not group:
                continue
            words = sum(r["reference_words"] for r in group)
            errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in group)
            result[f"{stage}_{speaker}"] = {"utterances": len(group), "reference_words": words,
                                             "errors": errors, "prompt_wer": errors / words}
    return result


def main():
    global MODEL, RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--model", choices=["openai/whisper-tiny.en", "openai/whisper-small.en"],
                        default="openai/whisper-tiny.en")
    args = parser.parse_args()
    MODEL = args.model
    RESULTS = ROOT / ("data/results/whisper_" + MODEL.split("-")[1].replace(".en", "") + f"_lora_fold{args.fold}_pilot")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    selected = select(args.fold)
    print("Loading model and features", flush=True)
    processor = WhisperProcessor.from_pretrained(MODEL)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL)
    # Six-second clips use the pretrained first 300 positions. This reduces
    # computation without changing the model's weights for those positions.
    base.config.max_source_positions = SECONDS * 50
    base.model.encoder.embed_positions = torch.nn.Embedding.from_pretrained(
        base.model.encoder.embed_positions.weight[:SECONDS * 50].clone(), freeze=True)
    base.generation_config.forced_decoder_ids = None
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base.to(device)
    feature_cache = {row["audio_path"]: inputs(row, processor) for row in selected}
    test = [row for row in selected if row["role"] == "test"]
    train = [row for row in selected if row["role"] == "train"]
    validation = [row for row in selected if row["role"] == "validation"]
    print(f"Data: {len(train)} train, {len(validation)} validation, {len(test)} test", flush=True)
    baseline_path = RESULTS / "baseline_predictions.csv"
    if baseline_path.exists():
        with baseline_path.open(newline="", encoding="utf-8") as handle:
            baseline_scores = list(csv.DictReader(handle))
        expected = {(r["speaker"], r["session"], r["utterance_id"]) for r in test}
        actual = {(r["speaker"], r["session"], r["utterance_id"]) for r in baseline_scores}
        if actual != expected:
            raise ValueError("Saved baseline predictions do not match the selected test set")
        for row in baseline_scores:
            for key in ("reference_words", "substitutions", "deletions", "insertions"):
                row[key] = int(row[key])
    else:
        baseline_scores = decode_rows(base, processor, test, feature_cache, "before_lora", device)
        with baseline_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(baseline_scores[0]))
            writer.writeheader()
            writer.writerows(baseline_scores)
    adapter_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(base, adapter_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    train_rng = random.Random(SEED)
    losses = []
    started = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        row = train_rng.choice(train)
        x, labels = feature_cache[row["audio_path"]]
        optimizer.zero_grad(set_to_none=True)
        output = model(input_features=x.unsqueeze(0).to(device), labels=labels.unsqueeze(0).to(device))
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 20 == 0 or step == args.steps:
            print(f"  step {step}/{args.steps}, loss {np.mean(losses[-20:]):.3f}, "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    model.save_pretrained(RESULTS / "adapter")
    adapted_scores = decode_rows(model, processor, test, feature_cache, "after_lora", device)
    with (RESULTS / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_scores[0]))
        writer.writeheader()
        writer.writerows(baseline_scores + adapted_scores)
    # Validation is diagnostic only; the test set was fixed before training.
    with torch.inference_mode():
        model.eval()
        val_losses = []
        for row in validation:
            x, labels = feature_cache[row["audio_path"]]
            val_losses.append(float(model(input_features=x.unsqueeze(0).to(device),
                                          labels=labels.unsqueeze(0).to(device)).loss.cpu()))
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {"model": MODEL, "fold": args.fold, "audio_limit_seconds": SECONDS,
              "steps": args.steps, "trainable_parameters": parameters,
              "training_utterances": len(train), "validation_utterances": len(validation),
              "test_utterances": len(test), "train_last20_loss": float(np.mean(losses[-20:])),
              "validation_mean_loss": float(np.mean(val_losses)),
              "training_seconds": time.monotonic() - started,
              "reference_type": "TORGO prompt-derived working target; not individually verified",
              "scope": "small feasibility pilot on Whisper Tiny English; separate from Whisper Small M0 benchmark",
              "speaker_prompt_wer": summary(baseline_scores + adapted_scores)}
    (RESULTS / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["speaker_prompt_wer"], indent=2), flush=True)


if __name__ == "__main__":
    main()
