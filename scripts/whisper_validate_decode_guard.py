"""Evaluate a bounded, anti-repetition decoder on validation speakers.

Run --split validation for all four folds before choosing a decoder. Test mode
is permitted only after that validation decision is recorded in the report.
"""
from __future__ import annotations

import argparse
import csv
import json

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_baseline import edit_counts, normalize
from whisper_long_asr import ROOT, MODEL, MAX_SECONDS, features, key, read

SPLITS = ROOT / "data/processed/torgo_long_asr_splits.csv"
TARGETS = ROOT / "data/exports/torgo_long_test_targets.csv"
OPTIONS = {"default": {"max_new_tokens": 160},
           "guard": {"max_new_tokens": 64, "no_repeat_ngram_size": 3}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--option", choices=OPTIONS, default="guard")
    parser.add_argument("--model-variant", choices=("prior", "full"), default="full")
    args = parser.parse_args()
    if args.split == "test" and args.option == "default":
        raise ValueError("Default test predictions are already saved")
    rows = [r for r in read(SPLITS) if int(r["fold"]) == args.fold and r["role"] == args.split]
    targets = {key(r): r for r in read(TARGETS)} if args.split == "test" else None
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
    adapter = (ROOT / f"data/results/whisper_long_fold{args.fold}/adapter" if args.model_variant == "prior"
               else ROOT / f"data/results/whisper_full_prompt_fold{args.fold}/adapter")
    model = PeftModel.from_pretrained(base, adapter).to(device)
    model.eval()
    output = ROOT / f"data/results/whisper_full_prompt_fold{args.fold}/{args.split}_{args.model_variant}_{args.option}_decode.csv"
    scored = []
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            x = features(row, processor, MAX_SECONDS).to(device).unsqueeze(0)
            tokens = model.generate(input_features=x, do_sample=False, **OPTIONS[args.option])
            hypothesis = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            reference = (targets[key(row)]["scoring_target"] if targets is not None else row["working_target"])
            ref, hyp = normalize(reference), normalize(hypothesis)
            sub, deletion, insertion = edit_counts(ref, hyp)
            scored.append({"fold": args.fold, "split": args.split, "option": args.option,
                           "model_variant": args.model_variant,
                           "speaker": row["speaker"], "session": row["session"],
                           "utterance_id": row["utterance_id"], "audio_path": row["audio_path"],
                           "reference": reference, "hypothesis": hypothesis,
                           "reference_words": len(ref), "substitutions": sub,
                           "deletions": deletion, "insertions": insertion,
                           "errors": sub + deletion + insertion})
            if number % 10 == 0 or number == len(rows):
                print(f"fold {args.fold} {args.split} {args.model_variant} {args.option}: {number}/{len(rows)}", flush=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    words = sum(r["reference_words"] for r in scored)
    errors = sum(r["errors"] for r in scored)
    long = [r for r, src in zip(scored, rows) if float(src["duration_seconds"]) > 6]
    long_words = sum(r["reference_words"] for r in long)
    long_errors = sum(r["errors"] for r in long)
    summary = {"fold": args.fold, "split": args.split, "option": args.option,
               "model_variant": args.model_variant,
               "clips": len(scored), "words": words, "errors": errors, "wer": errors / words,
               "long_clips": len(long), "long_words": long_words, "long_errors": long_errors,
               "long_wer": long_errors / long_words if long_words else None}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
