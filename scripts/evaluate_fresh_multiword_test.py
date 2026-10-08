"""One-shot paired evaluation on the 40 clips frozen before targeted training."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from assistant_core import AssistantCore
from whisper_baseline import edit_counts, normalize
from whisper_long_asr import ROOT, MODEL, MAX_SECONDS, features, key, read

TEST = ROOT / "data/processed/torgo_fresh_multiword_test_40.csv"
META = ROOT / "data/processed/torgo_fresh_multiword_test_40.json"
OUTPUT = ROOT / "data/results/torgo_fresh_multiword_test"


def metric(rows, stage):
    group = [r for r in rows if r["stage"] == stage]
    words = sum(int(r["reference_words"]) for r in group)
    errors = sum(int(r["errors"]) for r in group)
    return {"clips": len(group), "reference_words": words, "word_errors": errors,
            "prompt_wer": errors / words if words else None}


def decode(model, processor, rows, stage, device):
    model.eval()
    scored = []
    with torch.inference_mode():
        for row in rows:
            started = time.perf_counter()
            x = features(row, processor, MAX_SECONDS).to(device).unsqueeze(0)
            tokens = model.generate(input_features=x, max_new_tokens=160, do_sample=False)
            if device == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            hypothesis = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            ref, hyp = normalize(row["scoring_target"]), normalize(hypothesis)
            sub, deletion, insertion = edit_counts(ref, hyp)
            scored.append({"fold": row["fold"], "speaker": row["speaker"],
                           "session": row["session"], "utterance_id": row["utterance_id"],
                           "audio_path": row["audio_path"], "duration_seconds": row["duration_seconds"],
                           "stage": stage, "target_status": row["target_status"],
                           "reference": row["scoring_target"], "hypothesis": hypothesis,
                           "reference_words": len(ref), "substitutions": sub,
                           "deletions": deletion, "insertions": insertion,
                           "errors": sub + deletion + insertion,
                           "processing_seconds": round(elapsed, 5)})
    return scored


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "utterances.csv"
    if result_path.exists():
        raise FileExistsError("Fresh test has already been evaluated; preserve its one-shot result")
    metadata = json.loads(META.read_text())
    if hashlib.sha256(TEST.read_bytes()).hexdigest() != metadata["csv_sha256"]:
        raise ValueError("Frozen test CSV changed")
    test = read(TEST)
    if len(test) != 40 or len({key(r) for r in test}) != 40:
        raise ValueError("Expected 40 unique fresh test clips")
    selections = {}
    for fold in range(4):
        choice = json.loads((ROOT / f"data/results/whisper_targeted_fold{fold}/selection.json").read_text())
        if choice["fresh_test_used"]:
            raise ValueError("Selection used fresh test")
        selections[fold] = choice
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scored = []
    for fold in range(4):
        rows = [r for r in test if int(r["fold"]) == fold]
        processor = WhisperProcessor.from_pretrained(MODEL)
        base = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
        prior_path = ROOT / f"data/results/whisper_full_prompt_fold{fold}/adapter"
        prior = PeftModel.from_pretrained(base, prior_path).to(device)
        baseline = decode(prior, processor, rows, "full_prompt_m1", device)
        scored.extend(baseline)
        selected_path = ROOT / selections[fold]["selected_adapter"]
        if selected_path == prior_path:
            selected = [{**r, "stage": "validation_selected_m1"} for r in baseline]
            del prior
        else:
            del prior, base
            if device == "mps":
                torch.mps.empty_cache()
            base = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
            candidate = PeftModel.from_pretrained(base, selected_path).to(device)
            selected = decode(candidate, processor, rows, "validation_selected_m1", device)
            del candidate
        scored.extend(selected)
        del base
        if device == "mps":
            torch.mps.empty_cache()
        print(f"Evaluated fresh fold {fold}: {len(rows)} clips", flush=True)
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    by_speaker = defaultdict(list)
    for row in scored:
        by_speaker[row["speaker"]].append(row)
    summary = {"test_csv_sha256": metadata["csv_sha256"],
               "fold_selection_steps": {str(fold): selections[fold]["best_step"] for fold in range(4)},
               "all": {stage: metric(scored, stage) for stage in ("full_prompt_m1", "validation_selected_m1")},
               "by_speaker": {speaker: {stage: metric(group, stage)
                                        for stage in ("full_prompt_m1", "validation_selected_m1")}
                              for speaker, group in sorted(by_speaker.items())}}
    original = {key(r): r for r in scored if r["stage"] == "full_prompt_m1"}
    selected = {key(r): r for r in scored if r["stage"] == "validation_selected_m1"}
    if len(original) != 40 or original.keys() != selected.keys():
        raise ValueError("Paired fresh evaluation rows differ")
    paired = defaultdict(list)
    for clip, row in selected.items():
        paired[clip[0]].append((int(row["errors"]), int(original[clip]["errors"]),
                               int(row["reference_words"])))
    rng = np.random.default_rng(2042)
    speakers = sorted(paired)
    estimates = []
    for _ in range(2000):
        sample = []
        for speaker in rng.choice(speakers, size=len(speakers), replace=True):
            group = paired[speaker]
            sample.extend(group[int(i)] for i in rng.integers(0, len(group), len(group)))
        total = np.sum(sample, axis=0)
        estimates.append((total[0] - total[1]) / total[2])
    summary["paired_delta_wer_selected_minus_full"] = (summary["all"]["validation_selected_m1"]["prompt_wer"]
                                                       - summary["all"]["full_prompt_m1"]["prompt_wer"])
    summary["hierarchical_bootstrap_95pct"] = list(map(float, np.quantile(estimates, [0.025, 0.975])))
    latencies = sorted(float(r["processing_seconds"]) for r in selected.values())
    summary["selected_asr_processing_seconds"] = {"median": statistics.median(latencies),
                                                   "p95": latencies[int(.95 * (len(latencies) - 1))],
                                                   "scope": "audio load + feature extraction + model generation; excludes audio capture and UI"}
    intent = AssistantCore()
    summary["narrative_audio_negative_control"] = {
        "clips": len(selected),
        "accidental_intent_matches": sum(intent.parse(r["hypothesis"]) is not None for r in selected.values())}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
