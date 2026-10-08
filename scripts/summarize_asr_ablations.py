"""Summarize paired fold-0..3 Whisper and fusion pilot results."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/results/asr_ablation_summary.json"
METHODS = ("M0_HF_small", "M1_LoRA", "M2_additive", "M3_gated_cross_attention")


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def error(row):
    return sum(int(row[key]) for key in ("substitutions", "deletions", "insertions"))


def rate(rows, method):
    total = sum(item[method][0] for item in rows)
    words = sum(item[method][1] for item in rows)
    return total / words if words else float("nan")


def main():
    records = {}
    for fold in range(4):
        m1 = read(ROOT / f"data/results/whisper_small_lora_fold{fold}_pilot/test_predictions.csv")
        fusion = read(ROOT / f"data/results/whisper_fusion_fold{fold}_pilot/test_predictions.csv")
        for stage, source, label in [
            ("before_lora", m1, "M0_HF_small"), ("after_lora", m1, "M1_LoRA"),
            ("M2_additive", fusion, "M2_additive"),
            ("M3_gated_cross_attention", fusion, "M3_gated_cross_attention"),
        ]:
            for row in (r for r in source if r.get("stage", r.get("method")) == stage):
                key = (fold, row["speaker"], row["session"], row["utterance_id"])
                item = records.setdefault(key, {"speaker": row["speaker"],
                                                "reference_words": int(row["reference_words"])})
                if item["reference_words"] != int(row["reference_words"]) or label in item:
                    raise ValueError(f"Inconsistent paired data for {key}")
                item[label] = (error(row), int(row["reference_words"]))
        reference = {(r["speaker"], r["session"], r["utterance_id"]): r
                     for r in m1 if r["stage"] == "after_lora"}
        for row in (r for r in fusion if r["method"] == "M1_cached_control"):
            if error(row) != error(reference[row["speaker"], row["session"], row["utterance_id"]]):
                raise ValueError(f"Cached M1 control differs: fold {fold} {row['speaker']} {row['utterance_id']}")
    rows = list(records.values())
    if len(rows) != 280 or any(any(method not in row for method in METHODS) for row in rows):
        raise ValueError("Incomplete paired predictions")
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    if len(by_speaker) != 8 or any(len(group) != 35 for group in by_speaker.values()):
        raise ValueError("Unexpected test speaker coverage")
    rng = np.random.default_rng(2026)
    speaker_names = sorted(by_speaker)
    contrasts = [("M1_vs_M0", "M1_LoRA", "M0_HF_small"),
                 ("M2_vs_M1", "M2_additive", "M1_LoRA"),
                 ("M3_vs_M1", "M3_gated_cross_attention", "M1_LoRA")]
    intervals = {}
    for name, method, baseline in contrasts:
        estimates = []
        for _ in range(2000):
            sample = []
            for speaker in rng.choice(speaker_names, size=len(speaker_names), replace=True):
                group = by_speaker[speaker]
                sample.extend(group[int(i)] for i in rng.integers(0, len(group), size=len(group)))
            estimates.append(rate(sample, method) - rate(sample, baseline))
        intervals[name] = {"delta_wer": rate(rows, method) - rate(rows, baseline),
                           "hierarchical_bootstrap_95pct": list(map(float, np.quantile(estimates, [0.025, 0.975])))}
    result = {
        "test_utterances": len(rows), "speakers": speaker_names,
        "reference_words": sum(r["reference_words"] for r in rows),
        "overall": {method: {"errors": sum(r[method][0] for r in rows),
                              "prompt_wer": rate(rows, method)} for method in METHODS},
        "by_speaker": {speaker: {method: rate(group, method) for method in METHODS}
                       for speaker, group in sorted(by_speaker.items())},
        "by_prompt_length": {name: {method: {"utterances": len(group),
                                                "prompt_wer": rate(group, method)} for method in METHODS}
                             for name, group in (("1_to_3_words", [r for r in rows if r["reference_words"] <= 3]),
                                                 ("4_plus_words", [r for r in rows if r["reference_words"] >= 4]))},
        "paired_contrasts": intervals,
        "bootstrap_note": "2,000 replicates: resample the eight speakers then 35 paired utterances within each sampled speaker; descriptive for this selected pilot set",
        "reference_note": "TORGO elicitation prompt working target, not individually certified verbatim transcript",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "contrasts": intervals}, indent=2))


if __name__ == "__main__":
    main()
