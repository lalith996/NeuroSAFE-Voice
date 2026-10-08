"""Re-score saved long-clip hypotheses when clip-linked transcripts change."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from whisper_baseline import edit_counts, normalize

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data/processed/torgo_long_scoring_targets.csv"
OUTPUT = ROOT / "data/results/whisper_long_rescored"


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["speaker"], row["session"], row["utterance_id"]


def metric(rows, stage):
    group = [r for r in rows if r["stage"] == stage]
    words = sum(int(r["reference_words"]) for r in group)
    errors = sum(int(r["substitutions"]) + int(r["deletions"]) + int(r["insertions"]) for r in group)
    return {"utterances": len(group), "reference_words": words, "errors": errors,
            "wer": errors / words if words else None}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target_rows = read(TARGETS)
    targets = {key(r): r for r in target_rows}
    if len(targets) != len(target_rows):
        raise ValueError("Duplicate clip IDs in scoring targets")
    expected = {
        fold: {key(r) for r in target_rows if int(r["fold"]) == fold}
        for fold in range(4)
    }
    rows = []
    for fold in range(4):
        folder = ROOT / f"data/results/whisper_long_fold{fold}"
        for stage in ("before_lora", "after_lora"):
            source = read(folder / f"{stage}_predictions.csv")
            observed = [key(r) for r in source]
            if len(set(observed)) != len(observed) or set(observed) != expected[fold]:
                raise ValueError(f"Fold {fold} {stage} clip IDs differ from frozen test set")
            for row in source:
                target = targets[key(row)]
                if row["audio_path"] != target["audio_path"]:
                    raise ValueError(f"Audio path differs for {key(row)}")
                if target["target_status"] in ("unintelligible", "audio_issue"):
                    continue
                ref, hyp = normalize(target["scoring_target"]), normalize(row["hypothesis"])
                sub, deletion, insertion = edit_counts(ref, hyp)
                rows.append({"fold": fold, "speaker": row["speaker"], "session": row["session"],
                             "utterance_id": row["utterance_id"], "duration_seconds": row["duration_seconds"],
                             "stage": stage, "scoring_target": target["scoring_target"],
                             "target_status": target["target_status"], "hypothesis": row["hypothesis"],
                             "reference_words": len(ref), "substitutions": sub,
                             "deletions": deletion, "insertions": insertion})
    with (OUTPUT / "utterances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    summary = {"all": {stage: metric(rows, stage) for stage in ("before_lora", "after_lora")},
               "human_verified_only": {stage: metric([r for r in rows if r["target_status"] == "human_verified"], stage)
                                       for stage in ("before_lora", "after_lora")},
               "by_speaker": {speaker: {stage: metric(group, stage) for stage in ("before_lora", "after_lora")}
                              for speaker, group in sorted(by_speaker.items())},
               "reference_note": "The user designated written prompts as project targets. Human-verified rows record optional clip-specific review decisions."}
    # Paired hierarchical bootstrap over fixed speakers and their clips.
    before = {(r["speaker"], r["session"], r["utterance_id"]): r for r in rows if r["stage"] == "before_lora"}
    after = {(r["speaker"], r["session"], r["utterance_id"]): r for r in rows if r["stage"] == "after_lora"}
    if before.keys() != after.keys():
        raise ValueError("Before/after test sets differ")
    paired = defaultdict(list)
    for clip, a in after.items():
        b = before[clip]
        paired[clip[0]].append((sum(int(a[x]) for x in ("substitutions", "deletions", "insertions")),
                                sum(int(b[x]) for x in ("substitutions", "deletions", "insertions")),
                                int(a["reference_words"])))
    rng = np.random.default_rng(2030)
    speakers = sorted(paired)
    estimates = []
    for _ in range(2000):
        sample = []
        for speaker in rng.choice(speakers, size=len(speakers), replace=True):
            group = paired[speaker]
            sample.extend(group[int(i)] for i in rng.integers(0, len(group), len(group)))
        total = np.sum(sample, axis=0)
        estimates.append((total[0] - total[1]) / total[2])
    summary["paired_delta_wer_after_minus_before"] = summary["all"]["after_lora"]["wer"] - summary["all"]["before_lora"]["wer"]
    summary["hierarchical_bootstrap_95pct"] = list(map(float, np.quantile(estimates, [0.025, 0.975])))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all": summary["all"], "human_verified_only": summary["human_verified_only"],
                      "delta_wer": summary["paired_delta_wer_after_minus_before"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
