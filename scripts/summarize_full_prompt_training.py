"""Compare full-prompt training with the frozen prior long-recording M1."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from whisper_baseline import edit_counts, normalize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/results/whisper_full_prompt_summary"
TARGETS = ROOT / "data/exports/torgo_long_test_targets.csv"


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["speaker"], row["session"], row["utterance_id"]


def metric(rows, stage):
    group = [r for r in rows if r["stage"] == stage]
    words = sum(int(r["reference_words"]) for r in group)
    errors = sum(int(r["errors"]) for r in group)
    return {"clips": len(group), "reference_words": words, "word_errors": errors,
            "prompt_wer": errors / words if words else None}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target_rows = read(TARGETS)
    targets = {key(r): r for r in target_rows}
    if len(targets) != 187:
        raise ValueError("Frozen long test set changed")
    results = []
    fold_info = {}
    for fold in range(4):
        folder = ROOT / f"data/results/whisper_full_prompt_fold{fold}"
        history = json.loads((folder / "training_history.json").read_text())
        fold_info[str(fold)] = {"best_step": history["best_step"],
                                "best_validation_mean_loss": history["best_validation_mean_loss"],
                                "unique_training_clips_sampled": history["unique_training_clips_sampled"],
                                "training_seconds": history["training_seconds"]}
        files = {"prior_m1": ROOT / f"data/results/whisper_long_fold{fold}/after_lora_predictions.csv",
                 "full_prompt_m1": folder / "test_predictions.csv"}
        fold_ids = {k for k, r in targets.items() if int(r["fold"]) == fold}
        for stage, path in files.items():
            source = read(path)
            if len(source) != len(fold_ids) or {key(r) for r in source} != fold_ids:
                raise ValueError(f"Fold {fold}, {stage} IDs differ from fixed test set")
            for row in source:
                target = targets[key(row)]
                if row["audio_path"] != target["audio_path"]:
                    raise ValueError(f"Audio changed for {key(row)}")
                ref, hyp = normalize(target["scoring_target"]), normalize(row["hypothesis"])
                sub, deletion, insertion = edit_counts(ref, hyp)
                results.append({"fold": fold, "speaker": row["speaker"], "session": row["session"],
                                "utterance_id": row["utterance_id"], "stage": stage,
                                "target_status": target["target_status"],
                                "reference_words": len(ref), "substitutions": sub,
                                "deletions": deletion, "insertions": insertion,
                                "errors": sub + deletion + insertion,
                                "hypothesis": row["hypothesis"]})
    with (OUTPUT / "utterances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    by_speaker = defaultdict(list)
    for row in results:
        by_speaker[row["speaker"]].append(row)
    summary = {"all": {stage: metric(results, stage) for stage in ("prior_m1", "full_prompt_m1")},
               "by_speaker": {speaker: {stage: metric(group, stage) for stage in ("prior_m1", "full_prompt_m1")}
                              for speaker, group in sorted(by_speaker.items())},
               "fold_training": fold_info,
               "reference_note": "user-designated written TORGO prompts; same frozen test clips and decoding"}
    before = {key(r): r for r in results if r["stage"] == "prior_m1"}
    after = {key(r): r for r in results if r["stage"] == "full_prompt_m1"}
    clip_changes = []
    paired = defaultdict(list)
    for clip, a in after.items():
        b = before[clip]
        paired[clip[0]].append((int(a["errors"]), int(b["errors"]), int(a["reference_words"])))
        clip_changes.append({"speaker": clip[0], "session": clip[1], "utterance_id": clip[2],
                             "reference_words": a["reference_words"],
                             "prior_errors": b["errors"], "full_prompt_errors": a["errors"],
                             "errors_saved": int(b["errors"]) - int(a["errors"]),
                             "prior_insertions": b["insertions"],
                             "full_prompt_insertions": a["insertions"]})
    with (OUTPUT / "clip_changes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clip_changes[0]))
        writer.writeheader()
        writer.writerows(sorted(clip_changes, key=lambda r: int(r["errors_saved"]), reverse=True))
    rng = np.random.default_rng(2033)
    speakers = sorted(paired)
    samples = []
    for _ in range(2000):
        chosen = []
        for speaker in rng.choice(speakers, size=len(speakers), replace=True):
            group = paired[speaker]
            chosen.extend(group[int(i)] for i in rng.integers(0, len(group), len(group)))
        total = np.sum(chosen, axis=0)
        samples.append((total[0] - total[1]) / total[2])
    summary["paired_delta_wer_full_minus_prior"] = (summary["all"]["full_prompt_m1"]["prompt_wer"]
                                                    - summary["all"]["prior_m1"]["prompt_wer"])
    summary["hierarchical_bootstrap_95pct"] = list(map(float, np.quantile(samples, [0.025, 0.975])))
    dominant = max(clip_changes, key=lambda r: int(r["errors_saved"]))
    summary["largest_single_clip_gain"] = dominant
    remaining = [r for r in results if key(r) != (dominant["speaker"], dominant["session"], dominant["utterance_id"])]
    summary["excluding_largest_gain"] = {stage: metric(remaining, stage)
                                         for stage in ("prior_m1", "full_prompt_m1")}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
