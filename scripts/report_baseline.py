"""Validate finished baseline results and write a concise Phase 1 report."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "utterance_manifest.csv"
RESULTS = ROOT / "data" / "results" / "whisper_small_en_zero_shot"
REPORT = ROOT / "reports" / "phase1_data_and_baseline.md"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(rows: list[dict]) -> tuple[int, int, float]:
    words = sum(int(row["reference_words"]) for row in rows)
    errors = sum(int(row["substitutions"]) + int(row["deletions"]) + int(row["insertions"]) for row in rows)
    return errors, words, errors / words


def utterance_bootstrap(rows: list[dict], seed: int, replicates: int = 2000) -> tuple[float, float]:
    words = np.array([int(row["reference_words"]) for row in rows], dtype=np.float64)
    errors = np.array([int(row["substitutions"]) + int(row["deletions"]) + int(row["insertions"]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(replicates):
        sample = rng.integers(0, len(rows), size=len(rows))
        ratios.append(errors[sample].sum() / words[sample].sum())
    return tuple(float(value) for value in np.quantile(ratios, [0.025, 0.975]))


def speaker_bootstrap(groups: dict[str, list[dict]], seed: int, replicates: int = 10000) -> tuple[float, float]:
    metrics = [metric(rows)[:2] for rows in groups.values()]
    values = np.array(metrics, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    sampled = values[indices].sum(axis=1)
    ratios = sampled[:, 0] / sampled[:, 1]
    return tuple(float(value) for value in np.quantile(ratios, [0.025, 0.975]))


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    manifest = read_csv(MANIFEST)
    expected = {(row["speaker"], row["session"], row["utterance_id"]) for row in manifest if row["dataset"] == "TORGO" and row["asr_eligible"] == "1"}
    utterances = read_csv(RESULTS / "utterances.csv")
    actual = {(row["speaker"], row["session"], row["utterance_id"]) for row in utterances}
    if actual != expected or len(actual) != len(utterances):
        raise RuntimeError(f"Baseline is incomplete or duplicated: {len(actual)} unique results for {len(expected)} eligible utterances")
    errors_path = RESULTS / "errors.csv"
    if errors_path.exists() and errors_path.stat().st_size:
        raise RuntimeError("Baseline contains inference errors; inspect errors.csv")
    config = json.loads((RESULTS / "run_config.json").read_text(encoding="utf-8"))
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in utterances:
        by_speaker[row["speaker"]].append(row)
    speaker_rows = []
    for speaker, rows in sorted(by_speaker.items()):
        _, words, wer = metric(rows)
        low, high = utterance_bootstrap(rows, 2026 + sum(map(ord, speaker)))
        speaker_rows.append({"speaker": speaker, "condition": "control" if "C" in speaker else "dysarthric",
                             "utterances": len(rows), "reference_words": words, "wer": wer,
                             "ci_low": low, "ci_high": high})
    ci_path = RESULTS / "speaker_wer_with_ci.csv"
    with ci_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(speaker_rows[0]))
        writer.writeheader()
        writer.writerows(speaker_rows)

    groups = {
        "dysarthric": {speaker: rows for speaker, rows in by_speaker.items() if "C" not in speaker},
        "control": {speaker: rows for speaker, rows in by_speaker.items() if "C" in speaker},
    }
    group_rows = []
    for condition, speakers in groups.items():
        rows = [row for group in speakers.values() for row in group]
        _, words, wer = metric(rows)
        low, high = speaker_bootstrap(speakers, 2026 if condition == "dysarthric" else 2027)
        group_rows.append({"condition": condition, "speakers": len(speakers), "utterances": len(rows),
                           "reference_words": words, "wer": wer, "ci_low": low, "ci_high": high})
    with (RESULTS / "group_wer_with_ci.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    torgo = [row for row in manifest if row["dataset"] == "TORGO"]
    mocha = [row for row in manifest if row["dataset"] == "MOCHA-TIMIT"]
    mocha_pairs = read_csv(ROOT / "data" / "processed" / "mocha_aai_pairs.csv")
    torgo_ema = read_csv(ROOT / "data" / "processed" / "torgo_aai_validation.csv")
    ema_status = Counter(row["status"] for row in torgo_ema)
    verified_dysarthric = sum(row["status"] == "verified_head_aligned" and row["condition"] == "dysarthric" for row in torgo_ema)
    flags = Counter(flag for row in torgo for flag in row["quality_flags"].split("|") if flag)
    duration_gaps = [float(row["duration_gap_seconds"]) for row in mocha_pairs]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    if not summary["complete"] or summary["completed_utterances"] != len(expected):
        raise RuntimeError("summary.json does not represent a complete baseline")

    lines = [
        "# Phase 1 dataset preparation and zero-shot ASR baseline",
        "",
        "## Data prepared",
        "",
        f"- TORGO: {len(torgo):,} speaker/session/utterance IDs across 15 speakers; {sum(int(row['asr_eligible']) for row in torgo):,} ASR-scoreable recordings and {sum(int(row['aai_pair_candidate']) for row in torgo):,} filename-matched audio/EMA candidates. Of those AAI candidates, {sum(int(row['aai_pair_candidate']) for row in torgo if row['condition'] == 'dysarthric'):,} are from speakers with dysarthria.",
        f"- AG500 format and head-audio timing checks identified {ema_status['verified_head_aligned']:,} TORGO pairs within 20 ms, including {verified_dysarthric:,} dysarthric pairs. Another {ema_status['array_only_alignment_needed']:,} array-only pairs require alignment work, and {ema_status['head_timing_mismatch']:,} head-audio pairs have timing mismatches. Format/timing validation does not replace sensor-quality screening.",
        f"- MOCHA-TIMIT: {len(mocha):,} inventoried IDs across the two primary speakers and supplementary `maps0`; {len(mocha_pairs):,} primary-speaker audio/EMA pairs decoded with no parser errors. `fsew0` has 400/30/30 train/validation/test pairs; `msak0` has 399/30/30 after excluding its publisher-noted corrupt recording.",
        "- TORGO uses a fixed four-fold speaker-disjoint split. Each speaker is in the test set once, with two dysarthric speakers in each test fold.",
        f"- TORGO flags include {flags['missing_ema']:,} IDs without `.pos`, {flags['missing_prompt']:,} without a prompt, {flags['unusable_prompt']:,} with an unusable prompt, and {flags['missing_valid_audio']:,} without valid selected audio. Flags can overlap.",
        f"- The decoded MOCHA audio exceeds EMA track duration by {np.median(duration_gaps):.3f} seconds at the median. Original EMA timestamps were retained; no forced time alignment or coordinate normalization was applied.",
        "",
        "## Whisper baseline method",
        "",
        f"- Model: [`{config['model_repo']}`](https://huggingface.co/{config['model_repo']}) at revision `{config['model_revision']}` using `mlx-whisper` {config['mlx_whisper_version']} locally on Apple Silicon.",
        "- Zero-shot: no TORGO examples were used to train or prompt the model. English transcription, temperature 0, no previous-text context. Valid head-microphone audio was selected first, with array-microphone fallback.",
        "- WER uses total substitutions + deletions + insertions divided by total reference words. Text is lowercased, apostrophes removed, punctuation split, and ASCII letters/digits tokenized. All 9,104 eligible TORGO recordings were scored once.",
        "- References are TORGO *elicitation prompts*, not manually checked verbatim transcripts. Bracketed instructions, image prompts, `xxx`, missing prompts, and a malformed prompt were excluded. Inline pronunciation guidance was removed from the scoring reference.",
        "",
        "## Results",
        "",
        "| Group | Speakers | Recordings | Reference words | WER | 95% speaker-bootstrap interval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(f"| {row['condition'].title()} | {row['speakers']} | {row['utterances']:,} | {row['reference_words']:,} | {percent(row['wer'])} | {percent(row['ci_low'])}–{percent(row['ci_high'])} |")
    lines.extend(["", "| Speaker | Group | Recordings | WER | 95% utterance-bootstrap interval |", "|---|---|---:|---:|---:|"])
    for row in speaker_rows:
        lines.append(f"| {row['speaker']} | {row['condition']} | {row['utterances']:,} | {percent(row['wer'])} | {percent(row['ci_low'])}–{percent(row['ci_high'])} |")
    lines.extend([
        "",
        f"Across all speakers, references of 1–3 words have WER {percent(summary['short_references_1_to_3_words']['wer'])}; references of 4 or more words have WER {percent(summary['longer_references_4_plus_words']['wer'])}. WER can exceed 100% when a short prompt receives many inserted words.",
        "",
        "## Interpretation and remaining validation",
        "",
        "These are baseline recognition measurements, not results for the proposed personalized assistant. Prompt-reference WER may count a participant's deviation from the written prompt as an ASR error. The dysarthric/control comparison is descriptive: prompt mix, speaker characteristics, and microphone availability differ. Review a sample of recordings and references before comparing this figure with published WER values or the project's target. The bootstrap intervals are descriptive for this small set of speakers and repeated prompts, not population guarantees.",
        "",
        "The MOCHA files are decoded raw pairs ready for feature design. Train-only coordinate normalization and a chosen audio-to-EMA alignment policy remain to be implemented. TORGO `.pos` files are parsed and checked for the format/timing-verified subset; sensor-quality screening and alignment of the set-aside pairs remain before AAI training. The next model experiments can use the frozen folds and saved Whisper predictions as their reference baseline.",
        "",
        "Sources: [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html), [LDC TAPADM manual](https://catalog.ldc.upenn.edu/docs/LDC2012S02/Manual.pdf), [MOCHA-TIMIT](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html), [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).",
        "",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
