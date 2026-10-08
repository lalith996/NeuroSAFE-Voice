"""Run a resumable, local zero-shot Whisper baseline on TORGO speech."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import snapshot_download
import mlx_whisper


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "utterance_manifest.csv"
MODEL_REPO = "mlx-community/whisper-small.en-mlx"
MODEL_REVISION = "52a88bf6e98b114a210c21bb83e22d6e1505cb73"
RESULTS = ROOT / "data" / "results" / "whisper_small_en_zero_shot"
FIELDS = [
    "speaker", "session", "utterance_id", "audio_path", "microphone",
    "duration_seconds", "reference", "hypothesis", "reference_words",
    "substitutions", "deletions", "insertions", "wer", "inference_seconds",
]


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("'", "")
    return re.findall(r"[a-z0-9]+", text)


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Return substitutions, deletions, insertions with deterministic tie breaking."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_word in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hypothesis, 1):
            diagonal = previous[j - 1]
            if ref_word == hyp_word:
                best = diagonal
            else:
                best = (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3])
            delete = previous[j]
            delete = (delete[0] + 1, delete[1], delete[2] + 1, delete[3])
            insert = current[j - 1]
            insert = (insert[0] + 1, insert[1], insert[2], insert[3] + 1)
            # Prefer substitution to deletion to insertion when costs tie.
            current.append(min((best, delete, insert), key=lambda item: item[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


def key(row: dict) -> tuple[str, str, str]:
    return row["speaker"], row["session"], row["utterance_id"]


def load_manifest() -> list[dict]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted((row for row in rows if row["dataset"] == "TORGO" and row["asr_eligible"] == "1"), key=key)


def read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict], total: int) -> dict:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)

    def metrics(group: list[dict]) -> dict:
        counts = {name: sum(int(row[name]) for row in group) for name in ("reference_words", "substitutions", "deletions", "insertions")}
        errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
        return {"utterances": len(group), **counts, "wer": errors / counts["reference_words"] if counts["reference_words"] else None}

    speaker_rows = [{"speaker": speaker, "condition": "control" if "C" in speaker else "dysarthric", **metrics(group)}
                    for speaker, group in sorted(by_speaker.items())]
    with (RESULTS / "speaker_wer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker", "condition", "utterances", "reference_words", "substitutions", "deletions", "insertions", "wer"])
        writer.writeheader()
        writer.writerows(speaker_rows)
    length_rows = []
    for speaker, group in sorted(by_speaker.items()):
        for bucket, subset in (
            ("1-3 reference words", [row for row in group if int(row["reference_words"]) <= 3]),
            ("4+ reference words", [row for row in group if int(row["reference_words"]) >= 4]),
        ):
            length_rows.append({"speaker": speaker, "condition": "control" if "C" in speaker else "dysarthric",
                                "prompt_length": bucket, **metrics(subset)})
    with (RESULTS / "speaker_wer_by_prompt_length.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker", "condition", "prompt_length", "utterances", "reference_words", "substitutions", "deletions", "insertions", "wer"])
        writer.writeheader()
        writer.writerows(length_rows)
    summary = {
        "completed_utterances": len(rows), "expected_utterances": total,
        "complete": len(rows) == total,
        "overall": metrics(rows),
        "dysarthric": metrics([row for row in rows if "C" not in row["speaker"]]),
        "control": metrics([row for row in rows if "C" in row["speaker"]]),
        "short_references_1_to_3_words": metrics([row for row in rows if int(row["reference_words"]) <= 3]),
        "longer_references_4_plus_words": metrics([row for row in rows if int(row["reference_words"]) >= 4]),
        "per_speaker": speaker_rows,
        "interpretation": "WER is against TORGO elicitation prompts, which were not independently verified as verbatim transcripts.",
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=None, help="For a short pilot; rerun without it to resume all utterances")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    config_path = RESULTS / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["manifest_sha256"] != manifest_hash:
            raise RuntimeError("Manifest changed after this baseline started; use a fresh result directory")
        model_path = Path(config["model_snapshot"])
        if config.get("evaluation_scope") != "all eligible TORGO speakers":
            config["evaluation_scope"] = "all eligible TORGO speakers"
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    else:
        model_path = Path(snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION))
        config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_repo": MODEL_REPO, "model_snapshot": str(model_path),
            "model_revision": model_path.name,
            "mlx_whisper_version": version("mlx-whisper"),
            "manifest_sha256": manifest_hash,
            "decode": {"language": "en", "task": "transcribe", "temperature": 0.0, "condition_on_previous_text": False},
            "normalization": "NFKD ASCII, lowercase, remove apostrophes, tokenize contiguous a-z/0-9",
            "audio_selection": "TORGO head microphone when valid; array microphone fallback",
            "reference_type": "TORGO elicitation prompt, not independently corrected verbatim transcript",
            "evaluation_scope": "all eligible TORGO speakers",
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if not model_path.exists():
        raise RuntimeError(f"Model snapshot unavailable: {model_path}")

    targets = load_manifest()
    results_path = RESULTS / "utterances.csv"
    completed = read_existing(results_path)
    done = {key(row) for row in completed}
    if len(done) != len(completed):
        raise RuntimeError("Duplicate rows found in baseline result")
    pending = [row for row in targets if key(row) not in done]
    if args.max_new is not None:
        pending = pending[:args.max_new]
    print(f"Whisper baseline: {len(completed)} completed, {len(pending)} pending in this run, {len(targets)} total", flush=True)

    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
            handle.flush()
        for index, row in enumerate(pending, 1):
            audio = ROOT / row["audio_path"]
            start = time.monotonic()
            try:
                result = mlx_whisper.transcribe(
                    str(audio), path_or_hf_repo=str(model_path),
                    language="en", task="transcribe", temperature=0.0,
                    condition_on_previous_text=False, verbose=None,
                )
            except Exception as exc:
                with (RESULTS / "errors.csv").open("a", newline="", encoding="utf-8") as errors:
                    csv.writer(errors).writerow([*key(row), type(exc).__name__, str(exc)])
                print(f"ERROR {key(row)}: {exc}", flush=True)
                continue
            reference = row["scoring_reference"]
            hypothesis = result["text"].strip()
            ref_words, hyp_words = normalize(reference), normalize(hypothesis)
            substitutions, deletions, insertions = edit_counts(ref_words, hyp_words)
            output = {
                "speaker": row["speaker"], "session": row["session"],
                "utterance_id": row["utterance_id"], "audio_path": row["audio_path"],
                "microphone": "head" if row["audio_path"] == row["audio_head_path"] else "array",
                "duration_seconds": row["duration_seconds"], "reference": reference,
                "hypothesis": hypothesis, "reference_words": len(ref_words),
                "substitutions": substitutions, "deletions": deletions,
                "insertions": insertions,
                "wer": (substitutions + deletions + insertions) / len(ref_words),
                "inference_seconds": round(time.monotonic() - start, 5),
            }
            writer.writerow(output)
            handle.flush()
            completed.append({name: str(value) for name, value in output.items()})
            if index % 100 == 0 or index == len(pending):
                print(f"Completed {len(completed)}/{len(targets)}; latest {row['speaker']} {row['session']} {row['utterance_id']}", flush=True)
    summary = summarize(completed, len(targets))
    print(f"Completed {summary['completed_utterances']}/{summary['expected_utterances']}; corpus WER {summary['overall']['wer']:.3f}", flush=True)


if __name__ == "__main__":
    main()
