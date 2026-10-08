"""Score locally saved command recordings with the fixed Phase 1 ASR adapter.

Written prompts are the project targets. Optional actual_spoken_text is scored
separately as a verbatim reference. This evaluator makes no real-world actions.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from assistant_core import AssistantCore
from whisper_baseline import edit_counts, normalize
from whisper_long_asr import MODEL, ROOT


def score(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return len(ref), sum(edit_counts(ref, hyp))


def aggregate(rows: list[dict]) -> dict:
    words = sum(row["prompt_words"] for row in rows)
    errors = sum(row["prompt_errors"] for row in rows)
    command_rows = [row for row in rows if row["expected_intent"] != "none"]
    entity_rows = [row for row in command_rows if row["entity_scored"]]
    negative_rows = [row for row in rows if row["expected_intent"] == "none"]
    verbatim_rows = [row for row in rows if row["verbatim_words"]]
    verbatim_words = sum(row["verbatim_words"] for row in verbatim_rows)
    return {
        "clips": len(rows), "speakers": len({row["speaker_id"] for row in rows}),
        "prompt_words": words, "prompt_errors": errors,
        "prompt_wer": errors / words if words else None,
        "intent_correct": sum(row["intent_correct"] for row in rows),
        "intent_accuracy": sum(row["intent_correct"] for row in rows) / len(rows) if rows else None,
        "entity_scored": len(entity_rows),
        "entity_correct": sum(row["entity_correct"] for row in entity_rows),
        "entity_accuracy": (sum(row["entity_correct"] for row in entity_rows) / len(entity_rows)
                            if entity_rows else None),
        "negative_controls": len(negative_rows),
        "false_action_intents": sum(row["observed_intent"] != "none" for row in negative_rows),
        "verbatim_scored_clips": len(verbatim_rows),
        "verbatim_wer": (sum(row["verbatim_errors"] for row in verbatim_rows) / verbatim_words
                         if verbatim_words else None),
        "median_wav_to_decision_seconds": (statistics.median(row["wav_to_decision_seconds"] for row in rows)
                                            if rows else None),
        "real_actions_executed": 0,
        "real_task_success": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/new_recordings/recordings.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/results/new_recordings")
    parser.add_argument("--session", help="Optional session ID, for example S2")
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--contacts-json", type=Path, default=ROOT / "data/fixtures/demo_contacts.json")
    args = parser.parse_args()
    if not args.manifest.exists():
        print(json.dumps({"status": "awaiting_recordings", "clips": 0,
                          "manifest": str(args.manifest)}, indent=2))
        return
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    if args.session:
        source = [row for row in source if row["session_id"] == args.session]
    if not source:
        print(json.dumps({"status": "awaiting_recordings", "clips": 0,
                          "manifest": str(args.manifest), "session": args.session}, indent=2))
        return
    contacts = json.loads(args.contacts_json.read_text(encoding="utf-8"))
    core = AssistantCore(contacts=contacts)
    selection = ROOT / f"data/results/whisper_targeted_fold{args.fold}/selection.json"
    adapter = (ROOT / json.loads(selection.read_text())["selected_adapter"] if selection.exists()
               else ROOT / f"data/results/whisper_full_prompt_fold{args.fold}/adapter")
    processor = WhisperProcessor.from_pretrained(MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PeftModel.from_pretrained(
        WhisperForConditionalGeneration.from_pretrained(MODEL).to(device), adapter).to(device)
    model.eval()
    results = []
    with torch.inference_mode():
        for item in source:
            path = Path(item["audio_path"])
            if not path.is_absolute():
                path = ROOT / path
            audio, rate = sf.read(path, dtype="float32")
            if rate != 16000 or audio.ndim != 1 or not 0 < len(audio) <= 30 * rate:
                raise ValueError(f"Invalid 16 kHz mono command clip: {path}")
            started = time.perf_counter()
            features = processor.feature_extractor(audio, sampling_rate=rate, return_tensors="pt",
                                                   max_length=30 * rate, padding="max_length")
            tokens = model.generate(input_features=features.input_features.to(device),
                                    max_new_tokens=160, do_sample=False)
            if device == "mps":
                torch.mps.synchronize()
            hypothesis = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            decision = core.handle(hypothesis, asr_confidence=None)
            seconds = time.perf_counter() - started
            expected = core.parse(item["written_prompt"])
            expected_intent = item["intent"]
            if (expected.name if expected else "none") != expected_intent:
                raise ValueError(f"Prompt/intent mismatch for {item['recording_id']}")
            observed = decision["intent"] or "none"
            prompt_words, prompt_errors = score(item["written_prompt"], hypothesis)
            actual = item.get("actual_spoken_text", "").strip()
            verbatim_words, verbatim_errors = score(actual, hypothesis) if actual else (0, 0)
            entity_scored = int(bool(expected and expected.arguments))
            entity_correct = int(bool(entity_scored and observed == expected_intent
                                      and decision["arguments"] == expected.arguments))
            results.append({
                "recording_id": item["recording_id"], "speaker_id": item["speaker_id"],
                "session_id": item["session_id"], "prompt_id": item["prompt_id"],
                "written_prompt": item["written_prompt"], "actual_spoken_text": actual,
                "asr_hypothesis": hypothesis, "prompt_words": prompt_words,
                "prompt_errors": prompt_errors, "verbatim_words": verbatim_words,
                "verbatim_errors": verbatim_errors, "expected_intent": expected_intent,
                "observed_intent": observed, "intent_correct": int(observed == expected_intent),
                "entity_scored": entity_scored, "entity_correct": entity_correct,
                "decision_status": decision["status"], "wav_to_decision_seconds": round(seconds, 5),
            })
            print(f"{item['recording_id']}: {hypothesis!r} -> {observed}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "utterances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = aggregate(results)
    summary.update({"status": "evaluated", "session": args.session, "fold": args.fold,
                    "adapter": str(adapter), "target_policy": "written prompts are project targets",
                    "scope": "local WAV through ASR and simulated intent; no real action execution or capture latency"})
    summary["by_speaker"] = {speaker: aggregate([row for row in results if row["speaker_id"] == speaker])
                             for speaker in sorted({row["speaker_id"] for row in results})}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
