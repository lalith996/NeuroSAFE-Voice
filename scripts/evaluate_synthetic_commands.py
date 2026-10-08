"""Synthetic-voice integration check; never report as dysarthric-user accuracy."""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from assistant_core import AssistantCore
from whisper_baseline import edit_counts, normalize
from whisper_long_asr import ROOT, MODEL

PLAN = ROOT / "data/exports/command_recording_prompts.csv"
OUTPUT = ROOT / "data/results/assistant_core"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with PLAN.open(newline="", encoding="utf-8") as handle:
        prompts = list(csv.DictReader(handle))
    fold = 0
    selected = json.loads((ROOT / f"data/results/whisper_targeted_fold{fold}/selection.json").read_text())
    processor = WhisperProcessor.from_pretrained(MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
    model = PeftModel.from_pretrained(base, ROOT / selected["selected_adapter"]).to(device)
    model.eval()
    results = []
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for item in prompts:
            aiff = directory / f"{item['prompt_id']}.aiff"
            wav = directory / f"{item['prompt_id']}.wav"
            subprocess.run(["say", "-o", str(aiff), item["written_prompt"]], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff),
                            "-ar", "16000", "-ac", "1", str(wav)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
            started = time.perf_counter()
            audio, rate = sf.read(wav, dtype="float32")
            with torch.inference_mode():
                x = processor.feature_extractor(audio, sampling_rate=rate, return_tensors="pt",
                                                max_length=30 * rate, padding="max_length").input_features.to(device)
                tokens = model.generate(input_features=x, max_new_tokens=160, do_sample=False)
                if device == "mps":
                    torch.mps.synchronize()
            hypothesis = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            assistant = AssistantCore(contacts={"asha": "Asha", "sam": "Sam"})
            routed = assistant.handle(hypothesis, asr_confidence=None)
            seconds = time.perf_counter() - started
            ref, hyp = normalize(item["written_prompt"]), normalize(hypothesis)
            sub, deletion, insertion = edit_counts(ref, hyp)
            expected = None if item["intent"] == "none" else item["intent"]
            results.append({"prompt_id": item["prompt_id"], "written_prompt": item["written_prompt"],
                            "expected_intent": item["intent"], "asr_hypothesis": hypothesis,
                            "reference_words": len(ref), "word_errors": sub + deletion + insertion,
                            "observed_intent": routed["intent"] or "none",
                            "intent_correct": int(routed["intent"] == expected),
                            "decision_status": routed["status"],
                            "real_action_executed": int(bool(assistant.receipts)),
                            "wav_to_decision_seconds": round(seconds, 5)})
            print(f"{item['prompt_id']}: {hypothesis!r} -> {routed['intent'] or 'none'}", flush=True)
    with (OUTPUT / "synthetic_command_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    words = sum(r["reference_words"] for r in results)
    errors = sum(r["word_errors"] for r in results)
    latencies = sorted(r["wav_to_decision_seconds"] for r in results)
    summary = {"synthetic_voice": "macOS say; not dysarthric speech",
               "prompts": len(results), "prompt_words": words, "word_errors": errors,
               "prompt_wer": errors / words,
               "intent_correct": sum(r["intent_correct"] for r in results),
               "real_actions_executed": sum(r["real_action_executed"] for r in results),
               "warm_file_to_decision_median_seconds": statistics.median(latencies),
               "warm_file_to_decision_p95_seconds": latencies[int(.95 * (len(latencies) - 1))],
               "latency_scope": "loaded model, synthetic WAV read through simulated decision; excludes speech capture, model load, UI, and real tools"}
    (OUTPUT / "synthetic_command_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
