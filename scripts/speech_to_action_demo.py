"""Run one local WAV through ASR and the safe simulated action router."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from assistant_core import AssistantCore
from whisper_long_asr import ROOT, MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True, help="16 kHz mono WAV, at most 30 seconds")
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--contacts-json", type=Path, help="Optional local JSON map of spoken aliases to contact names")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat with the loaded model to measure warm processing")
    parser.add_argument("--output-json", type=Path, help="Optional path to save the result locally")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        raise ValueError("repeats must be between 1 and 20")
    started = time.perf_counter()
    audio, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or not 0 < len(audio) <= 30 * rate:
        raise ValueError("Expected a nonempty 16 kHz mono WAV of at most 30 seconds")
    contacts = json.loads(args.contacts_json.read_text()) if args.contacts_json else {}
    selection = ROOT / f"data/results/whisper_targeted_fold{args.fold}/selection.json"
    adapter = (ROOT / json.loads(selection.read_text())["selected_adapter"] if selection.exists()
               else ROOT / f"data/results/whisper_full_prompt_fold{args.fold}/adapter")
    processor = WhisperProcessor.from_pretrained(MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
    model = PeftModel.from_pretrained(base, adapter).to(device)
    model.eval()
    loaded_seconds = time.perf_counter() - started
    processing = []
    transcript = ""
    decision = {}
    with torch.inference_mode():
        for _ in range(args.repeats):
            run_started = time.perf_counter()
            x = processor.feature_extractor(audio, sampling_rate=rate, return_tensors="pt",
                                            max_length=30 * rate, padding="max_length").input_features.to(device)
            tokens = model.generate(input_features=x, max_new_tokens=160, do_sample=False)
            if device == "mps":
                torch.mps.synchronize()
            transcript = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            assistant = AssistantCore(contacts=contacts)
            decision = assistant.handle(transcript, asr_confidence=None)
            processing.append(time.perf_counter() - run_started)
    result = {"audio_path": str(args.audio.resolve()), "fold": args.fold,
              "adapter": str(adapter), "transcript": transcript,
              "decision": decision, "real_action_executed": False,
              "model_load_seconds": loaded_seconds,
              "processing_seconds_each_run": processing,
              "warm_processing_median_seconds": statistics.median(processing[1:]) if len(processing) > 1 else None,
              "file_to_all_decisions_seconds": time.perf_counter() - started,
              "latency_scope": "offline WAV load, model initialization, then repeat preprocessing + ASR + simulated decision; excludes speaking and microphone capture"}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
