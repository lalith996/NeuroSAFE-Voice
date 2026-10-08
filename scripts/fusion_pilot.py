"""M2/M3 speaker-held-out Whisper fusion pilot using inferred AAI.

M2 adds a learned AAI projection to the frozen M1 Whisper encoder sequence.
M3 uses cross-attention from Whisper states to inferred AAI plus a learned gate.
All AAI inference uses a fold-specific model fitted on ASR training speakers
only, keeping validation and test speakers' EMA targets out of the AAI weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

from train_mocha_ridge_aai import context
from whisper_baseline import edit_counts, normalize
from whisper_lora_pilot import inputs

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
AAI_RESULTS = ROOT / "data/results/torgo_mocha_transfer"
SEED = 2026
SECONDS = 6
MODEL = "openai/whisper-small.en"


class AdditiveFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.project = nn.Linear(12, hidden_size)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, hidden, aai):
        return hidden + self.project(aai)


class NoFusion(nn.Module):
    def forward(self, hidden, aai):
        return hidden


class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.project = nn.Linear(12, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, aai):
        source = self.project(aai)
        attended, _ = self.attention(hidden, source, source, need_weights=False)
        weight = torch.sigmoid(self.gate(torch.cat([hidden, attended], dim=-1)))
        return hidden + self.output(weight * attended)


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aai_for(row, index, model):
    key = row["speaker"], row["session"], row["utterance_id"]
    with np.load(ROOT / index[key]["feature_path"]) as item:
        mel = item["log_mel"].astype(np.float64)
    x = context(mel)
    predicted = np.empty((len(mel), 12), dtype=np.float32)
    for j in range(12):
        predicted[:, j] = ((((x - model["x_mean"][j]) / model["x_std"][j])
                            @ model["weights"][j]) * model["y_std"][j] + model["y_mean"][j])
    predicted -= predicted.mean(axis=0, keepdims=True)
    return predicted


def make_aai_cache(rows, fold):
    with np.load(AAI_RESULTS / f"fold{fold}_torgo_only_train_only.npz") as item:
        model = {name: item[name] for name in item.files}
    index = {(r["speaker"], r["session"], r["utterance_id"]): r
             for r in read(PROCESSED / "torgo_oof_aai_index.csv")}
    raw = {r["audio_path"]: aai_for(r, index, model) for r in rows}
    train = [raw[r["audio_path"]] for r in rows if r["role"] == "train"]
    mean = np.concatenate(train).mean(axis=0)
    std = np.maximum(np.concatenate(train).std(axis=0), 1e-4)
    result = {}
    for path, values in raw.items():
        normalized = np.clip((values - mean) / std, -5, 5)
        padded = np.zeros((SECONDS * 100, 12), dtype=np.float32)
        padded[:len(values)] = normalized
        result[path] = torch.from_numpy(padded.reshape(SECONDS * 50, 2, 12).mean(axis=1))
    return result, mean, std


def validation_loss(backbone, fusion, rows, cached, aai, device):
    fusion.eval()
    losses = []
    with torch.no_grad():
        for row in rows:
            hidden, labels = cached[row["audio_path"]]
            fused = fusion(hidden.to(device, dtype=torch.float32).unsqueeze(0), aai[row["audio_path"]].to(device).unsqueeze(0))
            loss = backbone(encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                            labels=labels.to(device).unsqueeze(0)).loss
            losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def decode(backbone, fusion, rows, cached, aai, device, method):
    fusion.eval()
    processor = WhisperProcessor.from_pretrained(MODEL)
    results = []
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            hidden, _ = cached[row["audio_path"]]
            fused = fusion(hidden.to(device, dtype=torch.float32).unsqueeze(0), aai[row["audio_path"]].to(device).unsqueeze(0))
            tokens = backbone.generate(encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                                       max_new_tokens=48, do_sample=False)
            text = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            ref, hyp = normalize(row["reference"]), normalize(text)
            sub, deletion, insertion = edit_counts(ref, hyp)
            results.append({"method": method, "speaker": row["speaker"], "session": row["session"],
                            "utterance_id": row["utterance_id"], "reference": row["reference"],
                            "hypothesis": text, "reference_words": len(ref),
                            "substitutions": sub, "deletions": deletion, "insertions": insertion})
            if number % 20 == 0 or number == len(rows):
                print(f"  {method} decoded {number}/{len(rows)}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    fold = args.fold
    source = ROOT / f"data/results/whisper_small_lora_fold{fold}_pilot"
    output = ROOT / f"data/results/whisper_fusion_fold{fold}_pilot"
    output.mkdir(parents=True, exist_ok=True)
    rows = read(source / "selected_utterances.csv")
    train = [r for r in rows if r["role"] == "train"]
    validation = [r for r in rows if r["role"] == "validation"]
    test = [r for r in rows if r["role"] == "test"]
    aai, aai_mean, aai_std = make_aai_cache(rows, fold)
    processor = WhisperProcessor.from_pretrained(MODEL)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL)
    base.config.max_source_positions = SECONDS * 50
    base.model.encoder.embed_positions = nn.Embedding.from_pretrained(
        base.model.encoder.embed_positions.weight[:SECONDS * 50].clone(), freeze=True)
    backbone = PeftModel.from_pretrained(base, source / "adapter")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    print(f"Caching M1 encoder features for {len(rows)} selected clips", flush=True)
    cached = {}
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            x, labels = inputs(row, processor)
            hidden = backbone.base_model.model.model.encoder(input_features=x.to(device).unsqueeze(0)).last_hidden_state
            cached[row["audio_path"]] = hidden[0].to("cpu", dtype=torch.float16), labels
            if number % 100 == 0 or number == len(rows):
                print(f"  cached {number}/{len(rows)}", flush=True)
    results = decode(backbone, NoFusion(), test, cached, aai, device, "M1_cached_control")
    config = {"fold": fold, "steps_each": args.steps, "m1_adapter": str((source / "adapter").relative_to(ROOT)),
              "aai_model": f"fold{fold}_torgo_only_train_only.npz",
              "aai_train_only_normalization_mean": aai_mean.tolist(),
              "aai_train_only_normalization_std": aai_std.tolist(),
              "train_utterances": len(train), "validation_utterances": len(validation), "test_utterances": len(test),
              "target_status": "TORGO prompt-derived working targets",
              "note": "Inferred AAI is derived from audio; these ablations test representation/fusion, not a new measured modality."}
    for method, fusion in (("M2_additive", AdditiveFusion(base.config.d_model)),
                           ("M3_gated_cross_attention", GatedCrossAttention(base.config.d_model))):
        fusion.to(device)
        method_rng = random.Random(SEED)
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=2e-4)
        initial_loss = validation_loss(backbone, fusion, validation, cached, aai, device)
        started = time.monotonic()
        fusion.train()
        losses = []
        for step in range(1, args.steps + 1):
            row = method_rng.choice(train)
            hidden, labels = cached[row["audio_path"]]
            h = hidden.to(device, dtype=torch.float32).unsqueeze(0)
            z = aai[row["audio_path"]].to(device).unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            fused = fusion(h, z)
            loss = backbone(encoder_outputs=BaseModelOutput(last_hidden_state=fused),
                            labels=labels.to(device).unsqueeze(0)).loss
            loss.backward()
            nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 20 == 0 or step == args.steps:
                print(f"  {method} step {step}/{args.steps}, loss {np.mean(losses[-20:]):.3f}", flush=True)
        final_loss = validation_loss(backbone, fusion, validation, cached, aai, device)
        torch.save(fusion.cpu().state_dict(), output / f"{method}.pt")
        fusion.to(device)
        predictions = decode(backbone, fusion, test, cached, aai, device, method)
        results.extend(predictions)
        config[method] = {"initial_validation_loss": initial_loss,
                          "final_validation_loss": final_loss,
                          "train_last20_loss": float(np.mean(losses[-20:])),
                          "train_seconds": time.monotonic() - started,
                          "trainable_parameters": sum(p.numel() for p in fusion.parameters())}
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for method in ("M1_cached_control", "M2_additive", "M3_gated_cross_attention"):
        group = [r for r in results if r["method"] == method]
        words = sum(r["reference_words"] for r in group)
        errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in group)
        if method == "M1_cached_control":
            config[method] = {}
        config[method]["prompt_wer"] = errors / words
        config[method]["errors"] = errors
        config[method]["reference_words"] = words
    (output / "summary.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: config[name]["prompt_wer"] for name in ("M1_cached_control", "M2_additive", "M3_gated_cross_attention")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
