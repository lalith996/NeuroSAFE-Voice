# Whisper LoRA and inferred AAI fusion: four-fold pilot

## What was run

The experiment follows the proposal's M0–M3 sequence on TORGO. Each of the eight dysarthric speakers was held out in one of four fixed folds. For each fold, a fresh `openai/whisper-small.en` model was loaded. M1 trained a rank-8 LoRA adapter on the other speakers (395 or 420 selected training clips, 120 steps), with 50 validation clips and 70 held-out dysarthric test clips. It changed 884,736 trainable parameters, about 0.37% of the 241.7 million parameters in the adapted model.

All clips were at most six seconds and had no audio audit flag. The 280 test clips were fixed before training, 35 from each dysarthric speaker. The 30-second Whisper input positions were shortened to the pretrained first six seconds for this pilot. References are TORGO elicitation prompts, designated by the user as the **project targets**. WER here means **prompt-reference WER**. The earlier listening checks had no saved clip IDs, so they do not create clip-specific corrections.

The paired M0 is the same Hugging Face Whisper Small model and decoding setup immediately before LoRA. It is separate from the earlier MLX Whisper Small zero-shot run over all 9,104 recordings; those results have different selection and decoding settings and should not be compared as a controlled ablation.

M2 froze M1 and trained an additive projection of inferred articulatory sequences. M3 froze M1 and trained cross-attention from Whisper encoder states to inferred articulatory sequences with a learned gate. Both used 120 training steps. Their AAI predictor was fitted only on that fold's ASR training speakers, then used to infer articulatory features for validation and test speakers from audio. Thus no measured EMA from a held-out speaker entered its fold's fusion training. The inferred features are derived from the same audio; this tests representation and fusion, not an independent sensor stream.

## Results on the same 280 test clips

| Variant | Word errors / 423 prompt words | Prompt WER |
|---|---:|---:|
| M0 Whisper Small before LoRA | 539 | 127.4% |
| M1 Whisper Small with LoRA | 202 | 47.8% |
| M2 M1 + inferred AAI projection | 203 | 48.0% |
| M3 M1 + gated AAI cross-attention | 204 | 48.2% |

The paired M1 reduction is 79.7 WER percentage points in this selected pilot. A hierarchical paired bootstrap over speakers and utterances gives a descriptive 95% interval of 35.5–146.9 points of reduction. Word error rate can exceed 100% when a one-word prompt elicits a long repeated output. M1 improved all eight held-out speakers; much of its gain is removal of such insertions on short prompts.

| Prompt length | Clips | M0 WER | M1 WER | M2 WER | M3 WER |
|---|---:|---:|---:|---:|---:|
| 1–3 words | 254 | 190.2% | 59.6% | 59.6% | 58.8% |
| 4+ words | 26 | 32.1% | 29.8% | 30.4% | 32.1% |

M2 adds one error and M3 adds two errors overall relative to M1. Their differences are too small and inconsistent to claim that articulatory fusion helps. The independent AAI check in `torgo_mocha_transfer_round1.md` also found a weak mean channel correlation (0.198 across dysarthric speakers), and the MOCHA prior improved it by only about 0.001. The proposal's decision rule therefore favors retaining M1 as the current ASR reference and improving the AAI branch before further fusion work. The proposal's 20–25% WER target is not met. M3 confidence calibration and full end-to-end latency were not evaluated in this pilot.

## Per-speaker prompt WER

| Test speaker | M0 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| F01 | 73.4% | 56.2% | 56.2% | 59.4% |
| F03 | 93.7% | 34.9% | 34.9% | 34.9% |
| F04 | 62.5% | 15.6% | 15.6% | 14.1% |
| M01 | 228.1% | 64.1% | 64.1% | 62.5% |
| M02 | 147.9% | 66.7% | 68.8% | 70.8% |
| M03 | 47.8% | 10.9% | 10.9% | 10.9% |
| M04 | 236.1% | 80.6% | 80.6% | 80.6% |
| M05 | 181.6% | 71.1% | 71.1% | 71.1% |

## Limits and next experiment

This is a feasibility pilot with a small fixed clip sample, 120 training steps per model, short-duration cropping, and prompt-derived references. Repeated prompts occur across different speakers, so the split tests unseen speakers rather than unseen text. The AAI feature extractor was trained on the fold's ASR training speakers; its train-side predictions are therefore in-sample while validation and test predictions are out-of-sample. A stronger AAI model and fully out-of-fold train features would make a cleaner fusion comparison.

Subsequent work tested a temporal AAI model and a larger, longer held-out ASR set. See `aai_improvement_round2.md` and `long_recording_evaluation.md`. Prompt targets are already available for continued training and evaluation. Optional clip-specific corrections, confidence calibration, and measured latency can refine later experiments.

## Artifacts

- `../data/results/whisper_small_lora_fold*_pilot/`: fixed clip lists, before/after predictions, LoRA adapters, and per-fold results.
- `../data/results/whisper_fusion_fold*_pilot/`: M1 cached control, M2/M3 predictions, trained fusion weights, and per-fold results.
- `../data/results/asr_ablation_summary.json`: paired totals, prompt-length breakdown, speaker scores, and bootstrap intervals.
- `../data/processed/torgo_oof_aai_index.csv`: sensor-free AAI estimate inventory. The fold-specific train-only AAI models used in fusion are saved separately in `../data/results/torgo_mocha_transfer/`.

Method references: [Hugging Face Whisper fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper), [Hugging Face PEFT adapter documentation](https://huggingface.co/docs/transformers/peft), [TORGO corpus documentation](https://catalog.ldc.upenn.edu/docs/LDC2012S02/README.txt).
