# NeuroSAFE-Voice: dysarthria voice assistant research prototype

This workspace prepares TORGO and MOCHA-TIMIT, checks audio/EMA alignment and sensor quality, and runs the proposal's first AAI and ASR ablations. It is a research prototype; no end-user voice assistant is deployed yet.

## Repository layout

| Location | Contents |
| --- | --- |
| `scripts/` | Data preparation, AAI and ASR training, evaluation, and local assistant programs. |
| `reports/` | Experiment reports, review pages, and the local command recorder page. |
| `data/fixtures/` | Fictional contacts and command examples for the assistant simulator. |
| `data/exports/` | Project target tables and pair lists. Review ZIPs with corpus audio stay local. |
| `data/processed/` | Manifests, quality audits, fixed splits, and feature indexes. Large feature arrays stay local. |
| `data/results/` | Metrics, predictions in CSV/JSON, trained adapters, and compact model checkpoints. Participant evaluation output stays local. |
| `output/` | Project presentation files. |

The original TORGO and MOCHA-TIMIT archives, extracted audio and EMA, generated feature arrays, review ZIPs, and participant recordings are **not in this public repository**. They remain in the same local `data/` paths so the scripts can be run after obtaining the corpora under their source terms. See [dataset locations and terms](data/README.md) and [preparation notes](data/processed/README.md). The included CSVs contain paths and research metadata, not the recordings themselves.

## Current findings

- TORGO: 9,104 recordings with usable written prompts designated by the user as project ASR targets; 6,267 head-audio/EMA pairs pass format and duration checks. The target CSV is `data/exports/torgo_asr_targets.csv`.
- MOCHA-TIMIT: 919 decoded audio/EMA pairs for the two primary speakers.
- Speaker-held-out TORGO AAI: per-recording acoustic normalization raised dysarthric mean channel Pearson *r* from 0.198 to 0.213. MOCHA pretraining and a tested temporal model did not add a useful gain. See `reports/aai_improvement_round2.md`.
- Four-fold Whisper Small pilot on 280 held-out dysarthric clips: paired prompt WER 127.4% before LoRA and 47.8% after LoRA. Inferred AAI fusion did not improve that result. See `reports/asr_aai_ablation_pilot.md`.
- Longer held-out recordings: 187 clips of 6–30 seconds from eight dysarthric speakers. Paired prompt WER fell from 68.2% to 50.2% after LoRA; the 20–25% target remains unmet. Their project targets are exported in `data/exports/torgo_long_test_targets.csv`. See `reports/long_recording_evaluation.md`.
- Broader prompt-pool LoRA training reduced aggregate WER on those same 187 clips to **39.8%**. Most of the 50.2% to 39.8% change comes from one M04 recording that previously triggered a repeated phrase; excluding it, WER changes from 40.5% to 39.7%. See `reports/whisper_full_prompt_round3.md`.
- A newly frozen 40-clip, five-per-speaker multiword test gives **32.8%** prompt WER for the broader-prompt model and **29.6%** for validation-selected multiword adaptation. The small-sample interval includes no improvement, and the 20–25% goal remains unmet. A local intent/safety simulator, command-recording page, and limited latency checks are ready. See `reports/phase1_continuation_fresh_test_and_assistant.md`.

## Open the local recorder

Double-click `Start NeuroSAFE-Voice.command`, or run `.venv/bin/python scripts/record_commands_server.py`, then open [the local recorder](http://127.0.0.1:8765/). Keep the server running while you record. Opening `reports/command_recorder.html` directly as a file cannot save audio; that page now explains the correct address.

The page captures one prompt at a time, saves a 16 kHz mono WAV, downloads the local recordings list as CSV, and can test a saved clip with the current ASR and safe intent simulator. Test results require confirmation and perform no real calls, messages, deletions, or device actions. The default demo contact names are fictional. See `reports/recording_protocol.md` for participant and evaluation instructions.

`.venv/bin/python scripts/evaluate_recorded_commands.py` produces clip-level and speaker-level reports for saved recordings; add `--session S2` once a separate evaluation session exists. See `data/results/new_recordings/` for the latest run. The evaluator reports `awaiting_recordings` only when the selected manifest or session has no clips.

**Project status:** the research data, target exports, baseline and adaptation experiments, local recorder, and simulated assistant flow are implemented. The proposal's WER and AAI accuracy targets remain unmet; real dysarthric-command accuracy, entity accuracy, task success, and full end-to-end latency cannot be established until consenting participant recordings and permitted real integrations are available.

## Main locations

- `data/exports/`: ASR prompt target CSVs, AAI pair lists, and recording review packages, including the 187-clip longer test set.
- `data/processed/`: data manifests, fixed speaker splits, quality flags, and derived AAI features. See `data/processed/README.md`.
- `data/results/`: Whisper and AAI predictions, model adapters, and numerical summaries.
- `reports/`: human-readable data audit and experiment reports.
- `scripts/`: reproducible preparation, training, evaluation, and export programs.

## Reproduce the latest experiments

Create a Python 3.12 environment and install the versions in `requirements-experiments.lock`. The downloaded corpus files must be available under `data/raw/` and `data/extracted/`. The launcher and examples below use a local `.venv` environment, which is not part of the repository.

```bash
.venv/bin/python scripts/map_torgo_sensors.py
.venv/bin/python scripts/prepare_torgo_aai_features.py
.venv/bin/python scripts/compare_aai_transfer.py
.venv/bin/python scripts/export_oof_aai_features.py
for fold in 0 1 2 3; do
  .venv/bin/python scripts/whisper_lora_pilot.py --model openai/whisper-small.en --fold "$fold" --steps 120
  .venv/bin/python scripts/fusion_pilot.py --fold "$fold" --steps 120
done
.venv/bin/python scripts/summarize_asr_ablations.py
```

The `whisper_lora_pilot.py` script saves the exact selected clips, model adapter, predictions, and metrics for each fold. Its six-second clip limit and 120-step training schedule are pilot choices. The written prompts are the project targets for training and prompt-reference WER; a larger test set remains useful before making a broad recognition claim.

The latest ASR training is `scripts/whisper_full_prompt_training.py`. It draws training clips from the full prompt-target CSV, holds out each test speaker, selects a checkpoint using validation speakers, and saves four adapters under `data/results/whisper_full_prompt_fold*/`. The paired results and error concentration are in `data/results/whisper_full_prompt_summary/`.

For the longer test, the saved fold adapters and predictions are in `data/results/whisper_long_fold*/`. The frozen test IDs are in `data/processed/torgo_long_asr_splits.csv`. The user-designated prompt targets already support training and evaluation. To make optional clip-specific corrections, extract `data/exports/torgo_long_test_transcript_review.zip`, open its `reports/long_test_transcript_review.html`, export the completed CSV, then run:

```bash
.venv/bin/python scripts/import_verified_transcripts.py --review-csv /absolute/path/to/torgo_long_transcript_review_completed.csv
.venv/bin/python scripts/rescore_long_asr.py
```

This rescoring uses saved model output and does not require retraining.

Corpus sources and terms: [TORGO publisher page](https://catalog.ldc.upenn.edu/LDC2012S02), [MOCHA-TIMIT publisher page](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html).
