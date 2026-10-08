# Longer TORGO recordings: speaker-held-out ASR evaluation

## What was evaluated

The test set contains 187 different recordings from all eight TORGO speakers with dysarthria. Every test clip lasts more than 6 and at most 30 seconds, has no audio-audit flag, and belongs to a speaker excluded from that fold's training and validation data. There are four fixed speaker-disjoint folds. Each fold trains a fresh rank-8 LoRA adapter for `openai/whisper-small.en` for 240 steps on selected clips from training speakers (at most 12 seconds). The unadapted and adapted models decode the **same full-length test audio** with the same 30-second input and decoding settings. The train, validation, and test IDs are saved in `../data/processed/torgo_long_asr_splits.csv`.

All 187 scoring references are TORGO written prompts, designated by the user as the **project targets**. The earlier listening checks have no saved recording IDs, so none is recorded as a clip-specific manual correction. The project metric here is prompt-reference WER. The target CSVs are `../data/exports/torgo_asr_targets.csv` for all 9,104 eligible recordings and `../data/exports/torgo_long_test_targets.csv` for this held-out test.

## Paired results

| Test set | Clips | Reference words | Whisper Small before LoRA | After LoRA |
|---|---:|---:|---:|---:|
| All four folds, project prompt targets | 187 | 1,567 | 1,069 errors; **68.2% WER** | 786 errors; **50.2% WER** |

The paired reduction is 18.1 WER percentage points. A descriptive hierarchical bootstrap across the eight fixed speakers and clips puts the change (after minus before) between **−69.4 and +6.1 points** at the 95% level. It includes no improvement, so this sample does not support a precise population-wide gain. One speaker, M04, accounts for 259 of the 283 errors saved. WER can exceed 100% when the model inserts many words beyond the reference.

| Speaker | Clips | Before | After |
|---|---:|---:|---:|
| F01 | 8 | 68.1% | 66.0% |
| F03 | 25 | 11.0% | 12.7% |
| F04 | 9 | 1.0% | 2.0% |
| M01 | 35 | 60.2% | 57.9% |
| M02 | 33 | 44.4% | 38.0% |
| M03 | 7 | 2.3% | 2.3% |
| M04 | 35 | 218.7% | 127.2% |
| M05 | 35 | 34.0% | 32.8% |

Five speakers improve, two worsen slightly, and one is unchanged. M04 remains particularly difficult. The project's 20–25% WER target is **not met** on this longer test set.

## Relationship to the short-clip pilot

The earlier pilot used 280 clips of at most six seconds, 120 LoRA steps per fold, and mostly prompts of one to three words. It reported 127.4% before LoRA and 47.8% after LoRA. This longer experiment changes clip selection, prompt length, training duration, and the full model input length; its WER values are **not a controlled short-versus-long comparison**. Within this longer experiment, the before/after comparison is paired on identical clips and references.

The earlier M2 and M3 inferred-articulation fusion did not beat LoRA on short clips. The improved AAI ridge model in `aai_improvement_round2.md` raises dysarthric held-out channel correlation from 0.198 to 0.213, still far below the proposed 0.70 and with no demonstrated ASR gain. This longer ASR test evaluates M1, not a renewed M2/M3 fusion claim.

## Optional clip-specific corrections

1. Extract `../data/exports/torgo_long_test_transcript_review.zip` and open `reports/long_test_transcript_review.html` **inside the extracted folder**. It includes the 187 audio files and a card for each speaker, session, and utterance ID.
2. Listen to a clip, enter a reviewer name, and choose **Prompt matches exactly**, **Corrected transcript** (type the spoken words), **Unintelligible**, or **Audio issue**. Export the completed review CSV from the page. A review can be submitted for only some clips; leave the rest unreviewed.
3. Import the CSV with `.venv/bin/python scripts/import_verified_transcripts.py --review-csv /absolute/path/to/torgo_long_transcript_review_completed.csv`, then run `.venv/bin/python scripts/rescore_long_asr.py`. The saved hypotheses are rescored without retraining. Reviewed corrections become human-verified targets; unintelligible or faulty clips are excluded from scored rows with their decision retained.

The current project targets remain valid without this optional review. The import validates clip IDs, prompt/audio identity, explicit status, and reviewer name before replacing a prompt target for a reviewed clip. `../data/processed/manual_verified_transcripts.csv` currently has no reviewed rows.

## Saved outputs and remaining limits

- `../data/results/whisper_long_fold0/` through `whisper_long_fold3/`: four adapters, before/after hypotheses, and fold summaries.
- `../data/results/whisper_long_rescored/utterances.csv` and `summary.json`: clip-level errors and pooled, speaker-level, and bootstrap summaries.
- `../reports/long_test_transcript_review.csv` and `.html`: starter review queue and local listening page.
- `../data/exports/torgo_long_test_transcript_review.zip`: portable 187-recording review package, SHA-256 recorded in `../data/exports/long_review_export_summary.json`.

This is a selected 187-clip evaluation, not a complete all-recording benchmark. The test speakers are unseen during adapter fitting, but many prompt texts recur across speakers; it tests speaker transfer rather than unseen wording. The written prompts are the project's chosen targets; individual recordings have not been transcribed independently. Model confidence, end-to-end assistant behavior, and latency were not evaluated here.
