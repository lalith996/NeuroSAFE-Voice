# Full-prompt Whisper training: longer TORGO speaker-held-out test

## Question and method

Would broader training from the user's 9,104 TORGO written-prompt targets improve Whisper Small recognition on the frozen longer-recording test? The new run keeps the four speaker-disjoint folds, the 187 long test clips, the project prompt targets, the Whisper Small base model, rank-8 LoRA, and the previous 30-second test decoding. It changes training only.

For each fold, the eligible training pool comes from `../data/exports/torgo_asr_targets.csv` after retaining the fold's **train speakers**, 0.3–12-second recordings, and unflagged audio. The pool sizes are 2,992, 2,341, 3,958, and 4,609; these overlap across folds and do not mean that every source recording entered every model. Each fold ran 900 single-clip updates, sampling 75% from dysarthric speakers and 30% from recordings longer than six seconds. Sampling was approximately uniform over speakers within each stratum. The four runs actually sampled 738, 780, 757, and 723 unique training clips. This is broader than the earlier selected 599–682-clip pools, but still a sampled training pilot rather than full-corpus optimization.

The 300-, 600-, and 900-step checkpoints were compared by mean teacher-forced loss on each fold's fixed validation speakers. The chosen steps were 600, 900, 900, and 900. Test speakers were not used for checkpoint selection. The previous and new adapters were decoded on exactly the same 187 test recordings with the same `max_new_tokens=160` settings. Prompt-reference word error rate (WER) is the project's chosen measure.

## Held-out result

| Model | Word errors / 1,567 target words | Prompt WER |
|---|---:|---:|
| Whisper Small without LoRA | 1,069 | 68.2% |
| Earlier LoRA pilot on longer clips | 786 | 50.2% |
| New full-prompt-pool LoRA | 624 | **39.8%** |

The new run saves 162 errors, a 10.3 WER-point reduction from the earlier LoRA run. A descriptive hierarchical paired bootstrap over the eight fixed test speakers and clips gives a 95% interval of **−44.6 to +2.4 WER points** for the new-minus-old change. The interval includes no improvement. This test set has been examined over multiple development rounds, so the figures are research results, not a fresh untouched benchmark.

| Held-out speaker | Clips | Earlier LoRA | New LoRA |
|---|---:|---:|---:|
| F01 | 8 | 66.0% | 58.5% |
| F03 | 25 | 12.7% | 11.4% |
| F04 | 9 | 2.0% | 2.0% |
| M01 | 35 | 57.9% | 53.5% |
| M02 | 33 | 38.0% | 41.2% |
| M03 | 7 | 2.3% | 2.3% |
| M04 | 35 | 127.2% | 71.7% |
| M05 | 35 | 32.8% | 35.7% |

## Why the total gain is fragile

The earlier model repeated a phrase on **M04 / Session2 / 0291**, producing 148 inserted words and 157 total errors on a 12-word target. The new model made seven errors and no insertions on that clip. This single correction accounts for **150 of the 162 errors saved**. Excluding it from both models yields 629/1,555 errors (40.5% WER) for the earlier LoRA and 617/1,555 (39.7%) for the new LoRA. The other 186 clips therefore show only a 0.8-point WER reduction. Four speakers improve, two worsen, and two are unchanged.

The earlier model's remaining errors comprised 485 substitutions, 60 deletions, and 241 insertions; that one M04 clip contributed 148 of the insertions. The clip-level tables are in `../data/results/whisper_long_rescored/error_analysis.csv` and `../data/results/whisper_full_prompt_summary/clip_changes.csv`. These records identify exact speaker, session, and utterance IDs for follow-up work.

## Decoder check on validation speakers

The anti-repetition option was motivated by the M04 test output, so this is an exploratory follow-up. I compared the original decoder (`max_new_tokens=160`) with a bounded option (`max_new_tokens=64`, `no_repeat_ngram_size=3`) on the **same 233 validation clips** using the new adapters. The original had 304 errors in 889 prompt words; the guard had 305. On the 73 validation clips longer than six seconds, it was 188 versus 189 errors in 633 words. The guard showed no validation gain, so it was **not applied to the test set**. The numeric selection used validation scores, while the idea itself came from inspecting a test error.

## Decision

The new adapter family is the best aggregate result so far on this selected long test, at **39.8% prompt WER**. It does **not** meet the project's 20–25% goal, and its aggregate advantage over the earlier LoRA is concentrated in one recording. Use it as an experimental ASR candidate, retaining the prior adapter and clip-level outputs for comparison. The next recognition study should focus on the ordinary substitution errors of F01, M01, M02, M04, and M05, then measure any change on a newly frozen held-out sample. Additional AAI fusion is justified only if it beats this stronger audio-only reference on the same clips.

## Reproduction and outputs

- `../scripts/whisper_full_prompt_training.py`: reads the user-designated prompt CSV, trains four adapters, and selects checkpoints using validation loss.
- `../data/results/whisper_full_prompt_fold0/` through `whisper_full_prompt_fold3/`: adapters, training configurations and histories, test hypotheses, and fold summaries.
- `../scripts/summarize_full_prompt_training.py` and `../data/results/whisper_full_prompt_summary/`: paired clip results, speaker scores, bootstrap interval, and sensitivity excluding the largest gain.
- `../scripts/whisper_validate_decode_guard.py`: validation-only decoder comparison. Per-fold validation predictions and summaries are saved beside the adapters.

All targets are written TORGO prompts designated by the user as project targets. `written_prompt` preserves the original wording; scoring removes bracketed pronunciation hints where present. This experiment evaluates speech recognition only. Intent handling, confidence calibration, end-to-end latency, and voice assistant actions remain unevaluated.
