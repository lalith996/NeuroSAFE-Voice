# Fresh TORGO multiword test and local assistant prototype

## New test frozen before targeted training

`../data/processed/torgo_fresh_multiword_test_40.csv` fixes five previously unused recordings from each of the eight TORGO dysarthric speakers. Each has at least four target words and no audio audit flag. The selection excludes the prior 187 long test clips, the prior short pilot's test clips, and the 180-clip listening-review queue. The CSV SHA-256 and selection seed are in `../data/processed/torgo_fresh_multiword_test_40.json`. The test file was not opened by the targeted trainer or used to choose checkpoints.

This is a **multiword** test, not a balanced long-recording test: only nine of its 40 clips exceed six seconds, four from M04 and five from M05. TORGO has no remaining untested long clips for five of the eight dysarthric speakers, so a fresh balanced long test requires new recordings.

## Targeted multiword adaptation

Starting from the four broader-prompt Whisper Small LoRA adapters, I ran 600 more updates per fold. Training drew from the user-designated TORGO prompt CSV with 75% dysarthric-speaker draws and 60% clips whose target has at least four words. The folds sampled 476, 476, 502, and 513 unique training recordings. No validation or test speaker entered its fold's training pool. I compared each fold's starting adapter with the 300- and 600-step checkpoints on its fixed validation speakers, choosing the fewest multiword errors and retaining the starting adapter on a tie. The selected steps were **300, 0, 300, 600**. The fresh 40 clips were decoded only after all four choices were saved.

| Model on fresh 40 clips | Errors / 274 prompt words | Prompt WER |
|---|---:|---:|
| Broader-prompt LoRA | 90 | 32.8% |
| Validation-selected multiword LoRA | 81 | **29.6%** |

The paired reduction is 3.3 WER points. A descriptive speaker-and-clip bootstrap gives a 95% interval of **−10.3 to +1.4 points** for selected-minus-prior, which includes no improvement. This is a small, mixed-duration sample and does not demonstrate that the model meets the project's **20–25%** target. Its WER should not be directly compared with the earlier 187-clip result because the clip sets differ.

| Speaker | Clips | Prior | Selected |
|---|---:|---:|---:|
| F01 | 5 | 51.3% | 30.8% |
| F03 | 5 | 6.1% | 6.1% |
| F04 | 5 | 0.0% | 0.0% |
| M01 | 5 | 53.3% | 50.0% |
| M02 | 5 | 38.5% | 38.5% |
| M03 | 5 | 0.0% | 0.0% |
| M04 | 5 | 78.0% | 80.5% |
| M05 | 5 | 33.3% | 30.0% |

The selected adapter worsened M04 by one word and left three other speakers unchanged. The frozen test and full paired hypotheses are in `../data/results/torgo_fresh_multiword_test/`.

## Local intent, safety, and response checks

The local prototype in `../scripts/assistant_core.py` recognizes a small allowlisted set of time, calendar, app, light, call, message, and delete intents. It uses local contact aliases when supplied. Because ASR confidence is not calibrated, an audio-derived command requires confirmation; high-risk calls, messages, and deletes require explicit confirmation even if a confidence value is high. Saying “confirm” to ASR cannot authorize a pending action. An explicit confirmation control produces a **simulated receipt only**. No external action connector is installed.

Internal checks passed 26/26 authored text fixtures for intent and safe routing. None of 187 earlier TORGO hypotheses or 40 fresh TORGO hypotheses matched an action intent in the narrative-speech negative-control scans. A synthesized macOS voice reading the 16 command prompts and two non-command controls produced 18/18 intended intents, two word errors over 80 target words, and zero executed real actions. These are **software and synthetic-voice checks**, not dysarthric-user intent/entity accuracy or tool success rates. The proposed ≥90% entity and ≥92% zero-click targets remain unmeasured.

On the 40 fresh TORGO clips, loaded-model audio processing took a median **0.302 s** and p95 **0.382 s** from WAV read through ASR output; it excludes speaking, microphone capture, UI, and real tools. A separate synthesized-command smoke check took about **5.0 s to initialize** the model and a warm median of about **0.26 s** from WAV through the simulated decision. The proposal's ≤1.5 s **end-to-end** target has therefore not been validated.

## New recording kit

`../data/exports/command_recording_prompts.csv` contains 16 supported commands and two negative controls. `../scripts/record_commands_server.py` serves a local microphone page at `http://127.0.0.1:8765/`; it saves 16 kHz mono WAVs and a local manifest in `data/new_recordings/`. The recorder passed a temporary end-to-end GET/POST/WAV-conversion/manifest check using generated silence. Its sample names are fictional, and it makes no external upload.

The page now identifies the direct `file://` opening error, offers a CSV manifest download, and tests a saved recording through the ASR and safe simulator. The included macOS launcher opens the running localhost page. These are software-flow features; recognition on real dysarthric commands still needs participant recordings.

`../scripts/evaluate_recorded_commands.py` is ready to aggregate a recorded session by clip and speaker. It retains written prompts as project targets, optionally scores separately recorded spoken-word corrections, and reports intent/entity accuracy with no real actions. It returns `awaiting_recordings` when the selected manifest or session has no clips.

The capture and evaluation instructions are in `recording_protocol.md`. No participant recordings were created in this work. A person must provide the voice and the institution's required participant permission before a fresh human-command evaluation can be run. The written prompts remain project targets; any intentionally different spoken wording can be recorded separately in the new manifest.

## Reproduce and limits

- `../scripts/prepare_fresh_multiword_test.py`: one-time test freeze; refuses to overwrite existing IDs.
- `../scripts/whisper_targeted_multiword.py`: train/validation-only continuation and checkpoint selection.
- `../scripts/evaluate_fresh_multiword_test.py`: one-shot paired test and ASR processing-time summary.
- `../scripts/evaluate_assistant_core.py`, `../scripts/evaluate_synthetic_commands.py`, and `../scripts/speech_to_action_demo.py`: safe local software checks.
- `../data/results/assistant_core/`: fixture, synthetic-command, and smoke-check outputs.

The AAI branch's held-out dysarthric correlation is still only 0.213, and prior inferred-AAI fusion did not improve ASR. It was not added to this stronger audio-only experiment. Live dysarthric command recognition, real entity resolution, zero-click task success, false real actions, and end-to-end latency remain open until suitable participant recordings and permitted tool integrations exist.
