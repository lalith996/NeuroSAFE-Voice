# Local recording plan for a fresh assistant-command evaluation

The current TORGO long recordings are too sparse for another balanced long-speech test: five of eight dysarthric speakers have no unused clips longer than six seconds outside the existing test. The new frozen TORGO multiword set has five unused recordings per speaker, but it is a small check. New command recordings are needed for a fresh end-to-end voice-assistant claim.

## Capture plan

- Use `../data/exports/command_recording_prompts.csv`: 16 supported commands across low, medium, and high risk, plus two non-command controls. Names and rooms are **fictional examples**; replace them in a private local context file only if participants consent.
- Each consenting participant records the 18 prompts in two sessions on different days. Keep session 2 locked for evaluation until the recognition and intent rules are fixed. Record the prompt as spoken once per session; repeat only when a clip is interrupted or technically faulty, and document the replacement.
- Aim for at least eight speakers with dysarthria across the same broad range of speech severity the project intends to serve. These are new human recordings, so the software cannot create them on the user's behalf.
- Obtain the institution's required participant permission and follow its data-handling process. Use pseudonymous speaker IDs in the local manifest. Audio remains on this computer unless the participant and institution permit transfer.
- The written text is the intended command. If a participant deliberately says different words, record those words in `actual_spoken_text` before calculating verbatim WER. The intent label can still reflect the intended command.

## Local recorder

Double-click `../Start NeuroSAFE-Voice.command`, or run `.venv/bin/python scripts/record_commands_server.py`, then open `http://127.0.0.1:8765/` in a browser on this computer. The saved HTML file alone cannot record. Enter a pseudonymous speaker ID and session ID, acknowledge permission, and use Start/Stop on each prompt. Recordings are normalized to 16 kHz mono WAV under `data/new_recordings/` with a CSV manifest. The page can download that manifest as CSV. It does not bundle audio into the CSV. The page and server use only localhost; the recorder never sends audio to an external service.

After saving a clip, **Test recognition** runs the current selected ASR adapter and safe intent simulator on that clip. The page displays the recognized words and decision. Each test starts the model, so the first result can take several seconds. All decisions remain simulated; no real call, message, deletion, or device action occurs. Asha and Sam are fictional demo aliases, not contact details.

After recordings have been collected, run `.venv/bin/python scripts/evaluate_recorded_commands.py --session S2` for a session-2 report. It writes per-clip results and speaker summaries under `data/results/new_recordings/`. This uses written prompts as the project targets. If `actual_spoken_text` is filled for a clip, it also reports a separate verbatim WER; it does not replace the prompt target. The evaluator returns `awaiting_recordings` when no clips are present. It reports simulated intent/entity accuracy, not real task success.

The browser must grant microphone access. The page makes an explicit recording request for each clip and shows when recording is active. The server has no real call, message, home-control, or delete connector.

## Evaluation gate

After collection, keep participants and sessions separate according to the research question. Report WER by speaker, command intent accuracy, entity accuracy, task success, false-action rate, and end-to-end latency. A command-only prompt set cannot replace the TORGO sentence benchmark. The current local intent prototype uses synthetic fixtures and simulated receipts, so it cannot yet establish the proposal's real-user entity or execution targets.
