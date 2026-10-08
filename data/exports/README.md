# Project exports

All manifest paths are relative to the project root. The source corpora remain in `data/extracted/`; model training reads them there without making a second full copy.

The CSV and JSON exports are included in the public code repository. The review ZIPs listed below contain licensed TORGO audio and are retained locally only; regenerate them from a permitted local corpus copy with the scripts in `scripts/`.

- `torgo_asr_targets.csv`: 9,104 TORGO recordings with written prompts designated by the user as project ASR targets, audio warnings, and fixed four-fold speaker roles. `written_prompt` preserves the original text; `target_text` is the spoken-word scoring target. In 85 rows, bracketed pronunciation hints are removed from `target_text`. `target_status=project_prompt_target` records the user's choice. Word error rate against these targets is **prompt-reference WER**. The older `torgo_asr_working_targets.csv` path contains the same rows for compatibility.
- `torgo_long_test_targets.csv`: the 187 frozen, longer held-out TORGO test clips with their chosen prompt targets, fold IDs, audio paths, and target status. This is the direct CSV used by the long-test evaluation.
- `torgo_aai_candidates.csv`: 6,267 AG500 format and head-audio timing checked pairs with sensor review flags and fold roles. These are candidates, not fully adjudicated clean trajectories.
- `mocha_aai_pairs.csv`: 919 decoded MOCHA primary-speaker pairs and their fixed utterance roles.
- `torgo_transcript_review_180.zip`: portable local review pack with 180 WAV files, the review page, and queue CSV. Extract the ZIP and open `reports/transcript_review.html`. Export a completed review CSV after listening. Do not distribute corpus audio outside its permitted academic use.
- `torgo_long_test_transcript_review.zip`: 187 WAVs from the frozen longer held-out test, with a clip-ID-linked listening page and starter CSV. Extract the ZIP and open `reports/long_test_transcript_review.html` inside it. The page exports a completed review CSV for direct import and rescoring. None of these clips is yet marked human verified.
- `long_review_export_summary.json`: long review package count and SHA-256 checksum.
- `export_summary.json`: row counts and ZIP checksum.

The user-designated written prompts are the ASR training and scoring targets. Any later clip-specific listening decision is kept separately from that designation. For AAI, the target is an articulatory trajectory. The project WER goal is an evaluation criterion, not a training label.
