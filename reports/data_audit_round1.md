# TORGO data audit: round 1

## What was checked

- Re-screened all 6,267 TORGO position tracks that had already passed AG500 format and head-audio duration checks.
- Measured exact flat channels and abrupt frame-to-frame position changes. The position stream is 200 Hz, so consecutive frames are 5 ms apart.
- Screened all 9,104 baseline audio recordings for low average level and samples at the 16-bit amplitude limit.
- Built a balanced listening queue of 180 ASR recordings: 12 per speaker, 90 short prompts and 90 longer prompts. The queue includes 21 array-microphone recordings, 31 with a sensor review flag, and 51 with an audio review flag.

## Sensor findings

| Finding | Tracks or channels |
|---|---:|
| Tracks screened | 6,267 |
| Tracks with at least one review flag | 1,024 |
| High priority review tracks | 159 |
| Medium priority review tracks | 865 |
| Tracks with a flat channel usually active in that speaker/session | 13 |
| Tracks with a position step greater than 20 recorded units in 5 ms | 1,014 |
| Channels usually flat in a speaker/session | 10 |

The 10 usually flat channels occur in specific M02, M04, and M05 sessions; see `data/processed/torgo_sensor_availability.csv`. Their flatness is recorded as sensor availability, not treated as a fault in every utterance. A large step is a candidate artifact, not proof of one. The threshold and priority ranking are triage heuristics and must be checked against plots or original recordings before any exclusion or smoothing decision.

The availability table is an **audit aid**, not a model-ready mask. Model normalization and any learned sensor masks must be fitted within each training fold to preserve speaker-disjoint evaluation.

## Audio findings

| Finding | Recordings |
|---|---:|
| Baseline audio screened | 9,104 |
| At least one audio review flag | 2,084 |
| Full-recording RMS below -45 dBFS | 367 |
| At least 0.1% of samples at the 16-bit limit | 1,717 |

The two flags did not overlap in this run. These thresholds are triage rules, not proof that speech is inaudible or distorted. Review by listening before changing any transcript, microphone selection, or inclusion decision. The audio audit includes the exact level and clipped-sample fraction for each recording.

## Transcript review procedure

Open `transcript_review.html` in a browser. For each item, play the audio and compare the words actually spoken with the written prompt and Whisper output. Mark one status, enter a verified transcript for corrected cases, add a note for uncertain or damaged audio, and export the CSV. The page stores a browser draft when storage is available, but the exported CSV is the durable record. The queue CSV is an unreviewed starter copy and is **not** verified ground truth.

The queue mixes random selections with deliberately high prompt-versus-Whisper disagreement. Its combined WER is therefore not a representative corpus estimate. Report WER on the reviewed random subset separately from targeted discrepancy cases, with the sample size and selection method stated. Only recordings actually listened to and adjudicated should receive `prompt_matches` or `corrected` status; the rest retain prompt-based, unverified labels.

Review statuses:

- `prompt_matches`: the spoken words match the written prompt; the prompt can be accepted after review.
- `corrected`: enter the words actually spoken in `verified_transcript`.
- `unintelligible`: speech is present but a reliable transcription cannot be made.
- `audio_issue`: missing, clipped, or otherwise unusable recording.
- `unreviewed`: no decision yet.

When the reviewed CSV is available, recalculate WER on reviewed transcripts and compare it with prompt-reference WER. Keep the reviewed sample separate from any model training. After sensor flags are adjudicated, create a versioned AAI training manifest with explicit inclusion decisions and reasons.

## Files

- `transcript_review.html` and `transcript_review_queue.csv`: listening interface and starting queue.
- `torgo_sensor_review.html`: visual triage of the 159 high-priority tracks, with peak-step plots, associated audio, and CSV export of review decisions. The plots cannot establish whether a motion is physically valid; inspect full trajectories before exclusion.
- `torgo_sensor_review_queue.csv`: all flagged position tracks, high priority first.
- `../data/processed/torgo_sensor_audit.csv`: all 6,267 track metrics and flags.
- `../data/processed/torgo_sensor_availability.csv`: speaker/session channel flatness rates.
- `../data/processed/torgo_sensor_audit_summary.json`: machine-readable counts and flag rules.
- `../data/processed/torgo_audio_audit.csv` and `../data/processed/torgo_audio_audit_summary.json`: waveform screening metrics and flag definitions.
