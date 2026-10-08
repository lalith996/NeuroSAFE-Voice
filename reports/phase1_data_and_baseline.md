# Phase 1 dataset preparation and zero-shot ASR baseline

## Data prepared

- TORGO: 9,889 speaker/session/utterance IDs across 15 speakers; 9,104 ASR-scoreable recordings and 7,158 filename-matched audio/EMA candidates. Of those AAI candidates, 2,414 are from speakers with dysarthria.
- AG500 format and head-audio timing checks identified 6,267 TORGO pairs within 20 ms, including 2,095 dysarthric pairs. Another 600 array-only pairs require alignment work, and 291 head-audio pairs have timing mismatches. Format/timing validation does not replace sensor-quality screening.
- MOCHA-TIMIT: 1,380 inventoried IDs across the two primary speakers and supplementary `maps0`; 919 primary-speaker audio/EMA pairs decoded with no parser errors. `fsew0` has 400/30/30 train/validation/test pairs; `msak0` has 399/30/30 after excluding its publisher-noted corrupt recording.
- TORGO uses a fixed four-fold speaker-disjoint split. Each speaker is in the test set once, with two dysarthric speakers in each test fold.
- TORGO flags include 2,712 IDs without `.pos`, 395 without a prompt, 387 with an unusable prompt, and 23 without valid selected audio. Flags can overlap.
- The decoded MOCHA audio exceeds EMA track duration by 0.384 seconds at the median. Original EMA timestamps were retained; no forced time alignment or coordinate normalization was applied.

## Whisper baseline method

- Model: [`mlx-community/whisper-small.en-mlx`](https://huggingface.co/mlx-community/whisper-small.en-mlx) at revision `52a88bf6e98b114a210c21bb83e22d6e1505cb73` using `mlx-whisper` 0.4.3 locally on Apple Silicon.
- Zero-shot: no TORGO examples were used to train or prompt the model. English transcription, temperature 0, no previous-text context. Valid head-microphone audio was selected first, with array-microphone fallback.
- WER uses total substitutions + deletions + insertions divided by total reference words. Text is lowercased, apostrophes removed, punctuation split, and ASCII letters/digits tokenized. All 9,104 eligible TORGO recordings were scored once.
- References are TORGO *elicitation prompts*, not manually checked verbatim transcripts. Bracketed instructions, image prompts, `xxx`, missing prompts, and a malformed prompt were excluded. Inline pronunciation guidance was removed from the scoring reference.

## Results

| Group | Speakers | Recordings | Reference words | WER | 95% speaker-bootstrap interval |
|---|---:|---:|---:|---:|---:|
| Dysarthric | 8 | 3,150 | 7,861 | 71.3% | 37.4%–110.7% |
| Control | 7 | 5,954 | 15,123 | 11.9% | 10.0%–14.7% |

| Speaker | Group | Recordings | WER | 95% utterance-bootstrap interval |
|---|---|---:|---:|---:|
| F01 | dysarthric | 118 | 72.7% | 64.0%–82.0% |
| F03 | dysarthric | 546 | 41.4% | 28.0%–65.9% |
| F04 | dysarthric | 429 | 15.5% | 12.6%–18.6% |
| FC01 | control | 151 | 10.0% | 6.5%–14.2% |
| FC02 | control | 1,221 | 8.7% | 7.4%–10.0% |
| FC03 | control | 964 | 11.2% | 9.7%–12.9% |
| M01 | dysarthric | 372 | 113.7% | 80.0%–158.5% |
| M02 | dysarthric | 388 | 83.0% | 66.5%–111.6% |
| M03 | dysarthric | 406 | 8.0% | 5.7%–11.0% |
| M04 | dysarthric | 391 | 169.8% | 113.9%–243.5% |
| M05 | dysarthric | 500 | 86.6% | 58.7%–124.9% |
| MC01 | control | 1,078 | 11.1% | 9.6%–13.1% |
| MC02 | control | 677 | 19.9% | 17.3%–22.7% |
| MC03 | control | 870 | 12.2% | 10.4%–14.1% |
| MC04 | control | 993 | 11.7% | 10.1%–13.4% |

Across all speakers, references of 1–3 words have WER 64.6%; references of 4 or more words have WER 18.4%. WER can exceed 100% when a short prompt receives many inserted words.

## Interpretation and remaining validation

These are baseline recognition measurements, not results for the proposed personalized assistant. Prompt-reference WER may count a participant's deviation from the written prompt as an ASR error. The dysarthric/control comparison is descriptive: prompt mix, speaker characteristics, and microphone availability differ. Review a sample of recordings and references before comparing this figure with published WER values or the project's target. The bootstrap intervals are descriptive for this small set of speakers and repeated prompts, not population guarantees.

The MOCHA files are decoded raw pairs ready for feature design. Train-only coordinate normalization and a chosen audio-to-EMA alignment policy remain to be implemented. TORGO `.pos` files are parsed and checked for the format/timing-verified subset; sensor-quality screening and alignment of the set-aside pairs remain before AAI training. The next model experiments can use the frozen folds and saved Whisper predictions as their reference baseline.

Sources: [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html), [LDC TAPADM manual](https://catalog.ldc.upenn.edu/docs/LDC2012S02/Manual.pdf), [MOCHA-TIMIT](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html), [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
