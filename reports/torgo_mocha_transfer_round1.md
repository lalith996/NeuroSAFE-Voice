# TORGO sensor mapping and first cross-corpus AAI comparison

## Prepared material

- Mapped the 12 default TORGO coils using the publisher's sensor key. Speech targets use lower incisor/jaw, upper lip, lower lip, tongue tip, tongue middle, and tongue back. The released `.pos` tracks are already head corrected.
- Parsed the publisher's `ERRORS.xls`: 663 annotation rows yielded 467 unique trial matches, 150 ambiguous trial matches, 35 unmatched rows, five ranges kept for review, and six notes that safely applied to a single available session. Ambiguous notes were not used to mask a particular recording.
- Prepared 6,267 head-audio/EMA pairs at 100 frames per second, totaling 1,995,333 frames. The initial pair check establishes file format and duration agreement within 20 ms. It does not establish that each sensor is physically correct.
- Masked each unavailable or suspect articulator channel separately, based on session-wide flat tracks, recording-level flat tracks or large steps, and unambiguous publisher fault notes. This preserves usable lip tracks in recordings whose tongue coils failed. Seven dysarthric and seven control speakers have paired EMA; F01 has no validated `.pos` pairs.
- Saved 9,104 speaker-held-out *inferred* AAI sequences for the ASR recordings. They are predicted from audio and should not be treated as measured EMA.

## Experiment

Both models predict 12 utterance-centered TORGO position traces (six articulators, two sagittal axes) from the same 40-band acoustic features and 70 ms context. They use the four frozen speaker-disjoint folds. Each test speaker appears in exactly one fold. Model selection uses validation speakers; the selected model is then refitted on training and validation speakers before test scoring. Only valid channels contribute to the Pearson correlation. Training and scoring use every third 100 Hz frame as a first computational pilot.

- **TORGO-only:** ridge regression trained on the fold's TORGO training speakers.
- **MOCHA-prior:** the same TORGO ridge model, with its weights pulled toward a ridge model trained on the MOCHA training utterances. MOCHA targets were reduced to the six nearest articulator identities and centered by utterance. This is an exploratory transfer test because sensor placement, coordinate systems, and speakers differ. Hyperparameters were chosen on TORGO validation speakers.

## Held-out results

| Speaker group | Test speakers with EMA | TORGO-only mean channel *r* | MOCHA-prior mean channel *r* |
|---|---:|---:|---:|
| Dysarthric | 7 | 0.198 | 0.199 |
| Control | 7 | 0.321 | 0.322 |

These are macro averages of each speaker's available channel scores. The MOCHA prior changes the mean by about 0.001 in each group. It does **not** show a useful transfer gain at this stage. The weak correlations, especially in dysarthric speech, are below the project's aspirational AAI target and are insufficient evidence that an AAI branch will improve recognition. The result is a baseline for improving the AAI model and quality screening, not a final AAI system.

## Artifacts

- `../data/processed/torgo_sensor_map.csv`: publisher sensor key.
- `../data/processed/torgo_publisher_note_match_audit.csv` and `torgo_publisher_sensor_notes.csv`: error-note join and ambiguity record.
- `../data/processed/torgo_aai_feature_index.csv` and `torgo_aai_features/`: aligned pair inventory, audio features, targets, and masks.
- `../data/results/torgo_mocha_transfer/channel_metrics.csv` and `run_config.json`: held-out channel scores, selected parameters, and experiment specification.
- `../data/results/torgo_mocha_transfer/fold*_torgo_only_train_only.npz`: models fitted only on each fold's training speakers for validation-safe ASR fusion experiments.
- `../data/processed/torgo_oof_aai_index.csv` and `torgo_oof_aai_features/`: sensor-free predicted traces for all ASR recordings, generated only with models that held out the corresponding speaker.

Sources: [TORGO publisher README](https://catalog.ldc.upenn.edu/docs/LDC2012S02/README.txt), [MOCHA-TIMIT corpus page](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html).
