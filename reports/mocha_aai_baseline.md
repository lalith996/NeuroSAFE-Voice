# MOCHA acoustic-to-articulatory baseline

## Method

- Used 919 decoded audio/EMA pairs: 460 from `fsew0`, 459 from `msak0` after the publisher-noted corrupt recording was excluded.
- Kept the supplied time synchronization. Feature frames occur every 10 ms only where the 16 kHz audio and 500 Hz EMA overlap. The audio generally extends beyond the EMA track; that trailing audio was not used for training. No time lag was estimated.
- Extracted 40-band log-mel features from 25 ms audio windows. Predicted the x and y positions of seven speech articulators relative to the upper incisor. Two unnamed `****` channels and the head-reference channels were not included as prediction targets.
- Fitted a ridge regression baseline with 70 ms of acoustic context. Selected regularization by validation correlation, then fitted the final model on that speaker's train plus validation utterances. All acoustic and target normalization used only that fitting set.
- Evaluated on each speaker's 30 held-out test utterances and on every available utterance of the other, unseen speaker. Correlation intervals resample whole utterances, 1,000 times.

## Results

| Train speaker | Evaluation speaker | Scope | Utterances | Mean Pearson r across 14 channels | Median channel r |
|---|---|---|---:|---:|---:|
| fsew0 | fsew0 | Held-out test | 30 | 0.581 | 0.627 |
| msak0 | msak0 | Held-out test | 30 | 0.598 | 0.615 |
| fsew0 | msak0 | Unseen-speaker swap | 459 | 0.317 | 0.303 |
| msak0 | fsew0 | Unseen-speaker swap | 460 | 0.313 | 0.310 |

These are reference results for a simple linear AAI model, not the proposed final neural AAI head. The lower speaker-swap correlations show a substantial transfer gap; differences in coil placement and coordinate orientation may contribute. Pearson r is pooled over frames for each channel; the confidence intervals in `channel_metrics.csv` use utterances as the resampling unit. The speaker-swap test uses the same sentence collection across the two speakers, so it tests speaker transfer rather than unseen text. The project's r ≥ 0.70 threshold has not been met as a 14-channel mean by this baseline.

## Saved outputs

- `../data/processed/mocha_aai_feature_index.csv` and `mocha_aai_feature_config.json`: 100 Hz feature inventory and extraction specification.
- `../data/processed/mocha_aai_features/`: per-utterance log-mel features, EMA targets, and timestamps.
- `../data/results/mocha_ridge_aai/channel_metrics.csv`: per-channel Pearson r with utterance-bootstrap 95% intervals.
- `../data/results/mocha_ridge_aai/summary.json`: aggregate results.
- `../data/results/mocha_ridge_aai/model_*.npz` and `predictions/`: fitted normalization, model weights, held-out predictions, and targets.

The next AAI experiment should establish TORGO sensor identity and quality masks, then adapt the MOCHA-trained method to dysarthric speakers under the fixed speaker-disjoint folds. The simple ridge model is an ablation reference for a later temporal neural model.

Source: [MOCHA-TIMIT corpus documentation](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html).
