# AAI improvement round 2: held-out TORGO speakers

## Change that helped

I added per-recording acoustic mean and variance normalization before the existing 70 ms-context ridge AAI model. Each test recording is normalized using **its own audio only**. Model normalization and weights are still fitted only on training speakers. The four fixed TORGO folds, target masks, test utterances, and scoring code are otherwise the same as round 1.

| AAI method | Dysarthric speaker mean channel *r* | Control speaker mean channel *r* |
|---|---:|---:|
| Original TORGO ridge | 0.198 | 0.321 |
| Ridge with per-recording acoustic normalization | **0.213** | **0.342** |
| Ridge with acoustic normalization and MOCHA weight prior | 0.214 | 0.342 |
| Temporal convolution, TORGO dysarthric and control training | 0.194 | 0.338 |
| Temporal convolution, MOCHA pretraining then TORGO training | 0.191 | 0.346 |
| Temporal convolution, dysarthric TORGO training only | 0.152 | 0.260 |
| Temporal convolution, MOCHA pretraining then dysarthric TORGO training | 0.158 | 0.258 |

The acoustic normalization improves the dysarthric mean by 0.015 correlation points; six of seven test speakers improve. A descriptive bootstrap across the seven fixed dysarthric test speakers gives a 95% interval of +0.007 to +0.023. This is a modest gain, still far below the proposal's *r* ≥ 0.70 target. MOCHA pretraining and the tested temporal model do not provide a further dysarthric gain. The retained AAI reference is therefore the **TORGO ridge with per-recording acoustic normalization**.

## How the alternatives were tested

The temporal model uses 40-band log-mel audio, 96 hidden channels, four dilated convolution blocks, and 12 masked articulator coordinates. It trained for 300 TORGO steps per fold using speaker-balanced windows and masked smooth-L1 loss. One variant was initialized by 300 steps on MOCHA training utterances; another used only dysarthric TORGO training speakers. All target and training normalization was fitted from training examples. Scoring used the same speaker-held-out folds and every third 100 Hz frame as the ridge comparison.

Only seven dysarthric speakers have validated TORGO audio/EMA pairs. F01 has no such pair and cannot contribute an AAI correlation. Several sessions have missing or faulty tongue coils; channel masks keep their valid lip or jaw traces in the analysis. The `pos` tracks are already head corrected according to the [TORGO publisher documentation](https://catalog.ldc.upenn.edu/docs/LDC2012S02/README.txt). Different coil placements and position conventions remain a major limitation for the approximate MOCHA-to-TORGO channel mapping.

## Saved results

- `../data/results/torgo_mocha_transfer_cmvn/`: best ridge models, channel scores, training-only AAI models, and run configuration.
- `../data/results/temporal_aai/`: all-speaker temporal results and weights.
- `../data/results/temporal_aai_dys_only/`: dysarthric-only temporal results and weights.
- `../data/processed/torgo_oof_aai_cmvn_index.csv` and `torgo_oof_aai_features_cmvn/`: predicted articulatory traces for ASR, made with a model that held out each speaker.

This change improves the AAI baseline, but the current inferred features have not shown an ASR gain in the short-recording fusion pilot. A larger fusion experiment should wait for stronger AAI validation or use the current predictions only as a research ablation.
