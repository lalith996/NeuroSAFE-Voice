# NeuroSAFE-Voice datasets

Downloaded from the original university hosts on 28 September 2026 for the academic project.

## Locations

- `raw/torgo/`: four original TORGO archives (`F`, `FC`, `M`, `MC`), the University of Toronto source and license page, recording notes, coil diagram, and a SHA-256 manifest.
- `extracted/torgo/`: the unpacked speaker folders. There are 15 speakers: 8 with dysarthria and 7 controls.
- `raw/mocha_timit/`: three original MOCHA-TIMIT archives, license, readmes, prompt text, and a SHA-256 manifest.
- `extracted/mocha_timit/`: each MOCHA-TIMIT archive unpacked into its own folder.
- `processed/`: utterance inventory, exclusions, fixed splits, and decoded MOCHA audio/EMA pairs. See [the preparation notes](processed/README.md).
- `results/whisper_small_en_zero_shot/`: resumable zero-shot ASR predictions and WER reports.

## Sources and terms

- [TORGO, University of Toronto](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html): free for academic, non-profit use. Cite a paper listed on the source page when publishing results.
- [MOCHA-TIMIT, University of Edinburgh](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html): free for non-commercial use. Retain `LICENCE.txt` with the data.

The original license and documentation files are kept in `raw/`. Do not place the recordings in a public code repository.

The public repository includes project CSV manifests, prompt targets, experiment summaries, and trained model artifacts. To reproduce audio-dependent processing, obtain the corpora from the original hosts under their terms and put them in the `raw/` and `extracted/` locations described above. Review ZIPs, derived audio/EMA arrays, and participant recordings also remain local. The recorder creates `new_recordings/` automatically when it is first used.

## Inventory notes for preprocessing

- MOCHA-TIMIT `fsew0_v1.1` and `msak0_v1.1` each have 460 `.wav`, `.ema`, `.epg`, `.lar`, and `.lab` files. These are the two main speakers described by the official overview.
- The additional `maps0` archive has 459 `.wav`, `.epg`, and `.lar` files, 460 `.ema` files, and no per-utterance `.lab` files. Treat it as supplementary until its missing and corrupt-file notes are handled.
- TORGO has audio, prompts, and EMA `.pos` files in speaker/session folders, but the modalities are not present for every utterance. In particular, the extracted `F01` folder has no `.pos` files. Build AAI training pairs only from utterances with matching usable audio and EMA.
- Archive extraction completed without compressed-file errors. The SHA-256 manifests in the raw folders were checked after download; they record local integrity hashes, not publisher-provided checksums.
