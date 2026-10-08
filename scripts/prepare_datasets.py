"""Inventory TORGO and MOCHA-TIMIT and freeze leakage-safe evaluation splits."""

from __future__ import annotations

import csv
import random
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "extracted"
OUTPUT = ROOT / "data" / "processed"
SEED = 2026

FIELDS = [
    "dataset", "speaker", "condition", "session", "utterance_id",
    "audio_path", "audio_head_path", "audio_array_path", "ema_path",
    "epg_path", "laryngograph_path", "label_path", "reference_path",
    "reference_text", "scoring_reference", "reference_kind", "sample_rate_hz", "duration_seconds",
    "asr_eligible", "aai_pair_candidate", "quality_flags",
]


def relative(path: Path | None) -> str:
    return str(path.relative_to(ROOT)) if path else ""


def existing(path: Path) -> Path | None:
    return path if path.is_file() and path.stat().st_size > 0 else None


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wave_metadata(path: Path, nist: bool = False) -> tuple[int, float]:
    if nist:
        with path.open("rb") as handle:
            header = handle.read(1024).decode("ascii", errors="replace")
        if not header.startswith("NIST_1A") or "end_head" not in header:
            raise ValueError("invalid NIST header")
        def integer(field: str) -> int:
            match = re.search(rf"^{field} -i (\d+)$", header, re.MULTILINE)
            if not match:
                raise ValueError(f"missing NIST {field}")
            return int(match.group(1))
        rate, count = integer("sample_rate"), integer("sample_count")
        if path.stat().st_size < 1024 + count * integer("sample_n_bytes"):
            raise ValueError("truncated NIST audio")
        return rate, count / rate
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        return rate, handle.getnframes() / rate


def scoring_reference(text: str) -> str:
    clean = text.strip()
    if (not clean or clean.lower() == "xxx" or clean.lower().endswith(".jpg")
            or clean.startswith("[") or any(ord(char) < 32 and char not in "\n\r\t" for char in clean)):
        return ""
    # TORGO sometimes appends pronunciation guidance to a spoken target word.
    clean = re.sub(r"\[[^\]]*\]", "", clean).strip()
    if "[" in clean or "]" in clean or not re.search(r"[A-Za-z0-9]", clean):
        return ""
    return clean


def candidate_audio(paths: list[Path | None], nist: bool, flags: list[str]) -> tuple[Path | None, int | str, float | str]:
    for index, path in enumerate(paths):
        if path is None:
            continue
        try:
            rate, duration = wave_metadata(path, nist)
            if rate != 16000 or duration <= 0:
                raise ValueError(f"unexpected audio rate/duration: {rate}/{duration}")
            if index:
                flags.append("used_array_mic_fallback")
            return path, rate, round(duration, 5)
        except (ValueError, wave.Error, EOFError) as exc:
            flags.append(f"invalid_audio_{index}:{type(exc).__name__}")
    flags.append("missing_valid_audio")
    return None, "", ""


def torgo_rows() -> list[dict]:
    rows = []
    root = EXTRACTED / "torgo"
    for speaker_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        speaker = speaker_dir.name
        condition = "control" if "C" in speaker else "dysarthric"
        for session_dir in sorted(speaker_dir.glob("Session*")):
            ids: set[str] = set()
            for folder, suffix in (("wav_headMic", ".wav"), ("wav_arrayMic", ".wav"), ("pos", ".pos"), ("prompts", ".txt")):
                ids.update(path.stem for path in (session_dir / folder).glob(f"*{suffix}"))
            for uid in sorted(ids):
                flags: list[str] = []
                head = existing(session_dir / "wav_headMic" / f"{uid}.wav")
                array = existing(session_dir / "wav_arrayMic" / f"{uid}.wav")
                ema = existing(session_dir / "pos" / f"{uid}.pos")
                prompt = existing(session_dir / "prompts" / f"{uid}.txt")
                audio, rate, duration = candidate_audio([head, array], False, flags)
                if ema is None:
                    flags.append("missing_ema")
                if prompt is None:
                    flags.append("missing_prompt")
                    reference = ""
                else:
                    reference = prompt.read_text(encoding="utf-8", errors="replace").strip()
                    if not scoring_reference(reference):
                        flags.append("unusable_prompt")
                rows.append({
                    "dataset": "TORGO", "speaker": speaker, "condition": condition,
                    "session": session_dir.name, "utterance_id": uid,
                    "audio_path": relative(audio), "audio_head_path": relative(head),
                    "audio_array_path": relative(array), "ema_path": relative(ema),
                    "epg_path": "", "laryngograph_path": "", "label_path": "",
                    "reference_path": relative(prompt), "reference_text": reference,
                    "scoring_reference": scoring_reference(reference),
                    "reference_kind": "elicitation_prompt", "sample_rate_hz": rate,
                    "duration_seconds": duration,
                    "asr_eligible": int(audio is not None and bool(scoring_reference(reference))),
                    "aai_pair_candidate": int(audio is not None and ema is not None),
                    "quality_flags": "|".join(flags),
                })
    return rows


def mocha_prompts() -> dict[str, str]:
    text = (ROOT / "data" / "raw" / "mocha_timit" / "mocha-timit.txt").read_text(encoding="utf-8", errors="replace")
    return {match.group(1): match.group(2).strip() for match in re.finditer(r"(?m)^(\d{3})\.\s+(.+)$", text)}


def mocha_rows() -> list[dict]:
    rows = []
    prompts = mocha_prompts()
    root = EXTRACTED / "mocha_timit"
    for speaker_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        speaker = speaker_dir.name.split("_")[0]
        ids = set(prompts)
        ids.update(path.stem.rsplit("_", 1)[-1] for path in speaker_dir.glob(f"{speaker}_*.*") if path.stem.rsplit("_", 1)[-1].isdigit())
        for uid in sorted(ids):
            flags: list[str] = []
            audio_file = existing(speaker_dir / f"{speaker}_{uid}.wav")
            ema = existing(speaker_dir / f"{speaker}_{uid}.ema")
            epg = existing(speaker_dir / f"{speaker}_{uid}.epg")
            lar = existing(speaker_dir / f"{speaker}_{uid}.lar")
            label = existing(speaker_dir / f"{speaker}_{uid}.lab")
            audio, rate, duration = candidate_audio([audio_file], True, flags)
            if ema is None:
                flags.append("missing_ema")
            else:
                with ema.open("rb") as handle:
                    if not handle.read(32).startswith(b"EST_File Track"):
                        flags.append("invalid_ema_header")
                        ema = None
            if label is None:
                flags.append("missing_phoneme_label")
            if epg is None:
                flags.append("missing_epg")
            if lar is None:
                flags.append("missing_laryngograph")
            if speaker == "maps0":
                flags.append("supplementary_speaker")
                if uid == "118":
                    flags.append("tongue_tip_corrupt_per_readme")
            if speaker == "msak0" and uid == "268":
                flags.append("audio_corrupt_per_readme")
            reference = prompts.get(uid, "")
            if not scoring_reference(reference):
                flags.append("missing_or_unusable_prompt")
            known_bad_audio = "audio_corrupt_per_readme" in flags
            known_bad_ema = "tongue_tip_corrupt_per_readme" in flags
            rows.append({
                "dataset": "MOCHA-TIMIT", "speaker": speaker, "condition": "control",
                "session": speaker_dir.name, "utterance_id": uid,
                "audio_path": relative(audio), "audio_head_path": "", "audio_array_path": "",
                "ema_path": relative(ema), "epg_path": relative(epg),
                "laryngograph_path": relative(lar), "label_path": relative(label),
                "reference_path": relative(ROOT / "data" / "raw" / "mocha_timit" / "mocha-timit.txt"),
                "reference_text": reference, "scoring_reference": scoring_reference(reference),
                "reference_kind": "shared_prompt",
                "sample_rate_hz": rate, "duration_seconds": duration,
                "asr_eligible": int(audio is not None and bool(scoring_reference(reference)) and not known_bad_audio),
                "aai_pair_candidate": int(audio is not None and ema is not None and not known_bad_audio and not known_bad_ema),
                "quality_flags": "|".join(flags),
            })
    return rows


def torgo_splits(rows: list[dict]) -> list[dict]:
    groups = [
        ["F01", "M01", "FC01", "MC01"],
        ["F03", "M02", "FC02", "MC02"],
        ["F04", "M03", "FC03", "MC03"],
        ["M04", "M05", "MC04"],
    ]
    speakers = {row["speaker"] for row in rows if row["dataset"] == "TORGO"}
    assert set().union(*map(set, groups)) == speakers
    assert sum(map(len, groups)) == len(speakers)
    counts = Counter(row["speaker"] for row in rows if row["dataset"] == "TORGO" and row["asr_eligible"])
    output = []
    for fold in range(4):
        test = set(groups[fold])
        validation = set(groups[(fold + 1) % 4])
        for speaker in sorted(speakers):
            output.append({
                "fold": fold, "speaker": speaker,
                "condition": "control" if "C" in speaker else "dysarthric",
                "role": "test" if speaker in test else "validation" if speaker in validation else "train",
                "asr_utterances": counts[speaker],
            })
    return output


def mocha_splits(rows: list[dict]) -> list[dict]:
    output = []
    for speaker in ("fsew0", "msak0"):
        speaker_rows = [row for row in rows if row["dataset"] == "MOCHA-TIMIT" and row["speaker"] == speaker]
        assert len(speaker_rows) == 460
        ids = sorted(row["utterance_id"] for row in speaker_rows if row["aai_pair_candidate"])
        random.Random(SEED + sum(map(ord, speaker))).shuffle(ids)
        train_count = len(ids) - 60
        roles = {uid: "train" if i < train_count else "validation" if i < train_count + 30 else "test" for i, uid in enumerate(ids)}
        output.extend({"speaker": speaker, "utterance_id": row["utterance_id"], "role": roles.get(row["utterance_id"], "excluded")}
                      for row in sorted(speaker_rows, key=lambda item: item["utterance_id"]))
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = torgo_rows() + mocha_rows()
    write_csv(OUTPUT / "utterance_manifest.csv", FIELDS, rows)
    exclusions = [row for row in rows if row["quality_flags"]]
    write_csv(OUTPUT / "quality_flags.csv", FIELDS, exclusions)
    exclusion_rows = []
    for row in rows:
        flags = row["quality_flags"].split("|")
        for task, eligible, relevant in (
            ("ASR scoring", row["asr_eligible"], ("missing_valid_audio", "missing_prompt", "unusable_prompt", "missing_or_unusable_prompt", "audio_corrupt_per_readme")),
            ("AAI pairing", row["aai_pair_candidate"], ("missing_valid_audio", "missing_ema", "invalid_ema_header", "tongue_tip_corrupt_per_readme", "audio_corrupt_per_readme")),
        ):
            if not eligible:
                reasons = [flag for flag in flags if any(flag.startswith(prefix) for prefix in relevant)]
                exclusion_rows.append({"dataset": row["dataset"], "speaker": row["speaker"],
                                       "session": row["session"], "utterance_id": row["utterance_id"],
                                       "task": task, "reason": "|".join(reasons) or "failed_eligibility_check"})
        if row["dataset"] == "MOCHA-TIMIT" and row["speaker"] == "maps0":
            exclusion_rows.append({"dataset": row["dataset"], "speaker": row["speaker"],
                                   "session": row["session"], "utterance_id": row["utterance_id"],
                                   "task": "primary MOCHA AAI pretraining", "reason": "supplementary_speaker"})
    write_csv(OUTPUT / "exclusion_log.csv", ["dataset", "speaker", "session", "utterance_id", "task", "reason"], exclusion_rows)
    write_csv(OUTPUT / "torgo_aai_pair_candidates.csv", FIELDS,
              [row for row in rows if row["dataset"] == "TORGO" and row["aai_pair_candidate"]])
    write_csv(OUTPUT / "torgo_speaker_splits.csv", ["fold", "speaker", "condition", "role", "asr_utterances"], torgo_splits(rows))
    write_csv(OUTPUT / "mocha_utterance_splits.csv", ["speaker", "utterance_id", "role"], mocha_splits(rows))
    write_csv(OUTPUT / "mocha_speaker_swap.csv", ["experiment", "train_speaker", "validation_speaker", "unseen_test_speaker"], [
        {"experiment": "fsew0_to_msak0", "train_speaker": "fsew0", "validation_speaker": "fsew0", "unseen_test_speaker": "msak0"},
        {"experiment": "msak0_to_fsew0", "train_speaker": "msak0", "validation_speaker": "msak0", "unseen_test_speaker": "fsew0"},
    ])
    for dataset in ("TORGO", "MOCHA-TIMIT"):
        subset = [row for row in rows if row["dataset"] == dataset]
        print(dataset, "utterances", len(subset), "ASR", sum(row["asr_eligible"] for row in subset), "AAI pairs", sum(row["aai_pair_candidate"] for row in subset))
        print("by speaker", dict(sorted(Counter(row["speaker"] for row in subset).items())))
        print("flag counts", dict(Counter(flag for row in subset for flag in row["quality_flags"].split("|") if flag)))


if __name__ == "__main__":
    main()
