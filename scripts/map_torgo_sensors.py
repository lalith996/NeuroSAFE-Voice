"""Record the official TORGO coil key and join publisher error annotations.

Ambiguous workbook trial IDs are preserved as review notes and never silently
assigned to a specific speaker/session/utterance.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import xlrd


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data/processed/torgo_aai_verified_pairs.csv"
WORKBOOK = ROOT / "data/raw/torgo/ERRORS.xls"
MAP_OUT = ROOT / "data/processed/torgo_sensor_map.csv"
NOTE_OUT = ROOT / "data/processed/torgo_publisher_sensor_notes.csv"
UNMATCHED = ROOT / "data/processed/torgo_publisher_note_match_audit.csv"
SENSOR_NAMES = [
    "tongue_back", "tongue_middle", "tongue_tip", "forehead_reference",
    "nose_bridge_reference", "upper_lip", "lower_lip", "lower_incisor_jaw",
    "left_mouth_corner", "right_mouth_corner", "left_ear_reference",
    "right_ear_reference",
]
SOURCE = "https://catalog.ldc.upenn.edu/docs/LDC2012S02/README.txt"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trial_numbers(value: object) -> list[int] | None:
    if isinstance(value, (int, float)) and int(value) == value:
        return [int(value)]
    text = str(value).strip().lower()
    if text in ("all trials", "all trials "):
        return None
    match = re.fullmatch(r"(\d+)\s+to\s+(\d+)", text)
    if match:
        start, stop = map(int, match.groups())
        return list(range(start, stop + 1)) if stop - start <= 1000 else []
    return [int(text)] if re.fullmatch(r"\d+", text) else []


def channels(value: object) -> str:
    numbers = [int(x) for x in re.findall(r"\d+", str(value))]
    return "|".join(str(x) for x in numbers if 1 <= x <= 12)


def main() -> None:
    sensor_rows = [{"sensor_index_1based": i, "name": name,
                    "coordinate_fields": "x|y|z", "source": SOURCE,
                    "status": "official_default_except_where_session_notes_indicate_otherwise"}
                   for i, name in enumerate(SENSOR_NAMES, 1)]
    write_csv(MAP_OUT, sensor_rows, list(sensor_rows[0]))
    pairs = read_csv(PAIRS)
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    by_speaker_trial: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in pairs:
        by_speaker[row["speaker"]].append(row)
        by_speaker_trial[(row["speaker"], int(row["utterance_id"]))].append(row)
    sheet = xlrd.open_workbook(str(WORKBOOK)).sheet_by_index(0)
    raw_label = ""
    note_rows = []
    audit_rows = []
    for row_number in range(1, sheet.nrows):
        raw = sheet.row_values(row_number)
        if raw[0]:
            raw_label = str(raw[0]).strip()
        speaker_match = re.match(r"^(?:MC|FC|M|F)\d{2}", raw_label)
        if not speaker_match or not str(raw[3]).strip():
            continue
        speaker = speaker_match.group(0)
        spec = raw[1]
        numbers = trial_numbers(spec)
        if numbers is None:
            candidates = by_speaker[speaker]
            scope = "all_trials_label"
        else:
            candidates = [match for number in numbers for match in by_speaker_trial[(speaker, number)]]
            scope = "numeric_or_range"
        unique = {(r["speaker"], r["session"], r["utterance_id"]): r for r in candidates}
        if numbers is not None and len(numbers) == 1:
            match_status = "unique" if len(unique) == 1 else "ambiguous" if len(unique) > 1 else "unmatched"
        elif numbers is None:
            match_status = "speaker_wide_single_session" if len({r["session"] for r in unique.values()}) == 1 else "speaker_wide_multiple_sessions"
        else:
            match_status = "range_review" if unique else "unmatched"
        comment = str(raw[3]).strip()
        channel_text = channels(raw[2])
        audit_rows.append({
            "workbook_row_1based": row_number + 1, "source_subject_label": raw_label,
            "speaker": speaker, "trial_spec": str(spec).strip(),
            "channel_spec": str(raw[2]).strip(), "channels_1based": channel_text,
            "comment": comment, "scope": scope, "match_status": match_status,
            "matched_candidate_rows": len(unique),
        })
        for candidate in unique.values():
            note_rows.append({
                "speaker": candidate["speaker"], "session": candidate["session"],
                "utterance_id": candidate["utterance_id"],
                "workbook_row_1based": row_number + 1,
                "source_subject_label": raw_label,
                "trial_spec": str(spec).strip(),
                "channels_1based": channel_text,
                "comment": comment, "match_status": match_status,
                "usable_as_record_specific_note": int(match_status in ("unique", "speaker_wide_single_session")),
            })
    write_csv(NOTE_OUT, note_rows, list(note_rows[0]))
    write_csv(UNMATCHED, audit_rows, list(audit_rows[0]))
    from collections import Counter
    print("Official coil map:", len(sensor_rows), "sensors")
    print("Workbook annotation rows:", len(audit_rows), "expanded candidate notes:", len(note_rows))
    print("Match statuses:", dict(Counter(row["match_status"] for row in audit_rows)))


if __name__ == "__main__":
    main()
