"""Package the fixed longer test recordings with the local listening page."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXPORT = ROOT / "data/exports"
CSV = REPORTS / "long_test_transcript_review.csv"
HTML = REPORTS / "long_test_transcript_review.html"
ZIP = EXPORT / "torgo_long_test_transcript_review.zip"


def main():
    with CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = sorted({row["audio_path"] for row in rows})
    if len(paths) != len(rows):
        raise ValueError("Review clip IDs are not unique")
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(HTML, "reports/long_test_transcript_review.html")
        archive.write(CSV, "reports/long_test_transcript_review.csv")
        archive.writestr("README.txt", "Extract the ZIP, open reports/long_test_transcript_review.html, "
                         "listen to each recording, choose a review status, enter corrected words when needed, "
                         "and click Export review CSV. Send that completed CSV for clip-linked scoring. "
                         "Written prompts and model output are clues, not verified speech.\n")
        for path in paths:
            archive.write(ROOT / path, path)
    with zipfile.ZipFile(ZIP) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt review ZIP")
    summary = {"recordings": len(paths), "zip_bytes": ZIP.stat().st_size,
               "sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
               "status": "review_pack_pending_human_clip_level_decisions"}
    (EXPORT / "long_review_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
