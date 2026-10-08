"""Select the decoder using only validation-speaker prompt WER."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/results/whisper_full_prompt_summary/decode_selection.json"


def main():
    totals = {option: {field: 0 for field in ("clips", "words", "errors", "long_clips", "long_words", "long_errors")}
              for option in ("default", "guard")}
    for fold in range(4):
        seen = {}
        for option in totals:
            path = ROOT / f"data/results/whisper_full_prompt_fold{fold}/validation_full_{option}_decode.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            ids = {(r["speaker"], r["session"], r["utterance_id"]) for r in rows}
            if len(ids) != len(rows):
                raise ValueError(f"Duplicate fold {fold} {option} validation IDs")
            seen[option] = ids
            summary = json.loads(path.with_suffix(".json").read_text())
            for field in totals[option]:
                totals[option][field] += int(summary[field])
        if seen["default"] != seen["guard"]:
            raise ValueError(f"Fold {fold} decoder clips differ")
    selected = min(totals, key=lambda option: (totals[option]["errors"] / totals[option]["words"],
                                                option != "default"))
    result = {"validation_totals": totals, "selected_decoder": selected,
              "selection_rule": "lower pooled validation WER; default on a tie",
              "candidate_motivated_by_prior_test_error": True,
              "numeric_selection_uses_validation_only": True}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
