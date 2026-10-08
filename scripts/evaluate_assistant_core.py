"""Check local intent routing, confirmation, false actions, and core latency."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from assistant_core import AssistantCore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/fixtures/assistant_intents.csv"
ASR_HYPOTHESES = ROOT / "data/results/whisper_full_prompt_summary/utterances.csv"
OUTPUT = ROOT / "data/results/assistant_core"


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixtures = read(FIXTURES)
    results = []
    for fixture in fixtures:
        assistant = AssistantCore(contacts={"asha": "Asha", "mum": "Asha", "sam": "Sam"})
        parsed = assistant.parse(fixture["text"])
        response = assistant.handle(fixture["text"], asr_confidence=None)
        expected = None if fixture["expected_intent"] == "none" else fixture["expected_intent"]
        correct = ((parsed.name if parsed else None) == expected
                   and (parsed.risk if parsed else "") == fixture["expected_risk"]
                   and (parsed.arguments.get("contact", "") if parsed else "") == fixture["expected_contact"])
        safe = not assistant.receipts and (response["status"] == "confirmation_required" if expected
                                           else response["status"] == "ask_clarification")
        results.append({"text": fixture["text"], "expected_intent": fixture["expected_intent"],
                        "observed_intent": parsed.name if parsed else "none", "intent_correct": int(correct),
                        "route_status": response["status"], "safe_without_confidence": int(safe),
                        "core_response_ms": response["response_ms"]})
    with (OUTPUT / "fixture_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    # A high-risk intent remains pending even if ASR later says "confirm".
    safety = AssistantCore(contacts={"asha": "Asha"})
    first = safety.handle("Call Asha", asr_confidence=1.0)
    spoken_confirm = safety.handle("confirm", asr_confidence=1.0)
    no_voice_execution = (first["status"] == "confirmation_required"
                          and spoken_confirm["status"] == "use_confirmation_control"
                          and not safety.receipts)
    explicit = safety.confirm_pending(True)
    explicit_only_simulated = (explicit["status"] == "simulated_execution"
                               and len(safety.receipts) == 1
                               and safety.receipts[0]["real_side_effect"] is False)
    # TORGO prompts are narrative speech, not a labeled command dataset. This
    # is a negative-control scan only, not an estimate of real-user specificity.
    hypotheses = [r["hypothesis"] for r in read(ASR_HYPOTHESES) if r["stage"] == "full_prompt_m1"]
    scan = AssistantCore()
    accidental_intents = [text for text in hypotheses if scan.parse(text) is not None]
    latencies = sorted(r["core_response_ms"] for r in results)
    summary = {"synthetic_fixture_count": len(results),
               "fixture_intent_correct": sum(r["intent_correct"] for r in results),
               "fixture_safe_without_confidence": sum(r["safe_without_confidence"] for r in results),
               "high_risk_voice_confirmation_blocked": no_voice_execution,
               "explicit_confirmation_simulation_only": explicit_only_simulated,
               "torgo_noncommand_hypotheses_checked": len(hypotheses),
               "accidental_intent_matches": len(accidental_intents),
               "core_latency_median_ms": statistics.median(latencies),
               "core_latency_p95_ms": latencies[int(.95 * (len(latencies) - 1))],
               "scope": "synthetic text and negative controls only; no ASR-to-intent or real action accuracy claim"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not all(r["intent_correct"] and r["safe_without_confidence"] for r in results):
        raise AssertionError("Intent fixture or safety routing failed")
    if not no_voice_execution or not explicit_only_simulated:
        raise AssertionError("Confirmation gate failed")


if __name__ == "__main__":
    main()
