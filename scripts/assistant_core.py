"""Local, auditable intent and safety prototype for NeuroSAFE-Voice.

This module performs no real-world side effects. It records simulated action
receipts so intent parsing, confirmation, and false-action behavior can be
tested before connecting external tools.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Intent:
    name: str
    arguments: dict[str, str]
    risk: str


class AssistantCore:
    def __init__(self, contacts: dict[str, str] | None = None):
        # Contact data is supplied locally by the user; the default is empty.
        self.contacts = {clean(alias): canonical for alias, canonical in (contacts or {}).items()}
        self.pending: Intent | None = None
        self.receipts: list[dict] = []

    def parse(self, text: str) -> Intent | None:
        phrase = clean(text)
        if phrase in {"what time is it", "tell me the time", "what is the time"}:
            return Intent("read_time", {}, "low")
        if phrase in {"show my calendar", "read my calendar", "what is on my calendar"}:
            return Intent("read_calendar", {}, "low")
        match = re.fullmatch(r"open (the )?(browser|calendar|notes)", phrase)
        if match:
            return Intent("open_app", {"app": match.group(2)}, "low")
        match = re.fullmatch(r"turn (on|off) the (bedroom|living room|kitchen) light", phrase)
        if match:
            return Intent("set_light", {"state": match.group(1), "room": match.group(2)}, "medium")
        match = re.fullmatch(r"call ([\w'-]+)", phrase)
        if match:
            contact = self.contacts.get(match.group(1))
            return Intent("place_call", {"contact": contact}, "high") if contact else None
        match = re.fullmatch(r"send (a )?message to ([\w'-]+) saying (.{1,160})", phrase)
        if match:
            contact = self.contacts.get(match.group(2))
            return (Intent("send_message", {"contact": contact, "message": match.group(3)}, "high")
                    if contact else None)
        if phrase in {"delete the old note", "delete my old note"}:
            return Intent("delete_note", {"note": "old note"}, "high")
        return None

    def handle(self, text: str, asr_confidence: float | None = None) -> dict:
        started = time.perf_counter()
        phrase = clean(text)
        if phrase == "cancel":
            had_pending = self.pending is not None
            self.pending = None
            return self._response("cancelled" if had_pending else "nothing_to_cancel", None, started)
        if phrase == "confirm":
            # An ASR transcript alone cannot approve a queued risky action.
            return self._response("use_confirmation_control" if self.pending else "nothing_to_confirm",
                                  self.pending, started)
        intent = self.parse(text)
        if intent is None:
            self.pending = None
            return self._response("ask_clarification", None, started)
        if asr_confidence is not None and not 0 <= asr_confidence <= 1:
            raise ValueError("ASR confidence must be between 0 and 1")
        # Current Whisper confidence is not calibrated. Unknown confidence
        # always routes to confirmation, even for nominally low-risk intents.
        needs_confirmation = (intent.risk == "high" or asr_confidence is None
                              or asr_confidence < (0.90 if intent.risk == "low" else 0.95))
        if needs_confirmation:
            self.pending = intent
            return self._response("confirmation_required", intent, started)
        self.pending = None
        receipt = {"intent": intent.name, "arguments": intent.arguments,
                   "status": "simulated_success", "real_side_effect": False}
        self.receipts.append(receipt)
        response = self._response("simulated_execution", intent, started)
        response["receipt"] = receipt
        return response

    def confirm_pending(self, confirmed_by_user: bool) -> dict:
        """Call only from an explicit confirmation control, never ASR text."""
        started = time.perf_counter()
        if self.pending is None:
            return self._response("nothing_to_confirm", None, started)
        if not confirmed_by_user:
            self.pending = None
            return self._response("cancelled", None, started)
        action = self.pending
        self.pending = None
        receipt = {"intent": action.name, "arguments": action.arguments,
                   "status": "simulated_success", "real_side_effect": False}
        self.receipts.append(receipt)
        response = self._response("simulated_execution", action, started)
        response["receipt"] = receipt
        return response

    @staticmethod
    def _response(status: str, intent: Intent | None, started: float) -> dict:
        return {"status": status, "intent": intent.name if intent else None,
                "arguments": intent.arguments if intent else {},
                "risk": intent.risk if intent else None,
                "response_ms": (time.perf_counter() - started) * 1000}
