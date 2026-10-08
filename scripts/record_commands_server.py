"""Localhost-only microphone recorder for consenting project participants."""
from __future__ import annotations

import csv
import json
import re
import secrets
import subprocess
import sys
import threading
import wave
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "reports/command_recorder.html"
PLAN = ROOT / "data/exports/command_recording_prompts.csv"
DEST = ROOT / "data/new_recordings"
MANIFEST = DEST / "recordings.csv"
FIELDS = ["recording_id", "speaker_id", "session_id", "prompt_id", "written_prompt",
          "intent", "risk_tier", "audio_path", "duration_seconds", "actual_spoken_text",
          "target_status", "recorded_utc"]
ID = re.compile(r"[A-Za-z0-9_-]{1,32}\Z")
RECORDING_ID = re.compile(r"[0-9]{8}T[0-9]{6}_[0-9a-f]{8}_[CN][0-9]{2}\Z")
LOCK = threading.Lock()


def prompts():
    with PLAN.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, data):
        self.send_bytes(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/":
            self.send_bytes(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/prompts.json":
            self.send_json(200, prompts())
        elif path == "/recordings.csv":
            if MANIFEST.exists():
                body = MANIFEST.read_bytes()
            else:
                body = (",".join(FIELDS) + "\n").encode("utf-8")
            self.send_bytes(200, body, "text/csv; charset=utf-8")
        elif path == "/analyze":
            recording_id = parse_qs(parsed.query).get("recording_id", [""])[0]
            if not RECORDING_ID.fullmatch(recording_id):
                self.send_json(400, {"error": "Invalid recording ID"})
                return
            with LOCK:
                if not MANIFEST.exists():
                    self.send_json(404, {"error": "Recording not found"})
                    return
                with MANIFEST.open(newline="", encoding="utf-8") as handle:
                    row = next((item for item in csv.DictReader(handle)
                                if item["recording_id"] == recording_id), None)
            if row is None:
                self.send_json(404, {"error": "Recording not found"})
                return
            audio_path = (ROOT / row["audio_path"]).resolve()
            if not audio_path.is_relative_to((DEST / "audio").resolve()) or not audio_path.is_file():
                self.send_json(400, {"error": "Saved audio is missing"})
                return
            command = [sys.executable, str(ROOT / "scripts/speech_to_action_demo.py"),
                       "--audio", str(audio_path), "--contacts-json",
                       str(ROOT / "data/fixtures/demo_contacts.json")]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired:
                self.send_json(504, {"error": "Speech analysis timed out"})
                return
            except OSError as exc:
                self.send_json(500, {"error": "Could not start speech analysis: " + str(exc)})
                return
            if completed.returncode:
                self.send_json(500, {"error": "Speech analysis failed: " + completed.stderr.strip()[-400:]})
                return
            result = json.loads(completed.stdout)
            self.send_json(200, {"recording_id": recording_id, "transcript": result["transcript"],
                                 "decision": result["decision"], "real_action_executed": False,
                                 "processing_seconds": result["file_to_all_decisions_seconds"]})
        elif path in {"/favicon.ico", "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"}:
            self.send_bytes(204, b"", "image/png")
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlsplit(self.path)
        if parsed.path != "/recording":
            self.send_json(404, {"error": "Not found"})
            return
        query = parse_qs(parsed.query)
        speaker = query.get("speaker", [""])[0]
        session = query.get("session", [""])[0]
        prompt_id = query.get("prompt_id", [""])[0]
        plan = {row["prompt_id"]: row for row in prompts()}
        if not ID.fullmatch(speaker) or not ID.fullmatch(session) or prompt_id not in plan:
            self.send_json(400, {"error": "Invalid speaker, session, or prompt ID"})
            return
        if self.headers.get("X-Participant-Consent") != "yes":
            self.send_json(400, {"error": "Participant permission acknowledgement is required"})
            return
        content_type = self.headers.get("Content-Type", "").split(";")[0]
        suffix = {"audio/webm": ".webm", "audio/mp4": ".m4a", "audio/wav": ".wav",
                  "audio/x-wav": ".wav"}.get(content_type)
        if suffix is None:
            self.send_json(400, {"error": "Unsupported audio format"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= 20_000_000:
            self.send_json(400, {"error": "Recording size must be 1–20 MB"})
            return
        payload = self.rfile.read(size)
        now = datetime.now(timezone.utc)
        recording_id = f"{now:%Y%m%dT%H%M%S}_{secrets.token_hex(4)}_{prompt_id}"
        incoming = DEST / ".incoming"
        destination = DEST / "audio" / speaker / session
        incoming.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        raw_path = incoming / f"{recording_id}{suffix}"
        wav_path = destination / f"{recording_id}.wav"
        raw_path.write_bytes(payload)
        try:
            completed = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw_path),
                                        "-ar", "16000", "-ac", "1", str(wav_path)],
                                       capture_output=True, text=True, timeout=120)
            if completed.returncode:
                raise ValueError(completed.stderr.strip()[:300] or "Audio conversion failed")
            with wave.open(str(wav_path)) as handle:
                if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
                    raise ValueError("Converted WAV has unexpected format")
                duration = handle.getnframes() / handle.getframerate()
            if not 0.3 <= duration <= 30:
                raise ValueError("Recording duration must be 0.3–30 seconds")
            prompt = plan[prompt_id]
            row = {"recording_id": recording_id, "speaker_id": speaker, "session_id": session,
                   "prompt_id": prompt_id, "written_prompt": prompt["written_prompt"],
                   "intent": prompt["intent"], "risk_tier": prompt["risk_tier"],
                   "audio_path": str(wav_path.relative_to(ROOT)),
                   "duration_seconds": f"{duration:.3f}", "actual_spoken_text": "",
                   "target_status": "project_prompt_target", "recorded_utc": now.isoformat()}
            with LOCK:
                MANIFEST.parent.mkdir(parents=True, exist_ok=True)
                exists = MANIFEST.exists()
                with MANIFEST.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
            self.send_json(201, {"recording_id": recording_id, "duration_seconds": row["duration_seconds"],
                                 "audio_path": row["audio_path"]})
        except (ValueError, subprocess.TimeoutExpired) as exc:
            wav_path.unlink(missing_ok=True)
            self.send_json(400, {"error": str(exc)})
        except OSError as exc:
            wav_path.unlink(missing_ok=True)
            self.send_json(500, {"error": "Recorder storage or audio conversion failed: " + str(exc)})
        finally:
            raw_path.unlink(missing_ok=True)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Open http://127.0.0.1:8765/ on this computer", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
