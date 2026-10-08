"""End-to-end local recorder smoke check with generated audio in a temp dir."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import record_commands_server as recorder


def main():
    with tempfile.TemporaryDirectory() as temp:
        recorder.ROOT = Path(temp)
        recorder.DEST = recorder.ROOT / "recordings"
        recorder.MANIFEST = recorder.DEST / "recordings.csv"
        server = ThreadingHTTPServer(("127.0.0.1", 0), recorder.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/") as response:
                assert b"Local command recorder" in response.read()
            with urllib.request.urlopen(base + "/prompts.json") as response:
                assert b'"prompt_id": "C01"' in response.read()
            with urllib.request.urlopen(base + "/recordings.csv") as response:
                assert response.read().decode().startswith("recording_id,speaker_id,")
            try:
                urllib.request.urlopen(base + "/analyze?recording_id=invalid")
            except urllib.error.HTTPError as exc:
                assert exc.code == 400
            else:
                raise AssertionError("Invalid recording ID was accepted")
            audio = io.BytesIO()
            with wave.open(audio, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\0\0" * 8000)
            request = urllib.request.Request(
                base + "/recording?speaker=TEST001&session=S1&prompt_id=C01",
                data=audio.getvalue(), method="POST",
                headers={"Content-Type": "audio/wav", "X-Participant-Consent": "yes"})
            with urllib.request.urlopen(request) as response:
                assert response.status == 201
                recording_id = json.load(response)["recording_id"]
            with recorder.MANIFEST.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 1 and rows[0]["prompt_id"] == "C01"
            with urllib.request.urlopen(base + "/recordings.csv") as response:
                assert recording_id in response.read().decode()
            fake_result = {"transcript": "Open the browser", "decision": {
                "intent": "open_app", "status": "confirmation_required"},
                "file_to_all_decisions_seconds": 0.25}
            with patch.object(recorder.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps(fake_result), stderr="")):
                with urllib.request.urlopen(base + "/analyze?recording_id=" + recording_id) as response:
                    analyzed = json.load(response)
            assert analyzed["decision"]["status"] == "confirmation_required"
            assert analyzed["real_action_executed"] is False
            path = recorder.ROOT / rows[0]["audio_path"]
            assert path.is_file()
            with wave.open(str(path)) as handle:
                assert handle.getframerate() == 16000 and handle.getnchannels() == 1
            print("Local recorder GET, POST, WAV conversion, and manifest: OK")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
