#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Audio conversion is unavailable: ffmpeg was not found."
  read -k 1 '?Press any key to close.'
  exit 1
fi
if curl -fsS http://127.0.0.1:8765/prompts.json >/dev/null 2>&1; then
  open http://127.0.0.1:8765/
  exit 0
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Project Python environment is missing."
  read -k 1 '?Press any key to close.'
  exit 1
fi
.venv/bin/python scripts/record_commands_server.py &
recorder_pid=$!
trap 'kill "$recorder_pid" 2>/dev/null' EXIT INT TERM
sleep 1
if ! kill -0 "$recorder_pid" 2>/dev/null; then
  echo "The recorder could not start."
  read -k 1 '?Press any key to close.'
  exit 1
fi
open http://127.0.0.1:8765/
echo "Recorder is running. Keep this window open while recording."
wait "$recorder_pid"
