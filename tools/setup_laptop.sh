#!/usr/bin/env bash
# koko — one-command workstation setup (x86_64, CUDA, Python >= 3.10).
#
# Installs deps, provisions TTS models, and self-checks ASR + TTS + phonemize.
# The LLM stays remote: set [llm] base_url/model in config.toml
# (llama-server with Qwen is the only thing left on the Jetson).
#
# Usage: tools/setup_laptop.sh [--python 3.11]
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv
uv venv

source .venv/bin/activate
echo "== pip install =="
uv pip install -r requirements-laptop.txt

echo "== TTS models (HF cache: ./hf_cache) =="
tools/fetch_models.sh

echo "== self-check =="
# phonemize self-check needs the endpoint; start one for the check if none is up
STATE=down
curl -s -m2 -o /dev/null -X POST -H 'content-type: application/json' \
    -d '{"text":"x"}' http://127.0.0.1:8788/phonemize || true
curl -s -m2 -o /dev/null http://127.0.0.1:8788/phonemize && STATE=up 2>/dev/null || true
if ! curl -s -m2 -o /dev/null -X POST -H 'content-type: application/json' \
        -d '{"text":"x"}' --fail http://127.0.0.1:8788/phonemize; then
  "$VENV/bin/python" tools/phonemize_server.py & PHM_PID=$!
  trap 'kill $PHM_PID 2>/dev/null' EXIT
  sleep 1
fi
"$VENV/bin/python" tools/check_laptop.py
STATUS=$?
[ -n "${PHM_PID:-}" ] && kill $PHM_PID 2>/dev/null
trap - EXIT
exit $STATUS

cat <<'EOF'

== run it ==

  # terminal 1 — phonemize endpoint (needs sea-g2p; auto-installed above)
  .venv-laptop/bin/python tools/phonemize_server.py

  # terminal 2 — the interpreter (Jetson llama-server for the LLM).
  # First point config.toml's [llm] at your Jetson:
  #   base_url = "http://<jetson-ip>:8081/v1"
  #   model    = "Qwen3.6-35B-A3B"
  .venv-laptop/bin/python main.py
EOF
