#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <model> [--host HOST] [--port PORT]"
    echo ""
    echo "Starts the TTS server using the backend's venv (.venv-<backend>)."
    echo "Models: chatterbox kokoro qwen3-0.6b qwen3-1.7b qwen3-0.6b-design qwen3-1.7b-design orpheus phonon phonon-40m"
    echo ""
    echo "Examples:"
    echo "  $0 kokoro"
    echo "  $0 qwen3-1.7b --port 9200"
    exit 1
}

[[ $# -ge 1 ]] || usage

MODEL="$1"
shift

# Map model name to the backend venv created by setup-backend.sh
case "$MODEL" in
    chatterbox|kokoro|orpheus) BACKEND="$MODEL" ;;
    qwen3-*)                   BACKEND="qwen3" ;;
    phonon|phonon-40m)         BACKEND="phonon" ;;
    -h|--help)                 usage ;;
    *) echo "Error: unknown model '$MODEL'"; usage ;;
esac

cd "$(dirname "$0")"

VENV_DIR=".venv-${BACKEND}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Error: $VENV_DIR not found. Run: ./setup-backend.sh $BACKEND"
    exit 1
fi

exec "$VENV_DIR/bin/python" server.py --model "$MODEL" "$@"
