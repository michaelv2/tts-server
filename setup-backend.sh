#!/usr/bin/env bash
set -euo pipefail

BACKENDS=("chatterbox" "kokoro" "qwen3" "orpheus" "phonon")

usage() {
    echo "Usage: $0 <backend>"
    echo ""
    echo "Creates a dedicated venv and installs dependencies for a TTS backend."
    echo "Backends: ${BACKENDS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0 qwen3          # creates .venv-qwen3"
    echo "  $0 chatterbox     # creates .venv-chatterbox"
    echo ""
    echo "Then activate with:"
    echo "  source .venv-<backend>/bin/activate"
    exit 1
}

[[ $# -eq 1 ]] || usage

BACKEND="$1"

# Validate backend name
valid=false
for b in "${BACKENDS[@]}"; do
    [[ "$b" == "$BACKEND" ]] && valid=true
done
$valid || { echo "Error: unknown backend '$BACKEND'. Choose from: ${BACKENDS[*]}"; exit 1; }

VENV_DIR=".venv-${BACKEND}"

echo "==> Creating venv: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Installing dependencies for: $BACKEND"
"$VENV_DIR/bin/pip" install --upgrade pip -q
CONSTRAINTS="constraints-${BACKEND}.txt"
if [[ -f "$CONSTRAINTS" ]]; then
    # Pinned, audited versions: keeps brand-new upstream releases out of a fresh venv.
    echo "==> Using pinned versions from $CONSTRAINTS"
    "$VENV_DIR/bin/pip" install -c "$CONSTRAINTS" -e ".[$BACKEND]"
else
    "$VENV_DIR/bin/pip" install -e ".[$BACKEND]"
fi

echo ""
echo "Done. To use:"
echo "  source $VENV_DIR/bin/activate"
echo "  python server.py --model $BACKEND"
