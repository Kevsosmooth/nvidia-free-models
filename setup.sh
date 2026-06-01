#!/usr/bin/env bash
# One-command setup: makes the virtual environment, installs everything, saves your
# API key, and turns `consult` and `nimchat` into commands you can run from anywhere.
# Safe to run again — it won't overwrite a key you've already saved.
set -euo pipefail
DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$DIR"

echo "==> Setting up in: $DIR"

# 1) Python present?
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed. Install Python 3.9+ and run this again." >&2
  exit 1
fi

# 2) Virtual environment
if [ ! -d .venv ]; then
  echo "==> Creating virtual environment (.venv)"
  python3 -m venv .venv
fi

# 3) Dependencies
echo "==> Installing dependencies (openai, python-dotenv, pillow)"
./.venv/bin/pip install -q --upgrade pip >/dev/null
./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/pip install -q pillow >/dev/null 2>&1 || true   # pillow only needed for vision.py

# 4) API key -> .env
if [ -f .env ] && grep -q '^NVIDIA_API_KEY=nvapi-' .env 2>/dev/null; then
  echo "==> .env already has a key — leaving it as is."
else
  echo
  echo "    Get a FREE key here:  https://build.nvidia.com/settings/api-keys"
  echo "    Steps: sign in / create an account  ->  verify your phone number if asked"
  echo "           ->  click 'Generate API Key'  ->  copy it (it starts with 'nvapi-')"
  echo
  printf "Paste your NVIDIA API key (it stays hidden as you type): "
  read -rs NV_KEY
  echo
  if [ -z "${NV_KEY}" ]; then
    echo "    No key entered. Add it later: put it in a file named .env like .env.example." >&2
  else
    printf 'NVIDIA_API_KEY=%s\nNVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1\n' "$NV_KEY" > .env
    chmod 600 .env
    echo "==> Saved your key to .env (readable only by you)."
  fi
fi

# 5) Make `consult` and `nimchat` runnable from anywhere
BINDIR="$HOME/.local/bin"
mkdir -p "$BINDIR"
ln -sf "$DIR/bin/consult" "$BINDIR/consult"
ln -sf "$DIR/bin/nimchat" "$BINDIR/nimchat"
echo "==> Linked 'consult' and 'nimchat' into $BINDIR"

case ":$PATH:" in
  *":$BINDIR:"*) : ;;
  *)
    echo
    echo "NOTE: $BINDIR is not on your PATH yet. Add this line to your ~/.bashrc, then reopen the terminal:"
    echo "      export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac

echo
echo "All set. Try one of these:"
echo "  consult ask \"what can you help me with?\""
echo "  nimchat \"tell me a joke\""
echo "  consult models            # see every available model"
