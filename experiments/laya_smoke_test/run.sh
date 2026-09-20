#!/usr/bin/env bash
set -eEo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/preflight.py"

if python -c "import torch" >/dev/null 2>&1; then
  python "$SCRIPT_DIR/smoke_loss.py"
else
  printf '%s\n' "SKIP: smoke_loss.py requires PyTorch; no package was installed automatically."
fi
