#!/usr/bin/env bash
# Agent Lab Remote Runner — setup script
# Creates the venv and installs all dependencies including PyTorch for CUDA 12.1.
#
# Usage:
#   bash setup.sh            # install / update in-place
#   bash setup.sh --upgrade  # force-reinstall torch (e.g. after CUDA driver upgrade)

set -euo pipefail

VENV_DIR="$(dirname "$0")/venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

echo "==> Creating/updating venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Installing runner requirements"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$(dirname "$0")/requirements.txt"

echo "==> Installing PyTorch with CUDA 12.1 support"
"$VENV_DIR/bin/pip" install torch torchvision torchaudio --index-url "$TORCH_INDEX"

echo ""
echo "Done. Verify with:"
echo "  $VENV_DIR/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
