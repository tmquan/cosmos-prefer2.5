#!/usr/bin/env fish
# Cosmos-Prefer2.5 Activation Script for Fish Shell
# Source this file to activate the environment:
#   source activate.fish

# Get the directory where this script is located
set -l SCRIPT_DIR (cd (dirname (status -f)); and pwd)
set -l PARENT_DIR (dirname $SCRIPT_DIR)

# Project paths
set -gx COSMOS_PREDICT_DIR "$PARENT_DIR/cosmos-predict2.5"
set -gx COSMOS_TRANSFER_DIR "$PARENT_DIR/cosmos-transfer2.5"
set -gx COSMOS_PREFER_DIR "$SCRIPT_DIR"

# Activate the virtual environment
if test -f "$SCRIPT_DIR/.venv/bin/activate.fish"
    source "$SCRIPT_DIR/.venv/bin/activate.fish"
else
    echo "Warning: Virtual environment not found. Run setup.sh first."
    return 1
end

# Set PYTHONPATH to include both cosmos packages
set -gx PYTHONPATH "$COSMOS_PREDICT_DIR:$COSMOS_TRANSFER_DIR:$COSMOS_PREDICT_DIR/packages/cosmos-oss:$COSMOS_TRANSFER_DIR/packages/cosmos-oss:$PYTHONPATH"

# Add uv to PATH if not already there
if test -d "$HOME/.local/bin"
    fish_add_path -g "$HOME/.local/bin"
end

# Convenience aliases
alias predict-infer="python -m cosmos_predict2.inference"
alias transfer-infer="python -m cosmos_transfer2.inference"

echo "Cosmos-Prefer2.5 Environment Activated"
echo ""
echo "Python: "(python --version 2>&1)
echo "PyTorch: "(python -c 'import torch; print(torch.__version__)' 2>/dev/null; or echo 'not installed')
echo "CUDA available: "(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null; or echo 'unknown')
echo ""
echo "PYTHONPATH includes:"
echo "  - cosmos-predict2.5"
echo "  - cosmos-transfer2.5"
echo "  - cosmos-oss (shared utilities)"
echo ""
echo "Quick commands:"
echo "  predict-infer  - Run cosmos_predict2 inference"
echo "  transfer-infer - Run cosmos_transfer2 inference"

