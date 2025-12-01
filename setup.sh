#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for cosmos-prefer2.5
# This script sets up a unified environment with access to both
# cosmos-predict2.5 and cosmos-transfer2.5 packages

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Project paths
PREDICT_DIR="$PARENT_DIR/cosmos-predict2.5"
TRANSFER_DIR="$PARENT_DIR/cosmos-transfer2.5"
PREFER_DIR="$SCRIPT_DIR"

echo -e "${BLUE}Cosmos-Prefer2.5 Setup Script${NC}"
echo -e "${BLUE}Unified environment for Predict2.5 & Transfer2.5${NC}"
echo ""

# Parse arguments
CUDA_VERSION="${1:-cu128}"
FORCE_REINSTALL=false
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cu128|cu128)
            CUDA_VERSION="cu128"
            shift
            ;;
        --cu129|cu129)
            CUDA_VERSION="cu129"
            shift
            ;;
        --cu130|cu130)
            CUDA_VERSION="cu130"
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  cu128, --cu128    Use CUDA 12.8 with PyTorch 2.7 (default, Ampere-Hopper)"
            echo "  cu129, --cu129    Use CUDA 12.9 with PyTorch 2.8"
            echo "  cu130, --cu130    Use CUDA 13.0 with PyTorch 2.9 (Blackwell)"
            echo "  --force           Force reinstall of all packages"
            echo "  --skip-deps       Skip uv sync (only configure PYTHONPATH)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${YELLOW}Configuration:${NC}"
echo "  CUDA version: $CUDA_VERSION"
echo "  Predict2.5 path: $PREDICT_DIR"
echo "  Transfer2.5 path: $TRANSFER_DIR"
echo "  Working directory: $PREFER_DIR"
echo ""

# Clone repositories if not present
if [[ ! -d "$PREDICT_DIR" ]]; then
    echo -e "${YELLOW}Cloning cosmos-predict2.5...${NC}"
    git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$PREDICT_DIR"
    cd "$PREDICT_DIR" && git lfs pull && cd "$PREFER_DIR"
    echo -e "${GREEN}✓ Cloned cosmos-predict2.5${NC}"
else
    echo -e "${GREEN}✓ Found cosmos-predict2.5${NC}"
fi

if [[ ! -d "$TRANSFER_DIR" ]]; then
    echo -e "${YELLOW}Cloning cosmos-transfer2.5...${NC}"
    git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$TRANSFER_DIR"
    cd "$TRANSFER_DIR" && git lfs pull && cd "$PREFER_DIR"
    echo -e "${GREEN}✓ Cloned cosmos-transfer2.5${NC}"
else
    echo -e "${GREEN}✓ Found cosmos-transfer2.5${NC}"
fi
echo ""

# Check and install git-lfs if not present
if ! command -v git-lfs &> /dev/null; then
    echo -e "${YELLOW}Installing git-lfs...${NC}"
    sudo apt install -y git-lfs
    git lfs install
fi

echo -e "${GREEN}✓ git-lfs is available${NC}"

# Check and install uv if not present
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Installing uv package manager...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
fi

echo -e "${GREEN}✓ uv is available: $(uv --version)${NC}"
echo ""

# Install system dependencies check
echo -e "${YELLOW}Checking system dependencies...${NC}"
MISSING_DEPS=()
for cmd in curl ffmpeg; do
    if ! command -v $cmd &> /dev/null; then
        MISSING_DEPS+=("$cmd")
    fi
done

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Missing system dependencies: ${MISSING_DEPS[*]}${NC}"
    echo "Install with: sudo apt install curl ffmpeg tree wget"
fi

# Sync dependencies with uv
if [[ "$SKIP_DEPS" == "false" ]]; then
    echo ""
    echo -e "${YELLOW}Syncing dependencies with uv (CUDA $CUDA_VERSION)...${NC}"
    echo "This may take a while on first run..."
    echo ""
    
    cd "$PREFER_DIR"
    
    SYNC_OPTS="--extra=$CUDA_VERSION"
    if [[ "$FORCE_REINSTALL" == "true" ]]; then
        SYNC_OPTS="$SYNC_OPTS --reinstall"
    fi
    
    uv sync $SYNC_OPTS
    
    echo -e "${GREEN}✓ Dependencies synced successfully${NC}"
fi

# Create activation script
echo ""
echo -e "${YELLOW}Creating activation script...${NC}"

cat > "$PREFER_DIR/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Cosmos-Prefer2.5 Activation Script
# Source this file to activate the environment:
#   source activate.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Project paths
export COSMOS_PREDICT_DIR="$PARENT_DIR/cosmos-predict2.5"
export COSMOS_TRANSFER_DIR="$PARENT_DIR/cosmos-transfer2.5"
export COSMOS_PREFER_DIR="$SCRIPT_DIR"

# Activate the virtual environment
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "Warning: Virtual environment not found. Run setup.sh first."
    return 1
fi

# Set PYTHONPATH to include both cosmos packages
export PYTHONPATH="$COSMOS_PREDICT_DIR:$COSMOS_TRANSFER_DIR:$COSMOS_PREDICT_DIR/packages/cosmos-oss:$COSMOS_TRANSFER_DIR/packages/cosmos-oss:$PYTHONPATH"

# Add uv to PATH if not already there
if [[ -d "$HOME/.local/bin" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Convenience aliases
alias predict-infer="python -m cosmos_predict2.inference"
alias transfer-infer="python -m cosmos_transfer2.inference"

echo "Cosmos-Prefer2.5 Environment Activated"
echo ""
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo ""
echo "PYTHONPATH includes:"
echo "  - cosmos-predict2.5"
echo "  - cosmos-transfer2.5"
echo "  - cosmos-oss (shared utilities)"
echo ""
echo "Quick commands:"
echo "  predict-infer  - Run cosmos_predict2 inference"
echo "  transfer-infer - Run cosmos_transfer2 inference"
ACTIVATE_EOF

chmod +x "$PREFER_DIR/activate.sh"

echo -e "${GREEN}✓ Activation script created${NC}"

# Create a Python test script
cat > "$PREFER_DIR/test_setup.py" << 'TEST_EOF'
#!/usr/bin/env python3
"""Test script to verify the cosmos-prefer2.5 environment setup."""

import sys
import os

def main():
    print("=" * 60)
    print("Cosmos-Prefer2.5 Environment Test")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check PYTHONPATH
    print("PYTHONPATH entries:")
    for path in sys.path[:10]:
        print(f"  - {path}")
    print()
    
    # Test imports
    print("Testing imports:")
    
    tests = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
    ]
    
    for module, name in tests:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
    
    # Test cosmos packages
    print()
    print("Testing Cosmos packages:")
    
    cosmos_tests = [
        ("cosmos_predict2", "Cosmos Predict2"),
        ("cosmos_transfer2", "Cosmos Transfer2"),
        ("cosmos_oss", "Cosmos OSS"),
    ]
    
    for module, name in cosmos_tests:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
    
    # Test CUDA
    print()
    print("CUDA status:")
    try:
        import torch
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"  Error checking CUDA: {e}")
    
    print()
    print("=" * 60)
    print("Environment test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
TEST_EOF

chmod +x "$PREFER_DIR/test_setup.py"

echo ""
echo -e "${GREEN}Setup Complete!${NC}"
echo ""
echo -e "${BLUE}To activate the environment:${NC}"
echo "  cd $PREFER_DIR"
echo "  source activate.sh"
echo ""
echo -e "${BLUE}To test the setup:${NC}"
echo "  python test_setup.py"
echo ""
echo -e "${BLUE}To run inference:${NC}"
echo "  # Predict2.5 inference:"
echo "  python -m cosmos_predict2.inference --help"
echo ""
echo "  # Transfer2.5 inference:"
echo "  python -m cosmos_transfer2.inference --help"
echo ""

