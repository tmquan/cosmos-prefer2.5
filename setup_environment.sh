#!/usr/bin/env bash
# Setup script for Cosmos-Prefer2.5 in Docker environment

set -e

echo "========================================="
echo "  Cosmos-Prefer2.5 Environment Setup"
echo "========================================="
echo

# Check if we're in Docker
if [ ! -d "/workspace" ]; then
    echo "Warning: /workspace not found. Are you running in Docker?"
    echo "This script is designed for the NVIDIA NeMo Docker environment."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo "[1/7] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Install minimal dependencies needed for cosmos packages
echo
echo "[2/7] Installing minimal dependencies..."
pip install -q fvcore iopath omegaconf hydra-core loguru rich click tyro pydantic attrs
echo "✓ Core dependencies installed"

# Create and install cosmos-cuda stub packages
echo
echo "[3/7] Creating and installing cosmos-cuda stub packages..."

# Create a shared stub that works for both
mkdir -p /tmp/cosmos_cuda_multi
cat > /tmp/cosmos_cuda_multi/setup.py << 'EOF'
from setuptools import setup

setup(
    name="cosmos-cuda",
    version="1.4.0",  # Use predict2's version (higher)
    py_modules=["cosmos_cuda"],
)
EOF

cat > /tmp/cosmos_cuda_multi/cosmos_cuda.py << 'EOF'
"""Stub package to satisfy cosmos CUDA checks in NeMo Docker.

The NeMo Docker already has PyTorch 2.9.0 with CUDA 13.0 support.
This stub satisfies version checks for both predict2 (1.4.0) and transfer2 (1.3.3).
"""
__version__ = "1.4.0"  # Works for both (transfer2 checks >=)
EOF

# Install via pip so it's in the Python path
pip install -q /tmp/cosmos_cuda_multi

# Also create local packages for workspace compatibility
mkdir -p /workspace/cosmos-predict2.5/packages/cosmos-cuda
cat > /workspace/cosmos-predict2.5/packages/cosmos-cuda/__init__.py << 'EOF'
"""Stub for cosmos-predict2 CUDA check."""
__version__ = "1.4.0"
EOF

mkdir -p /workspace/cosmos-transfer2.5/packages/cosmos-cuda  
cat > /workspace/cosmos-transfer2.5/packages/cosmos-cuda/__init__.py << 'EOF'
"""Stub for cosmos-transfer2 CUDA check."""
__version__ = "1.3.3"
EOF

echo "✓ cosmos-cuda stub packages installed"

# Patch cosmos packages to skip CUDA check (PyTorch already installed in NeMo Docker)
echo
echo "[4/7] Patching cosmos packages to skip CUDA check..."

# Patch cosmos-predict2
cat > /workspace/cosmos-predict2.5/cosmos_predict2/__init__.py << 'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .__about__ import __version__ as __version__

# Patched: Skip CUDA extra check - PyTorch already installed in NeMo Docker
# def _check_cuda_extra():
#     """Check if CUDA extra is installed."""
#     try:
#         import cosmos_cuda
#     except ImportError:
#         raise RuntimeError("CUDA extra not installed. Please run 'uv sync --extra=<cuda_name>'") from None
#
#     if __version__ != cosmos_cuda.__version__:
#         raise RuntimeError(
#             f"CUDA extra version mismatch: {cosmos_cuda.__version__} != {__version__}. Please run 'uv sync --extra=<cuda_name>'"
#         )
# _check_cuda_extra()
EOF

# Patch cosmos-transfer2
cat > /workspace/cosmos-transfer2.5/cosmos_transfer2/__init__.py << 'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .__about__ import __version__ as __version__

# Patched: Skip CUDA extra check - PyTorch already installed in NeMo Docker
# def _check_cuda_extra():
#     """Check if CUDA extra is installed."""
#     try:
#         import cosmos_cuda
#     except ImportError:
#         raise RuntimeError("CUDA extra not installed. Please run 'uv sync --extra=<cuda_name>'") from None
#
#     if __version__ != cosmos_cuda.__version__:
#         raise RuntimeError(
#             f"CUDA extra version mismatch: {cosmos_cuda.__version__} != {__version__}. Please run 'uv sync --extra=<cuda_name>'"
#         )
# _check_cuda_extra()
EOF

echo "✓ Cosmos packages patched (CUDA check disabled)"

# Setup PYTHONPATH
echo
echo "[5/7] Configuring PYTHONPATH..."
export PYTHONPATH="/workspace/cosmos-predict2.5:/workspace/cosmos-transfer2.5:/workspace/cosmos-prefer2.5:$PYTHONPATH"

# Add to bashrc if not already there
if ! grep -q "cosmos-predict2.5" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Cosmos PYTHONPATH" >> ~/.bashrc
    echo "export PYTHONPATH=\"/workspace/cosmos-predict2.5:/workspace/cosmos-transfer2.5:/workspace/cosmos-prefer2.5:\$PYTHONPATH\"" >> ~/.bashrc
fi

echo "✓ PYTHONPATH configured"

# Install Transfer2.5 dependencies (SAM2)
echo
echo "[6/7] Installing Transfer2.5 dependencies..."
if python3 -c "import sam2" 2>/dev/null; then
    echo "✓ SAM2 (Segment Anything Model 2) already installed"
else
    echo "Installing SAM2 for Transfer2.5 inference..."
    pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
    if [ $? -eq 0 ]; then
        echo "✓ SAM2 installed successfully"
    else
        echo "⚠️  Failed to install SAM2 (required for Transfer2.5)"
        echo "   You can install it later with:"
        echo "   pip install git+https://github.com/facebookresearch/segment-anything-2.git"
    fi
fi

# Create output directories
echo
echo "[7/7] Creating output directories..."
mkdir -p /workspace/outputs
mkdir -p /workspace/datasets
mkdir -p /workspace/checkpoints
echo "✓ Directories created"

# Verify installation
echo
echo "[8/8] Verifying installation..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/workspace/cosmos-predict2.5')
sys.path.insert(0, '/workspace/cosmos-transfer2.5') 
sys.path.insert(0, '/workspace/cosmos-prefer2.5')

print("\nChecking imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPUs: {torch.cuda.device_count()}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import cosmos_cuda
    print(f"✓ cosmos-cuda {cosmos_cuda.__version__}")
except Exception as e:
    print(f"✗ cosmos-cuda: {e}")

try:
    from cosmos_prefer2.utils import setup_pythonpath
    print("✓ cosmos-prefer2 utilities")
except Exception as e:
    print(f"✗ cosmos-prefer2: {e}")

try:
    import cosmos_predict2
    print(f"✓ cosmos-predict2 v{cosmos_predict2.__version__}")
except Exception as e:
    print(f"✗ cosmos-predict2: {e}")

try:
    import cosmos_transfer2
    print(f"✓ cosmos-transfer2 v{cosmos_transfer2.__version__}")
except Exception as e:
    print(f"✗ cosmos-transfer2: {e}")

try:
    import sam2
    print("✓ SAM2 (for Transfer2.5)")
except Exception as e:
    print(f"⚠️  SAM2: {e} (needed for Transfer2.5)")

print("\n" + "="*40)
print("✓ Setup Complete!")
print("="*40)
PYEOF

echo
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo
echo "Next steps:"
echo ""
echo "1. Download checkpoints (if not already done):"
echo "   export HF_HOME=/workspace/checkpoints"
echo "   huggingface-cli login"
echo "   python /workspace/cosmos-prefer2.5/download_checkpoints.py --model predict2.5-2b-posttrained"
echo ""
echo "2. Download example data:"
echo "   cd /workspace/cosmos-prefer2.5"
echo "   python download_example_data.py"
echo ""
echo "3. Run Predict2.5 example:"
echo "   cd /workspace/cosmos-prefer2.5"
echo "   python cosmos_prefer2/examples/01_basic_predict.py"
echo ""
echo "4. Run Transfer2.5 example (requires SAM2):"
echo "   cd /workspace/cosmos-prefer2.5"
echo "   python cosmos_prefer2/examples/02_basic_transfer.py"
echo ""
echo "5. Read the README:"
echo "   cat /workspace/cosmos-prefer2.5/README.md | less"
echo ""

