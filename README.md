# Cosmos-Prefer2.5

A unified working environment for NVIDIA's **Cosmos-Predict2.5** and **Cosmos-Transfer2.5** World Foundation Models.

This project provides a single environment with access to both:
- **[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)** - World simulation and video generation
- **[Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)** - Multi-controlnet for spatial control inputs

## System Requirements

- NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
- NVIDIA driver >=570.124.06 compatible with [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)
- Linux x86-64
- glibc>=2.35 (e.g., Ubuntu >=22.04)
- Python 3.10

## Directory Structure

```
/localhome/local-tranminhq/
├── cosmos-predict2.5/    # Cosmos Predict2.5 repository
├── cosmos-transfer2.5/   # Cosmos Transfer2.5 repository
└── cosmos-prefer2.5/     # This unified working folder
    ├── pyproject.toml    # Combined dependencies
    ├── setup.sh          # Setup script
    ├── activate.sh       # Environment activation (generated)
    └── test_setup.py     # Test script (generated)
```

## Quick Start

### 1. Prerequisites

Make sure you have both Cosmos repositories cloned:

```bash
cd /localhome/local-tranminhq

# Clone if not already present
git clone git@github.com:nvidia-cosmos/cosmos-predict2.5.git
git clone git@github.com:nvidia-cosmos/cosmos-transfer2.5.git

# Pull LFS files
cd cosmos-predict2.5 && git lfs pull && cd ..
cd cosmos-transfer2.5 && git lfs pull && cd ..
```

Install system dependencies:

```bash
sudo apt install curl ffmpeg tree wget git-lfs
```

### 2. Run Setup

```bash
cd cosmos-prefer2.5

# Default: CUDA 12.8 with PyTorch 2.7 (Ampere-Hopper GPUs)
bash setup.sh

# Or specify CUDA version:
bash setup.sh cu128    # CUDA 12.8 + PyTorch 2.7 (default)
bash setup.sh cu129    # CUDA 12.9 + PyTorch 2.8
bash setup.sh cu130    # CUDA 13.0 + PyTorch 2.9 (Blackwell)

# Force reinstall all packages:
bash setup.sh --force

# Skip dependency sync (only setup PYTHONPATH):
bash setup.sh --skip-deps
```

### 3. Activate Environment

```bash
source activate.sh
```

### 4. Verify Installation

```bash
python test_setup.py
```

## Usage

After activating the environment, you have access to both Cosmos packages:

### Cosmos-Predict2.5

```bash
# Run inference
python -m cosmos_predict2.inference --help

# Or use the alias
predict-infer --help
```

### Cosmos-Transfer2.5

```bash
# Run inference
python -m cosmos_transfer2.inference --help

# Or use the alias
transfer-infer --help
```

### Python API

```python
# Import both packages
import cosmos_predict2
import cosmos_transfer2

# Access shared utilities
import cosmos_oss

# PyTorch and CUDA
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Environment Variables

After activation, the following environment variables are set:

| Variable | Description |
|----------|-------------|
| `COSMOS_PREDICT_DIR` | Path to cosmos-predict2.5 |
| `COSMOS_TRANSFER_DIR` | Path to cosmos-transfer2.5 |
| `COSMOS_PREFER_DIR` | Path to cosmos-prefer2.5 |
| `PYTHONPATH` | Includes all Cosmos packages |

## Downloading Model Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install Hugging Face CLI: `uv tool install -U "huggingface_hub[cli]"`
3. Login: `hf auth login`
4. Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1)

Checkpoints are automatically downloaded during inference. To change cache location:

```bash
export HF_HOME=/path/to/cache
```

## Package Versions

| CUDA Option | PyTorch | CUDA Toolkit | Target GPUs |
|-------------|---------|--------------|-------------|
| `cu128` | 2.7.1 | 12.8 | Ampere, Ada, Hopper |
| `cu129` | 2.8.0 | 12.9 | Ampere, Ada, Hopper |
| `cu130` | 2.9.0 | 13.0 | Blackwell |

## Troubleshooting

### Import errors

Make sure you've activated the environment:
```bash
source activate.sh
```

### CUDA not available

Check your NVIDIA driver version:
```bash
nvidia-smi
```

Required: driver >=570.124.06

### Package conflicts

Force reinstall with:
```bash
bash setup.sh --force
```

### Missing dependencies

Run the full setup again:
```bash
bash setup.sh
```

## License

- This project: Apache-2.0
- Cosmos models: [NVIDIA Open Model License](https://huggingface.co/nvidia/Cosmos-Guardrail1)

## Links

- [Cosmos-Predict2.5 GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [Cosmos-Transfer2.5 GitHub](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [NVIDIA Cosmos Product Page](https://research.nvidia.com/labs/dir/cosmos-transfer2.5)
- [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)
- [Hugging Face Models](https://huggingface.co/nvidia)
