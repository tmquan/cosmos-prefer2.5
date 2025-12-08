# Cosmos-Prefer2.5

A **unified learning and working environment** for NVIDIA's **Cosmos-Predict2.5** and **Cosmos-Transfer2.5** World Foundation Models.

This project provides:
- ✅ **Consolidated environment** with Python 3.12 + CUDA 13.0 + PyTorch 2.9
- ✅ **Comprehensive documentation** of architectures, training, and inference
- ✅ **Step-by-step tutorials** with rich progress bars
- ✅ **Ready-to-use examples** for quick experimentation
- ✅ **Checkpoint management** utilities

## What are World Foundation Models?

World Foundation Models (WFMs) understand and simulate physical world dynamics:
- **[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)** - Generate future video frames from text/image/video
- **[Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)** - Transfer control maps (depth, edge, seg) to diverse scenarios

## System Requirements

- **GPU**: NVIDIA Blackwell (H100/H200), Hopper, or Ampere (A100, RTX 3090/4090)
- **VRAM**: Minimum 24GB (2B model), 80GB recommended (14B model)
- **Driver**: >=525.x compatible with CUDA 13.0
- **OS**: Linux x86-64 (Ubuntu >=22.04)
- **Python**: 3.12
- **CUDA**: 13.0

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

### 1. Clone Repositories

```bash
cd /localhome/local-tranminhq

# Clone Cosmos repositories if not already present
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git

# Pull LFS files
cd cosmos-predict2.5 && git lfs pull && cd ..
cd cosmos-transfer2.5 && git lfs pull && cd ..
```

### 2. Setup Environment

```bash
cd cosmos-prefer2.5

# Install system dependencies (if needed)
sudo apt install curl ffmpeg tree wget git-lfs

# Create conda environment with Python 3.12 + CUDA 13.0
bash setup_env.sh
```

### 3. Activate Environment

```bash
source activate_env.sh
```

This sets up `PYTHONPATH` to include both Cosmos-Predict2.5 and Cosmos-Transfer2.5.

### 4. Download Checkpoints

```bash
# List available models
python download_checkpoints.py --list

# Download specific model
python download_checkpoints.py --model predict2.5-2b-posttrained

# Or download all models
python download_checkpoints.py --all
```

### 5. Run Your First Inference

```bash
# Simple text-to-video generation
cd cosmos_prefer2/examples
python simple_inference.py --prompt "A robot arm picks up a red cube from a wooden table"

# Output: output.mp4
```

## Documentation

This repository includes comprehensive pedagogical documentation:

### 📖 Architecture Deep Dives
- **[Architecture Overview](ARCHITECTURE_README.md)** - Complete system overview
- **[DiT (Diffusion Transformer)](docs/ARCHITECTURE_DiT.md)** - Core generative model
- **[Video Tokenizer (VAE)](docs/ARCHITECTURE_Tokenizer.md)** - Latent space compression
- **[EDM Loss & Rectified Flow](docs/TRAINING_EDM_Loss.md)** - Training objectives

### 🚀 Practical Guides
- **[Step-by-Step Inference Tutorial](docs/INFERENCE_TUTORIAL.md)** - Complete walkthrough with progress bars
- **[Simple Inference Example](cosmos_prefer2/examples/simple_inference.py)** - Ready-to-run script
- **[Batch Inference Example](cosmos_prefer2/examples/batch_inference.py)** - Process multiple prompts

### 📁 Repository Structure

```
cosmos-prefer2.5/
├── docs/                           # Detailed documentation
│   ├── ARCHITECTURE_DiT.md        # DiT transformer architecture
│   ├── ARCHITECTURE_Tokenizer.md  # Video VAE tokenizer
│   ├── TRAINING_EDM_Loss.md       # Training & loss functions
│   └── INFERENCE_TUTORIAL.md      # Step-by-step inference guide
├── cosmos_prefer2/                # Python package
│   ├── __init__.py
│   ├── __about__.py
│   └── examples/                  # Example scripts
│       ├── simple_inference.py    # Basic text2video
│       └── batch_inference.py     # Batch processing
├── checkpoints/                   # Downloaded model weights
├── environment.yaml               # Conda environment (Python 3.12 + CUDA 13.0)
├── pyproject.toml                # Python package config
├── setup_env.sh                  # Environment setup script
├── activate_env.sh               # Activation script (generated)
├── download_checkpoints.py       # Checkpoint downloader
├── ARCHITECTURE_README.md        # Architecture overview
└── README.md                     # This file
```

## Usage Examples

### Text-to-Video (Text2World)

```bash
python cosmos_prefer2/examples/simple_inference.py \
    --prompt "A robot arm picks up a red cube" \
    --output robot_cube.mp4 \
    --frames 121 \
    --guidance 7.5
```

### Batch Processing

```bash
# Create prompts.txt with one prompt per line
python cosmos_prefer2/examples/batch_inference.py \
    --prompts prompts.txt \
    --output-dir outputs/batch/
```

### Python API

```python
import sys
sys.path.insert(0, "/localhome/local-tranminhq/cosmos-predict2.5")

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

# Initialize pipeline
pipe = Video2WorldInference(
    experiment_name="vid2world_cosmos_720p_t24",
    ckpt_path="nvidia/Cosmos-Predict2.5-2B/base/post-trained/...",
    s3_credential_path="",
    context_parallel_size=1,
)

# Generate video
video = pipe.generate_vid2world(
    prompt="Your prompt here",
    num_video_frames=121,
    guidance=7.5,
)
```

## Environment Variables

After activation, the following environment variables are set:

| Variable | Description |
|----------|-------------|
| `COSMOS_PREDICT_DIR` | Path to cosmos-predict2.5 |
| `COSMOS_TRANSFER_DIR` | Path to cosmos-transfer2.5 |
| `COSMOS_PREFER_DIR` | Path to cosmos-prefer2.5 |
| `PYTHONPATH` | Includes all Cosmos packages |

## Key Features

### 🎯 Consolidated Dependencies
- **Single environment** for both Predict and Transfer models
- **Python 3.12** + **CUDA 13.0** + **PyTorch 2.9**
- All dependencies from both repositories merged

### 📚 Pedagogical Documentation
- **Step-by-step explanations** of every component
- **Architecture diagrams** and code walkthroughs
- **Training objectives** (EDM, Rectified Flow) explained
- **Practical examples** with rich progress bars

### 🛠️ Utilities
- **Checkpoint downloader** with progress tracking
- **Inference examples** ready to run
- **Batch processing** support
- **Environment management** scripts

### 🚀 Optimizations
- Flash Attention 2 for memory efficiency
- Context Parallelism for multi-GPU
- Mixed precision (bfloat16)
- Selective activation checkpointing

## Model Checkpoints

### Available Models

| Model | Type | Size | Description |
|-------|------|------|-------------|
| `predict2.5-2b-pretrained` | Predict | 2B | Pre-trained base |
| `predict2.5-2b-posttrained` | Predict | 2B | Post-trained base (recommended) |
| `predict2.5-2b-auto-multiview` | Predict | 2B | Autonomous driving |
| `predict2.5-2b-robot-action` | Predict | 2B | Robot action-conditioned |
| `transfer2.5-2b-depth` | Transfer | 2B | Depth control |
| `transfer2.5-2b-edge` | Transfer | 2B | Edge control |
| `transfer2.5-2b-seg` | Transfer | 2B | Segmentation control |
| `transfer2.5-2b-auto-multiview` | Transfer | 2B | Multi-camera driving |

### Download Instructions

```bash
# List all available models
python download_checkpoints.py --list

# Download specific model
python download_checkpoints.py --model predict2.5-2b-posttrained

# Download to custom directory
python download_checkpoints.py --model transfer2.5-2b-depth --cache-dir ./checkpoints

# Download all models (large!)
python download_checkpoints.py --all
```

### Hugging Face Setup

1. Get [HF Access Token](https://huggingface.co/settings/tokens) with Read permission
2. Login: `huggingface-cli login`
3. Accept [NVIDIA Open Model License](https://huggingface.co/nvidia/Cosmos-Guardrail1)

Checkpoints auto-download during first inference. Change cache location:
```bash
export HF_HOME=/path/to/cache
```

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
