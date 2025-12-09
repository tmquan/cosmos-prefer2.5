# Cosmos-Prefer2.5 🌌

**Simplified toolkit for NVIDIA Cosmos World Foundation Models (Predict2.5 and Transfer2.5)**

Generate high-quality videos from text prompts using state-of-the-art diffusion models.

---

## Quick Start

### 1. Clone Repositories and Start Docker

```bash
# Create directories
mkdir -p ~/datasets ~/checkpoints

# Clone repositories
cd ~
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5  
git clone https://github.com/tmquan/cosmos-prefer2.5

# Start Docker container with HF_HOME set
sudo docker run \
  --gpus all \
  --ipc=host \
 -it \
 -e HF_HOME=/workspace/checkpoints \
  -v "$HOME/datasets:/workspace/datasets" \
  -v "$HOME/checkpoints:/workspace/checkpoints" \
  -v "$HOME/cosmos-prefer2.5:/workspace/cosmos-prefer2.5" \
  -v "$HOME/cosmos-predict2.5:/workspace/cosmos-predict2.5" \
  -v "$HOME/cosmos-transfer2.5:/workspace/cosmos-transfer2.5" \
  nvcr.io/nvidia/nemo:25.11 \
 bash
```

### 2. Setup Environment (Inside Docker)

```bash
cd /workspace/cosmos-prefer2.5
bash setup_environment.sh

# Verify HF_HOME is set (should output: /workspace/checkpoints)
echo $HF_HOME
```

**Note:** The setup script will:
- ✅ Install core dependencies
- ✅ Create cosmos-cuda stub packages
- ✅ Configure PYTHONPATH for all Cosmos modules
- ✅ Install SAM2 (required for Transfer2.5)
- ✅ Create output directories

**Transfer2.5 Requirements:**
The setup script automatically installs SAM2 (Segment Anything Model 2) which is required for Transfer2.5 inference. If you encounter any issues, you can manually install it:

```bash
# Install SAM2 manually if needed
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Verify installation
python -c "import sam2; print('✓ SAM2 installed')"
```

### 3. Download Checkpoints

```bash
# Login to Hugging Face
huggingface-cli login

# Download specific model
python download_checkpoints.py --model predict2.5-2b-posttrained

# Or download all models
python download_checkpoints.py --all
```

**Get access to models here:**
- [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [Cosmos-Predict2.5-14B](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B)
- [Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)

### 4. Download Example Data (Optional)

```bash
# Download example videos and prompts from README
python download_example_data.py
```

This will create `data/inputs/` with:
- Predict2.5 examples (image2world, video2world)
- Transfer2.5 examples (sim2real, real2real with control maps)

### 5. Run Inference

**Predict2.5 (Text-to-World):**
```bash
python cosmos_prefer2/examples/01_basic_predict.py
```

**Transfer2.5 (Video-to-Video with Control):**
```bash
python cosmos_prefer2/examples/02_basic_transfer.py
```

The script will:
- ✅ Use cached checkpoints from `/workspace/checkpoints/hub/`
- ✅ No duplicate downloads (respects `HF_HOME`)
- ✅ Generate a sample video

---

## Usage Examples

### Text-to-World Generation (Simple)

```bash
# Run the example script
python cosmos_prefer2/examples/01_predict_text2world.py
```

### Text-to-World Generation (Python API)

```python
from cosmos_prefer2.utils import check_downloads, write_video
from cosmos_prefer2.inference.predict_helpers import (
    setup_inference_environment,
    build_inference_pipeline,
    yield_video,
)
from pathlib import Path

# Setup environment
original_cwd, cosmos_dir = setup_inference_environment()

# Initialize pipeline (auto-downloads if needed)
pipe = build_inference_pipeline(
    model_name="nvidia/Cosmos-Predict2.5-2B"
)

# Generate video
video = yield_video(
    pipe,
    prompt="A robot arm picks up a red cube on a wooden table",
    num_frames=121,      # ~5 seconds at 24fps
    resolution=(704, 1280),
    guidance=7.5,
    num_steps=50,
    seed=42
)

# Save
output_path = Path("/workspace/outputs/robot_cube.mp4")
write_video(video, output_path, fps=24)
```

### Advanced Options

```python
# High quality generation
video = yield_video(
    pipe,
    prompt="A futuristic city at sunset with flying cars",
    num_frames=121,
    resolution=(704, 1280),
    guidance=10.0,        # Stronger prompt adherence
    num_steps=100,        # More denoising steps
    seed=123
)

# Memory-efficient inference (for GPUs with less VRAM)
pipe = build_inference_pipeline(
    model_name="nvidia/Cosmos-Predict2.5-2B",
    offload_diffusion_model=True,
    offload_text_encoder=True,
)

# Or use lower resolution
video = yield_video(
    pipe,
    prompt="...",
    resolution=(512, 896),  # Lower resolution
    num_steps=25            # Fewer steps
)
```

---

## Available Models

### Cosmos-Predict2.5-2B (Text-to-Video)

```bash
# Base models
python download_checkpoints.py --model predict2.5-2b-pretrained
python download_checkpoints.py --model predict2.5-2b-posttrained   # Recommended

# Specialized models
python download_checkpoints.py --model predict2.5-2b-auto-multiview  # Autonomous driving
python download_checkpoints.py --model predict2.5-2b-robot-action    # Robot action-conditioned
```

### Cosmos-Predict2.5-14B (Larger Model)

```bash
python download_checkpoints.py --model predict2.5-14b-posttrained
```

---

## Key Parameters

| Parameter | Range | Description | Recommended |
|-----------|-------|-------------|-------------|
| `num_steps` | 20-100 | Denoising steps (quality vs speed) | 50 |
| `guidance` | 1.0-15.0 | Prompt adherence strength | 7.5 |
| `num_frames` | 24-241 | Video length in frames | 121 (5 sec) |
| `resolution` | (H, W) | Output resolution | (704, 1280) |
| `seed` | int | Random seed for reproducibility | None |

---

## Tips for Better Results

### Prompt Engineering

✅ **Good prompts:**
- "A red robotic arm picks up a wooden cube on a table"
- "A car driving through a tunnel with bright lights"
- "A drone flying over a city at sunset"

❌ **Avoid:**
- Vague prompts: "something happens"
- Contradictions: "a stationary car driving"
- Too complex: Multiple unrelated actions

### Quality vs Speed

- **Fast preview**: `num_steps=20` (1x time)
- **Balanced**: `num_steps=50` (2.5x time) ⭐ Recommended
- **High quality**: `num_steps=100` (5x time)

### Memory Usage (Approximate)

- Base model: ~10GB VRAM
- With text encoder: ~15GB  
- During generation: ~25GB
- Peak: ~40GB

---

## Troubleshooting

### Out of Memory

```python
# Enable CPU offloading
pipe = build_inference_pipeline(
    model_name="nvidia/Cosmos-Predict2.5-2B",
    offload_diffusion_model=True,
    offload_text_encoder=True
)

# Or use lower resolution
video = generate_video(
    pipe,
    prompt="...",
    resolution=(512, 896),  # Lower resolution
    num_steps=25            # Fewer steps
)
```

### HF_HOME Not Set

If you see checkpoint downloads happening during inference:

```bash
# Make sure HF_HOME is set BEFORE running Python
export HF_HOME=/workspace/checkpoints
echo $HF_HOME

# Then run inference
python cosmos_prefer2/examples/01_predict_text2world.py
```

Or set it in your Docker run command:
```bash
docker run ... -e HF_HOME=/workspace/checkpoints ... nvcr.io/nvidia/nemo:25.11 bash
```

### Module Not Found

```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Re-run setup if needed
cd /workspace/cosmos-prefer2.5
bash setup_environment.sh
```

### Verify Installation

```bash
# Check GPU
nvidia-smi

# Test import
python3 -c "from cosmos_prefer2.inference.predict_helpers import build_inference_pipeline; print('OK')"
```

---

## Repository Structure

```
cosmos-prefer2.5/
├── cosmos_prefer2/              # Main package
│   ├── utils/                   # Utilities
│   │   ├── environment.py       # Environment setup
│   │   └── checkpoint_loader.py # Checkpoint management
│   ├── inference/               # Inference helpers
│   │   └── predict_helpers.py   # Pipeline creation & utilities
│   └── examples/                # Example scripts
│       └── 01_predict_text2world.py  # Simple text-to-world example
├── download_checkpoints.py      # Checkpoint downloader
├── setup_environment.sh         # One-command setup
└── README.md                    # This file
```

---

## How It Works

### Checkpoint Management

The toolkit uses HuggingFace's standard cache structure:

```
/workspace/checkpoints/
└── hub/
    └── models--nvidia--Cosmos-Predict2.5-2B/
        └── snapshots/
            └── {revision}/
                └── base/
                    └── post-trained/
                        └── {checkpoint}.pt
```

### Avoiding Duplicate Downloads

1. **Set `HF_HOME`** before running any Python scripts
2. **Download checkpoints** using `download_checkpoints.py`
3. **Run inference** - it will use the cached checkpoints automatically

### Pipeline Creation

The `build_inference_pipeline()` function:
1. Searches for checkpoints in `HF_HOME/hub/`
2. If found → uses cached checkpoint directly
3. If not found → triggers automatic download via `imaginaire` checkpoint database
4. Returns a ready-to-use `Video2WorldInference` instance

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on top of:
- [NVIDIA Cosmos](https://github.com/nvidia-cosmos) - Foundation models
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Hugging Face](https://huggingface.co/) - Model hosting

---

## Resources

  - [Cosmos-Predict2.5 GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5)
  - [Cosmos-Transfer2.5 GitHub](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [Models on HuggingFace](https://huggingface.co/nvidia)
  - [NVIDIA Cosmos Research](https://research.nvidia.com/labs/dir/cosmos/)

---

**Happy Generating! 🎬**
