# Cosmos-Prefer2.5 Installation Guide

A complete step-by-step guide for setting up and running Cosmos-Prefer2.5 with NVIDIA NeMo Docker.

## 📋 Prerequisites

- NVIDIA GPU with CUDA support (tested with NVIDIA B300, 8x GPUs with 275GB memory each)
- NVIDIA Docker (`nvidia-docker2` or Docker with `--gpus` support)
- Hugging Face account with access to gated Cosmos models
- Sufficient disk space for models (~50GB+ for all checkpoints)

## 🚀 Quick Start

### Step 1: Create Directories and Clone Repositories

```bash
# Create necessary directories
mkdir -p ~/datasets
mkdir -p ~/checkpoints

# Clone the Cosmos repositories
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5
git clone https://github.com/tmquan/cosmos-prefer2.5
```

### Step 2: Start Docker Container

```bash
# Run the NVIDIA NeMo Docker container with GPU support
sudo docker run \
  --gpus all \
  --ipc=host \
  -d \
  --name cosmos-workspace \
  -v "$HOME/datasets:/workspace/datasets" \
  -v "$HOME/checkpoints:/workspace/checkpoints" \
  -v "$HOME/cosmos-prefer2.5:/workspace/cosmos-prefer2.5" \
  -v "$HOME/cosmos-predict2.5:/workspace/cosmos-predict2.5" \
  -v "$HOME/cosmos-transfer2.5:/workspace/cosmos-transfer2.5" \
  nvcr.io/nvidia/nemo:25.11 \
  sleep infinity
```

**Container Features:**
- PyTorch 2.9.0a0 (NVIDIA NeMo 25.09 build)
- CUDA 13.0.88
- cuDNN 9.13.1
- Python 3.12.3
- Ubuntu 24.04.3 LTS
- PyTorch Lightning, Transformers, and more pre-installed

### Step 3: Verify Container Setup

```bash
# Check container is running
sudo docker ps | grep cosmos-workspace

# Verify GPU availability
sudo docker exec cosmos-workspace nvidia-smi

# Check PyTorch environment
sudo docker exec cosmos-workspace python -m torch.utils.collect_env
```

### Step 4: Enter the Container

```bash
sudo docker exec -it cosmos-workspace bash
```

Once inside the container, you'll be in `/workspace/` directory with access to:
- `/workspace/datasets` - Your datasets directory
- `/workspace/checkpoints` - Model checkpoints storage
- `/workspace/cosmos-prefer2.5` - This repository
- `/workspace/cosmos-predict2.5` - Cosmos Predict2.5 repository
- `/workspace/cosmos-transfer2.5` - Cosmos Transfer2.5 repository

### Step 5: Install Additional Dependencies

Inside the container:

```bash
cd /workspace/cosmos-prefer2.5

# Install minimal additional requirements
pip install -r requirements.txt
```

### Step 6: Authenticate with Hugging Face

#### Request Model Access (One-time setup)

1. Go to https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
   - Click "Request Access" and wait for approval
2. Go to https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
   - Click "Request Access" and wait for approval

#### Create Access Token

1. Visit https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "cosmos-download")
4. Select **"Read"** permission
5. Copy the generated token

#### Login to Hugging Face

Inside the container:

```bash
huggingface-cli login
```

Paste your token when prompted and press Enter.

### Step 7: Download Model Checkpoints

```bash
cd /workspace/cosmos-prefer2.5

# List available models
python download_checkpoints.py --list

# Download a specific model
python download_checkpoints.py --model predict2.5-2b-posttrained --cache-dir /workspace/checkpoints

# Or download all models (~50GB+)
python download_checkpoints.py --all --cache-dir /workspace/checkpoints
```

**Available Models:**
- `predict2.5-2b-pretrained` - Pre-trained base model (1 file)
- `predict2.5-2b-posttrained` - Post-trained base model (1 file)
- `predict2.5-2b-auto-multiview` - Autonomous driving (2 files)
- `predict2.5-2b-robot-action` - Robot action-conditioned (2 files)
- `transfer2.5-2b-depth` - Depth control (2 files)
- `transfer2.5-2b-edge` - Edge control (2 files)
- `transfer2.5-2b-seg` - Segmentation control (2 files)
- `transfer2.5-2b-auto-multiview` - Autonomous driving transfer (2 files)

## 🔧 Container Management

### Start/Stop Container

```bash
# Stop the container
sudo docker stop cosmos-workspace

# Start the container again
sudo docker start cosmos-workspace

# Remove the container (keeps volumes intact)
sudo docker rm cosmos-workspace
```

### Execute Commands in Container

```bash
# Run a single command
sudo docker exec cosmos-workspace bash -c "cd /workspace && ls -la"

# Enter interactive shell
sudo docker exec -it cosmos-workspace bash
```

## 📊 Verify Installation

Inside the container:

```bash
# Check Python packages
pip list | grep -E "(torch|transformers|diffusers|huggingface)"

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check downloaded checkpoints
ls -lh /workspace/checkpoints/
```

## 🐛 Troubleshooting

### Issue: 401 Client Error when downloading models

**Solution:** You need to authenticate with Hugging Face and request access to the gated models:
1. Request access to both Cosmos repositories on Hugging Face
2. Create and use an access token: `huggingface-cli login`

### Issue: Container exits immediately

**Solution:** The container is running `sleep infinity` to keep it alive. Check logs:
```bash
sudo docker logs cosmos-workspace
```

### Issue: GPU not available in container

**Solution:** Ensure NVIDIA Docker is properly installed:
```bash
# Test GPU access
sudo docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Issue: Permission denied when accessing mounted volumes

**Solution:** The container runs as root. Files created inside will be owned by root. To fix permissions:
```bash
sudo chown -R $USER:$USER ~/checkpoints ~/datasets
```

## 📝 Environment Details

**Hardware Setup (Example):**
- CPU: 2x Intel Xeon 6776P (256 cores total)
- GPU: 8x NVIDIA B300 SXM6 AC (275GB memory each)
- RAM: Varies by system

**Software Stack:**
- Docker Image: `nvcr.io/nvidia/nemo:25.11`
- PyTorch: 2.9.0a0 (NVIDIA build)
- CUDA: 13.0
- Python: 3.12.3

## 🔗 Resources

- [Cosmos Predict2.5 Repository](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [Cosmos Transfer2.5 Repository](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [NVIDIA NeMo Docker Hub](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
- [Hugging Face - Cosmos Predict2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [Hugging Face - Cosmos Transfer2.5](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)

## 💡 Tips

1. **Persistent Storage:** All data in mounted volumes (`~/datasets`, `~/checkpoints`, etc.) persists even if you remove the container.

2. **Multiple Sessions:** You can have multiple terminal sessions connected to the same container:
   ```bash
   # Terminal 1
   sudo docker exec -it cosmos-workspace bash
   
   # Terminal 2 (same container)
   sudo docker exec -it cosmos-workspace bash
   ```

3. **Port Forwarding:** If you need to access Jupyter or TensorBoard, add port mappings:
   ```bash
   sudo docker run ... -p 8888:8888 -p 6006:6006 ... nvcr.io/nvidia/nemo:25.11
   ```

4. **Resource Limits:** You can limit GPU usage:
   ```bash
   sudo docker run --gpus '"device=0,1"' ...  # Only use GPU 0 and 1
   ```

## 🎯 Next Steps

After installation, you can:
1. Explore the example notebooks in `/workspace/cosmos-prefer2.5/`
2. Run inference with the downloaded models
3. Fine-tune models on your custom datasets
4. Experiment with different model configurations

Happy experimenting! 🚀

