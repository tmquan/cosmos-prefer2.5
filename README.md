# Cosmos-Prefer2.5

Preference optimization and fine-tuning utilities for NVIDIA Cosmos-Predict2.5 and Cosmos-Transfer2.5 models.

## 🚀 Quick Start

See [INSTALLATION.md](INSTALLATION.md) for complete step-by-step setup instructions.

### TL;DR

```bash
# 1. Setup directories and clone repos
mkdir -p ~/datasets ~/checkpoints
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5
git clone https://github.com/tmquan/cosmos-prefer2.5

# 2. Start Docker container
sudo docker run --gpus all --ipc=host -d --name cosmos-workspace \
  -v "$HOME/datasets:/workspace/datasets" \
  -v "$HOME/checkpoints:/workspace/checkpoints" \
  -v "$HOME/cosmos-prefer2.5:/workspace/cosmos-prefer2.5" \
  -v "$HOME/cosmos-predict2.5:/workspace/cosmos-predict2.5" \
  -v "$HOME/cosmos-transfer2.5:/workspace/cosmos-transfer2.5" \
  nvcr.io/nvidia/nemo:25.11 sleep infinity

# 3. Enter container and install dependencies
sudo docker exec -it cosmos-workspace bash
cd /workspace/cosmos-prefer2.5
pip install -r requirements.txt

# 4. Login to Hugging Face (requires access token)
huggingface-cli login

# 5. Download models
python download_checkpoints.py --list
python download_checkpoints.py --all --cache-dir /workspace/checkpoints
```

## 📚 Features

- **Checkpoint Downloader**: Easy-to-use script for downloading Cosmos model checkpoints from Hugging Face
- **Docker Support**: Pre-configured setup for NVIDIA NeMo Docker environment
- **Multi-Model Support**: Works with both Predict2.5 and Transfer2.5 models
- **Preference Training**: Tools for fine-tuning models with preference optimization

## 🔧 Tools

### Checkpoint Downloader

```bash
# List available models
python download_checkpoints.py --list

# Download specific model
python download_checkpoints.py --model predict2.5-2b-posttrained

# Download all models
python download_checkpoints.py --all --cache-dir /workspace/checkpoints
```

**Available Models:**
- Cosmos-Predict2.5-2B: Pre-trained, Post-trained, Auto Multiview, Robot Action
- Cosmos-Transfer2.5-2B: Depth, Edge, Segmentation, Auto Multiview

## 📖 Documentation

- [INSTALLATION.md](INSTALLATION.md) - Complete installation guide
- [download_checkpoints.py](download_checkpoints.py) - Checkpoint downloader script

## 🔗 Related Projects

- [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) - World foundation model for video prediction
- [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) - Controllable video generation
- [Hugging Face - Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [Hugging Face - Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)

## ⚙️ Environment

This project is designed to work with:
- Docker Image: `nvcr.io/nvidia/nemo:25.11`
- PyTorch 2.9.0a0 (NVIDIA NeMo build)
- CUDA 13.0
- Python 3.12.3
- 8x NVIDIA B300 GPUs (or similar)

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note:** Access to Cosmos models requires approval from NVIDIA on Hugging Face. Request access at:
- https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
- https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
