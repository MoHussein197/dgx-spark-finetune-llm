# dgx-spark-finetune

[![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.8+-red)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> LLM fine-tuning with LoRA + NVFP4/MXFP8 on NVIDIA DGX Spark (Blackwell GB10)

**Note: This project is a work in progress.**

Fine-tune large language models using LoRA adapters with 4-bit/8-bit quantization optimized for **NVIDIA DGX Spark** and **Blackwell GPUs**.

## Features

- **NVFP4 (4-bit)**: Native Blackwell FP4 training via Transformer Engine
- **MXFP8 (8-bit)**: High-precision training with Transformer Engine
- **bitsandbytes FP4**: Works on any CUDA GPU
- **DGX Spark optimized**: Tested on GB10 (~41GB VRAM for 3B model)
- **LoRA adapters**: Memory-efficient fine-tuning (~240MB output)
- **TensorBoard logging**: Real-time training metrics
- **Extended thinking**: `/think` and `/no_think` modes for reasoning models

## Quantization Backends

| Backend | Bits | VRAM (3B model) | GPU Support | Best For |
|---------|------|-----------------|-------------|----------|
| **bitsandbytes FP4** | 4-bit | ~45GB | Any CUDA GPU | Development |
| **Transformer Engine NVFP4** | 4-bit | ~41GB | Blackwell | Production |
| **Transformer Engine MXFP8** | 8-bit | ~50GB | Blackwell | Higher precision |

## Quick Start

### Option 1: bitsandbytes FP4 (Any GPU)

```bash
# Activate environment
conda activate pytorch

# Train (saves adapter only)
./run_training.sh

# Train + save merged model
python finetune.py --save-merged
```

### Option 2: NVFP4 4-bit (Blackwell - Docker)

```bash
./run_training_docker.sh nvfp4
```

### Option 3: MXFP8 8-bit (Blackwell - Docker)

```bash
./run_training_docker.sh mxfp8
```

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. TRAIN                    2. EXPORT                  3. SERVE            │
│  ──────────────────────      ──────────────────────     ─────────────────── │
│                                                                             │
│  ./run_training_docker.sh    ./run_export_nvfp4.sh      ./run_serve.sh      │
│         nvfp4                                                               │
│           │                         │                         │             │
│           ▼                         ▼                         ▼             │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐    │
│  │ LoRA Adapter    │──────▶│ NVFP4 Model     │──────▶│ OpenAI API      │    │
│  │ (~462MB, bf16)  │       │ (~1.5GB, FP4)   │       │ localhost:8000  │    │
│  └─────────────────┘       └─────────────────┘       └─────────────────┘    │
│                                                                             │
│  Training with TE           Merge + Quantize          TensorRT-LLM          │
│  NVFP4 compute             nvidia-modelopt            OpenAI compatible     │
│  ~41GB VRAM                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Commands

```bash
# Step 1: Train with NVFP4 (Blackwell optimized)
./run_training_docker.sh nvfp4

# Step 2: Export to NVFP4 format for TensorRT-LLM
./run_export_nvfp4.sh

# Step 3: Serve with OpenAI-compatible API
./run_serve.sh

# Step 4: Use the API
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "smollm3-3b-nvfp4", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Test Inference (without serving)

```bash
# Interactive chat with fine-tuned model
./run_inference_docker.sh nvfp4

# Single prompt
./run_inference_docker.sh nvfp4 "" "Explain machine learning"
```

---

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (Blackwell recommended for NVFP4/MXFP4)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Docker (for Transformer Engine)

### Setup Conda Environment

```bash
# Create environment
conda create -n pytorch python=3.11 -y
conda activate pytorch

# Install PyTorch
pip install torch torchvision torchaudio

# Install dependencies
pip install transformers datasets accelerate peft trl bitsandbytes tensorboard

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Setup Docker (for NVFP4/MXFP8)

```bash
# Pull NVIDIA PyTorch container (includes Transformer Engine 2.9+)
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# Pull TensorRT-LLM container for serving (OpenAI API)
docker pull nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

---

## Training Options

### Command Line Arguments

```bash
python finetune.py [OPTIONS]

# Model and dataset
--model ID              # HuggingFace model ID (default: HuggingFaceTB/SmolLM3-3B)
--dataset ID            # HuggingFace dataset ID (default: TeichAI/claude-4.5-opus-high-reasoning-250x)
--output-dir PATH       # Output directory for LoRA adapter (default: ./output/smollm3-3b-reasoning-lora)

# Quantization backends (pick one)
--use-fp4               # bitsandbytes FP4 4-bit (default, any GPU)
--use-nvfp4             # Transformer Engine NVFP4 4-bit (Blackwell + Docker)
--use-mxfp8             # Transformer Engine MXFP8 8-bit (Blackwell + Docker)

# Output options
--save-merged           # Save merged model (adapter + base)
--merged-output PATH    # Path for merged model (default: ./output/smollm3-3b-merged)
```

### Examples

```bash
# Basic training with bitsandbytes FP4 (any GPU)
python finetune.py

# Training with NVFP4 4-bit (Blackwell, inside Docker)
python finetune.py --use-nvfp4

# Training with MXFP8 8-bit (Blackwell, inside Docker)
python finetune.py --use-mxfp8

# Training + save merged model
python finetune.py --use-nvfp4 --save-merged

# Custom model and dataset
python finetune.py --use-nvfp4 --model meta-llama/Llama-3.2-3B --dataset your-org/your-dataset

# Custom output directory
python finetune.py --use-nvfp4 --output-dir ./output/my-custom-lora
```

---

## Inference

### Using Docker Scripts (Recommended)

```bash
# NVFP4 inference - interactive mode
./run_inference_docker.sh nvfp4

# NVFP4 inference - single prompt
./run_inference_docker.sh nvfp4 "" "Explain quantum computing"

# MXFP8 inference
./run_inference_docker.sh mxfp8

# Custom adapter path
./run_inference_docker.sh nvfp4 ./output/my-custom-adapter
```

### Command Line (without Docker)

```bash
# Basic inference (FP4)
python inference.py --adapter ./output/smollm3-3b-reasoning-lora --prompt "Hello"

# Without extended thinking
python inference.py --adapter ./output/smollm3-3b-reasoning-lora --prompt "What is 2+2?" --no-think

# Interactive mode
python inference.py --adapter ./output/smollm3-3b-reasoning-lora
```

### Extended Thinking Mode

The model supports extended thinking with `/think` and `/no_think` flags:

```bash
# Enable thinking (default) - detailed reasoning
python inference.py --prompt "Explain quantum computing"

# Disable thinking - direct answers
python inference.py --prompt "What is 2+2?" --no-think
```

In interactive mode, prefix your prompt:
```
User: /think Explain AI
User: /no_think What is 5+5?
```

### Inference Options

```bash
python inference.py [OPTIONS]

--adapter PATH          # Path to LoRA adapter
--backend [fp4|nvfp4-te]  # Quantization backend
--prompt TEXT           # Single prompt (omit for interactive)
--no-think              # Disable extended thinking mode
--max-tokens INT        # Max new tokens (default: 2048)
--temperature FLOAT     # Temperature (default: 0.7)
--top-p FLOAT           # Top-p sampling (default: 0.9)

# Export options
--merge                 # Merge LoRA with base model
--export-nvfp4          # Export to NVFP4 for TensorRT-LLM
```

---

## Model Export & Serving

### Export to NVFP4 (One Command)

```bash
# Export fine-tuned model to NVFP4 format for TensorRT-LLM
./run_export_nvfp4.sh

# With custom paths
./run_export_nvfp4.sh ./output/smollm3-3b-reasoning-nvfp4-lora ./output/merged ./output/nvfp4
```

This script:
1. Merges LoRA adapter with base model
2. Quantizes to NVFP4 using nvidia-modelopt
3. Outputs TensorRT-LLM compatible model

### Serve with TensorRT-LLM (OpenAI API)

```bash
# Start server on port 8000
./run_serve.sh

# Custom port and batch size
./run_serve.sh ./output/smollm3-3b-nvfp4 8080 8
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Chat completion (OpenAI compatible)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "smollm3-3b-nvfp4",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### Manual Export (without scripts)

```bash
# Merge only
python inference.py --merge --adapter ./output/smollm3-3b-reasoning-nvfp4-lora

# Export to NVFP4
python inference.py --export-nvfp4
```

---

## Output Structure

```
./output/
├── smollm3-3b-reasoning-lora/        # bitsandbytes FP4 adapter (~240MB)
│   └── logs/                         # TensorBoard logs
├── smollm3-3b-reasoning-nvfp4-lora/  # NVFP4 4-bit adapter
│   └── logs/                         # TensorBoard logs
├── smollm3-3b-reasoning-mxfp8-lora/  # MXFP8 8-bit adapter
│   └── logs/                         # TensorBoard logs
├── smollm3-3b-merged/                # Merged model (~6GB)
└── smollm3-3b-nvfp4/                 # NVFP4 export for TensorRT-LLM
```

### Output Size Comparison

| Output | Size | Use Case |
|--------|------|----------|
| LoRA Adapter | ~240MB | Development, multiple versions |
| Merged Model | ~6GB | Production inference |
| NVFP4 Export | ~1.5GB | TensorRT-LLM deployment |

---

## Monitoring

### TensorBoard

```bash
# Monitor specific training
tensorboard --logdir ./output/smollm3-3b-reasoning-mxfp4-lora/logs --port 6006

# Monitor all trainings at once
tensorboard --logdir ./output --port 6006
```

Open http://localhost:6006

---

## Configuration

### Hyperparameters

Edit `finetune.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_SEQ_LENGTH` | 8192 | Max sequence length |
| `per_device_train_batch_size` | 16 | Batch size per GPU |
| `gradient_accumulation_steps` | 1 | Effective batch = batch × accumulation |
| `num_train_epochs` | 3 | Training epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `r` (LoRA rank) | 64 | LoRA rank |
| `lora_alpha` | 128 | LoRA alpha scaling |

### Memory Optimization

For OOM errors:

```python
# In finetune.py
MAX_SEQ_LENGTH = 4096  # Reduce from 8192
per_device_train_batch_size = 1  # Reduce from 2
gradient_accumulation_steps = 8  # Increase to maintain effective batch
```

---

## File Structure

```
.
├── finetune.py               # Main training script
├── inference.py              # Inference + merge + export
├── run_training.sh           # bitsandbytes FP4 (Conda, any GPU)
├── run_training_docker.sh    # NVFP4/MXFP8 training (Docker, Blackwell)
├── run_inference_docker.sh   # Inference with Docker
├── run_export_nvfp4.sh       # Export to NVFP4 for TensorRT-LLM
├── run_serve.sh              # Serve with TensorRT-LLM (OpenAI API)
├── nvfp4.py                  # Custom NVFP4 implementation (reference)
├── quantize_nvfp4_tensorrt.py  # TensorRT export script
└── README.md
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce sequence length and batch size in finetune.py
MAX_SEQ_LENGTH = 4096
per_device_train_batch_size = 1
```

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
# Logout and login again
```

### Transformer Engine Not Found

Use Docker instead of local installation:

```bash
./run_training_nvfp4.sh
```

### HuggingFace Token

For gated models:

```bash
export HF_TOKEN="your_token_here"
```

---

## Workflow Recommendations

### Development (Any GPU)

```bash
# Fast iteration with bitsandbytes FP4
./run_training.sh
python inference.py --adapter ./output/smollm3-3b-reasoning-lora
```

### Production (Blackwell)

```bash
# Full pipeline: Train → Export → Serve
./run_training_docker.sh nvfp4
./run_export_nvfp4.sh
./run_serve.sh
```

### Quick Test

```bash
# Test inference without serving
./run_inference_docker.sh nvfp4
```

---

## References

- [DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) - Official NVIDIA examples
- [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
