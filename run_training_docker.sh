#!/bin/bash
# =============================================================================
# Fine-tuning with MXFP4/NVFP4 using Docker + Transformer Engine
# Optimized for Blackwell GPUs (GB10, GB100, GB200)
# =============================================================================

set -e

# Container 25.11 has Transformer Engine 2.9+ with NVFP4 support
CONTAINER="nvcr.io/nvidia/pytorch:25.11-py3"
WORKSPACE="$(pwd)"

# Default to NVFP4 (4-bit), can override with: ./run_training_docker.sh mxfp8
QUANT_TYPE="${1:-nvfp4}"

echo "=============================================="
echo "Fine-tuning: SmolLM3-3B + LoRA + ${QUANT_TYPE^^}"
echo "Backend: NVIDIA Transformer Engine (Docker)"
echo "=============================================="
echo ""
echo "Container: $CONTAINER"
echo "Workspace: $WORKSPACE"
echo "Quantization: $QUANT_TYPE"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Pull container if not present
if ! docker image inspect $CONTAINER > /dev/null 2>&1; then
    echo "Pulling container..."
    docker pull $CONTAINER
fi

# Run training inside container
docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WORKSPACE:/workspace" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e TOKENIZERS_PARALLELISM=false \
    -w /workspace \
    $CONTAINER \
    bash -c "
        echo 'Installing dependencies...'
        pip install -q --root-user-action=ignore datasets peft trl accelerate 2>/dev/null
        echo ''
        python finetune.py --use-${QUANT_TYPE}
    "

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Output: ./output/smollm3-3b-reasoning-${QUANT_TYPE}-lora/"
echo "Logs:   ./output/smollm3-3b-reasoning-${QUANT_TYPE}-lora/logs/"
