#!/bin/bash
# =============================================================================
# Inference with fine-tuned model using Docker
# Supports NVFP4 and MXFP8 backends via Transformer Engine
# =============================================================================

set -e

# Container with Transformer Engine
CONTAINER="nvcr.io/nvidia/pytorch:25.11-py3"
WORKSPACE="$(pwd)"

# Default settings
BACKEND="${1:-nvfp4}"
ADAPTER_PATH="${2:-./output/smollm3-3b-reasoning-${BACKEND}-lora}"
PROMPT="${3:-}"

# Map backend names
case $BACKEND in
    nvfp4|nvfp4-te)
        TE_BACKEND="nvfp4-te"
        ;;
    mxfp8|mxfp8-te)
        TE_BACKEND="mxfp8-te"
        ADAPTER_PATH="${2:-./output/smollm3-3b-reasoning-mxfp8-lora}"
        ;;
    fp4)
        TE_BACKEND="fp4"
        ADAPTER_PATH="${2:-./output/smollm3-3b-reasoning-lora}"
        ;;
    *)
        echo "Unknown backend: $BACKEND"
        echo "Usage: ./run_inference_docker.sh [nvfp4|mxfp8|fp4] [adapter_path] [prompt]"
        exit 1
        ;;
esac

echo "=============================================="
echo "Inference: SmolLM3-3B + LoRA + ${BACKEND^^}"
echo "=============================================="
echo ""
echo "Container: $CONTAINER"
echo "Backend:   $TE_BACKEND"
echo "Adapter:   $ADAPTER_PATH"
echo ""

# Check if adapter exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter not found at $ADAPTER_PATH"
    echo "Run training first: ./run_training_docker.sh $BACKEND"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Pull container if not present
if ! docker image inspect $CONTAINER > /dev/null 2>&1; then
    echo "Pulling container..."
    docker pull $CONTAINER
fi

# Build command
if [ -n "$PROMPT" ]; then
    # Single prompt mode
    CMD="python inference.py --backend $TE_BACKEND --adapter $ADAPTER_PATH --prompt \"$PROMPT\""
else
    # Interactive mode
    CMD="python inference.py --backend $TE_BACKEND --adapter $ADAPTER_PATH"
fi

# Run inference
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
        pip install -q --root-user-action=ignore peft 2>/dev/null
        $CMD
    "
