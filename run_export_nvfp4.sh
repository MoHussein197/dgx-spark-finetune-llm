#!/bin/bash
# =============================================================================
# Export fine-tuned model to NVFP4 format for TensorRT-LLM
# Optimized for DGX Spark (Blackwell GB10)
# =============================================================================

set -e

# Container with nvidia-modelopt
CONTAINER="nvcr.io/nvidia/pytorch:25.11-py3"
WORKSPACE="$(pwd)"

# Default paths (can override with arguments)
ADAPTER_PATH="${1:-./output/smollm3-3b-reasoning-nvfp4-lora}"
MERGED_PATH="${2:-./output/smollm3-3b-nvfp4-merged}"
NVFP4_PATH="${3:-./output/smollm3-3b-nvfp4}"

echo "=============================================="
echo "Export to NVFP4 for TensorRT-LLM"
echo "=============================================="
echo ""
echo "Container: $CONTAINER"
echo "Adapter:   $ADAPTER_PATH"
echo "Merged:    $MERGED_PATH"
echo "NVFP4:     $NVFP4_PATH"
echo ""

# Check if adapter exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter not found at $ADAPTER_PATH"
    echo "Run training first: ./run_training_docker.sh nvfp4"
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

# Step 1: Merge LoRA adapter with base model
echo "=============================================="
echo "Step 1/2: Merging LoRA adapter with base model"
echo "=============================================="

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WORKSPACE:/workspace" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -w /workspace \
    $CONTAINER \
    bash -c "
        pip install -q --root-user-action=ignore peft 2>/dev/null
        python inference.py --merge --adapter $ADAPTER_PATH --merged-path $MERGED_PATH
    "

echo ""
echo "=============================================="
echo "Step 2/2: Exporting to NVFP4 format"
echo "=============================================="

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WORKSPACE:/workspace" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -w /workspace \
    $CONTAINER \
    bash -c "
        pip install -q --root-user-action=ignore peft nvidia-modelopt datasets 2>/dev/null
        python inference.py --export-nvfp4 --merged-path $MERGED_PATH --nvfp4-path $NVFP4_PATH
    "

echo ""
echo "=============================================="
echo "Export Complete!"
echo "=============================================="
echo ""
echo "Merged model: $MERGED_PATH"
echo "NVFP4 model:  $NVFP4_PATH"
echo ""
echo "To serve with TensorRT-LLM (OpenAI API):"
echo ""
echo "  ./run_serve.sh $NVFP4_PATH"
echo ""
echo "Or manually:"
echo ""
echo "  docker run --rm -it --gpus all \\"
echo "      -v \"\$(pwd)/$NVFP4_PATH:/workspace/model\" \\"
echo "      --ulimit memlock=-1 --ulimit stack=67108864 \\"
echo "      --ipc=host --network host \\"
echo "      nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \\"
echo "      trtllm-serve /workspace/model --backend pytorch --port 8000"
echo ""
