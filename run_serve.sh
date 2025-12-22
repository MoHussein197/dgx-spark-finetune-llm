#!/bin/bash
# =============================================================================
# Serve NVFP4 model with TensorRT-LLM (OpenAI-compatible API)
# Optimized for DGX Spark (Blackwell GB10)
# =============================================================================

set -e

# TensorRT-LLM container for DGX Spark
CONTAINER="nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"

# Default paths and settings
MODEL_PATH="${1:-./output/smollm3-3b-nvfp4}"
PORT="${2:-8000}"
MAX_BATCH_SIZE="${3:-4}"

echo "=============================================="
echo "TensorRT-LLM Server (OpenAI API)"
echo "=============================================="
echo ""
echo "Container:  $CONTAINER"
echo "Model:      $MODEL_PATH"
echo "Port:       $PORT"
echo "Batch size: $MAX_BATCH_SIZE"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo ""
    echo "Export your model first:"
    echo "  ./run_export_nvfp4.sh"
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

# Get absolute path
ABS_MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"

echo "Starting server..."
echo ""
echo "API Endpoints:"
echo "  - Health:     http://localhost:$PORT/health"
echo "  - Models:     http://localhost:$PORT/v1/models"
echo "  - Chat:       http://localhost:$PORT/v1/chat/completions"
echo "  - Completions: http://localhost:$PORT/v1/completions"
echo ""
echo "Example usage:"
echo ""
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{"
echo "          \"model\": \"smollm3-3b-nvfp4\","
echo "          \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]"
echo "      }'"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="
echo ""

docker run \
    --rm -it \
    --gpus all \
    --ipc=host \
    --network host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$ABS_MODEL_PATH:/workspace/model" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    $CONTAINER \
    trtllm-serve /workspace/model \
        --backend pytorch \
        --max_batch_size $MAX_BATCH_SIZE \
        --port $PORT
