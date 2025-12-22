#!/bin/bash
# =============================================================================
# Multi-GPU Fine-tuning Launch Script with Accelerate
# =============================================================================

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1

# Number of GPUs to use (adjust as needed)
NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}

echo "=============================================="
echo "Multi-GPU Fine-tuning: SmolLM3-3B + LoRA + NVFP4"
echo "Using $NUM_GPUS GPU(s)"
echo "=============================================="
echo ""

# Run with accelerate for multi-GPU
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    finetune.py

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
