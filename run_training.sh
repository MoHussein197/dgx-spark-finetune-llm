#!/bin/bash
# =============================================================================
# Fine-tuning Launch Script for SmolLM3-3B with LoRA + FP4/NVFP4
#
# Usage:
#   ./run_training.sh                 # bitsandbytes FP4 (default)
#   ./run_training.sh --use-nvfp4-te  # Transformer Engine NVFP4 (Blackwell)
# =============================================================================

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0  # Adjust for multi-GPU
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Enable TF32 for faster training on Ampere+ GPUs
export NVIDIA_TF32_OVERRIDE=1

# HuggingFace cache (optional - adjust path as needed)
# export HF_HOME=/path/to/cache
# export TRANSFORMERS_CACHE=/path/to/cache

echo "=============================================="
echo "Starting Fine-tuning: SmolLM3-3B + LoRA + NVFP4"
echo "=============================================="
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Run training (pass any arguments to the script)
python finetune.py "$@"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
