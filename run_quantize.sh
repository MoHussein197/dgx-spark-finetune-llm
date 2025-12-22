#!/bin/bash
# Quantize merged model to NVFP4 for inference

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nvfp4

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

python quantize_nvfp4_tensorrt.py
