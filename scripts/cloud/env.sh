#!/usr/bin/env bash
set -euo pipefail

mkdir -p cache/huggingface data/processed outputs/cloud_logs outputs/evaluation

export HF_HOME="${HF_HOME:-$(pwd)/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$(pwd)/cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$(pwd)/cache/huggingface/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python --version
nvidia-smi || true
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
