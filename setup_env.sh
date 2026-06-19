#!/bin/bash
# chronos_aiframework 环境变量 & 缓存初始化
# source setup_env.sh 后即可使用

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/ma-user/work/cache/huggingface
export HF_DATASETS_CACHE=/home/ma-user/work/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/home/ma-user/work/cache/huggingface/transformers
export TOKENIZERS_PARALLELISM=false

echo "HF mirror: $HF_ENDPOINT"
echo "Cache dir: $HF_HOME"
