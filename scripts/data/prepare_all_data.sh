#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/processed data/eval outputs/logs cache/huggingface

export HF_HOME="${HF_HOME:-$(pwd)/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$(pwd)/cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$(pwd)/cache/huggingface/transformers}"
export TOKENIZERS_PARALLELISM=false

DATA_LEVEL="${1:-debug}"

python scripts/data/prepare_chronos_corpus_subset.py \
  --output_dir data/processed \
  --data_level "${DATA_LEVEL}" \
  --min_length 288 \
  --max_length 2048 \
  --seed 42 \
  --compression lz4

if [ "${DATA_LEVEL}" = "debug" ]; then
  python scripts/data/inspect_arrow_dataset.py \
    --paths data/processed/chronos_debug_tsmixup_900.arrow data/processed/chronos_debug_kernel_synth_100.arrow \
    --freq h \
    --min_length 288 \
    --max_examples 5
else
  python scripts/data/inspect_arrow_dataset.py \
    --paths data/processed/chronos_tsmixup_90k.arrow data/processed/chronos_kernel_synth_10k.arrow \
    --freq h \
    --min_length 288 \
    --max_examples 5
fi
