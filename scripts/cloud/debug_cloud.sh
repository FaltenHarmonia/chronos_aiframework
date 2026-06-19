#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/env.sh

LOG_DIR="outputs/cloud_logs"
mkdir -p "${LOG_DIR}"

python scripts/data/prepare_chronos_corpus_subset.py \
  --output_dir data/processed \
  --data_level debug \
  --min_length 288 \
  --max_length 1024 \
  --seed 42 \
  --compression lz4 \
  2>&1 | tee "${LOG_DIR}/debug_prepare_data.log"

python scripts/data/inspect_arrow_dataset.py \
  --paths data/processed/chronos_debug_tsmixup_900.arrow data/processed/chronos_debug_kernel_synth_100.arrow \
  --freq h \
  --min_length 288 \
  --max_examples 5 \
  2>&1 | tee "${LOG_DIR}/debug_inspect_data.log"

python scripts/training/train.py \
  --config scripts/training/configs/chronos_t5_tiny_debug.yaml \
  2>&1 | tee "${LOG_DIR}/debug_train.log"
