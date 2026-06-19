#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/env.sh

LOG_DIR="outputs/cloud_logs"
mkdir -p "${LOG_DIR}"

python scripts/data/prepare_chronos_corpus_subset.py \
  --output_dir data/processed \
  --data_level v100 \
  --min_length 576 \
  --max_length 2048 \
  --seed 42 \
  --compression lz4 \
  2>&1 | tee "${LOG_DIR}/tiny_v100_prepare_data.log"

python scripts/data/inspect_arrow_dataset.py \
  --paths data/processed/chronos_tsmixup_900k.arrow data/processed/chronos_kernel_synth_100k.arrow \
  --freq h \
  --min_length 576 \
  --max_examples 5 \
  2>&1 | tee "${LOG_DIR}/tiny_v100_inspect_data.log"

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

python scripts/training/train.py \
  --config scripts/training/configs/chronos_t5_tiny_v100.yaml \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/tiny_v100_train.log"
