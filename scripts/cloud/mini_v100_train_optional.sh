#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/env.sh

LOG_DIR="outputs/cloud_logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f data/processed/chronos_tsmixup_900k.arrow || ! -f data/processed/chronos_kernel_synth_100k.arrow ]]; then
  echo "V100 data is missing; preparing it before optional Mini training."
  python scripts/data/prepare_chronos_corpus_subset.py \
    --output_dir data/processed \
    --data_level v100 \
    --min_length 576 \
    --max_length 2048 \
    --seed 42 \
    --compression lz4 \
    2>&1 | tee "${LOG_DIR}/mini_v100_prepare_data.log"
fi

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

python scripts/training/train.py \
  --config scripts/training/configs/chronos_t5_mini_v100_probe.yaml \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/mini_v100_train.log"
