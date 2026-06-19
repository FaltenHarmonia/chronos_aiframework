#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/env.sh

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -td outputs/checkpoints/chronos_t5_mini_v100_probe_10k/run-* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${RUN_DIR}" ]]; then
  echo "Mini run directory not found. Pass it as the first argument." >&2
  exit 1
fi

python scripts/evaluation/run_checkpoint_evaluation.py \
  --run_dir "${RUN_DIR}" \
  --model_name mini_v100_probe \
  --checkpoints "${CHECKPOINTS:-checkpoint-final}" \
  --suites "${EVAL_SUITES:-quick-zero-shot,quick-in-domain}" \
  --device "${EVAL_DEVICE:-cuda:0}" \
  --torch_dtype "${EVAL_DTYPE:-float16}" \
  --batch_size "${EVAL_BATCH_SIZE:-8}"
