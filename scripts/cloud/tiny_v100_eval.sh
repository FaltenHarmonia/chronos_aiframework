#!/usr/bin/env bash
set -euo pipefail

source scripts/cloud/env.sh

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -td outputs/checkpoints/chronos_t5_tiny_v100_50k/run-* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${RUN_DIR}" ]]; then
  echo "Tiny run directory not found. Pass it as the first argument." >&2
  exit 1
fi

CHECKPOINTS="${CHECKPOINTS:-checkpoint-10000,checkpoint-25000,checkpoint-50000,checkpoint-final}"
QUICK_SUITES="${QUICK_SUITES:-quick-zero-shot,quick-in-domain}"
FINAL_SUITES="${FINAL_SUITES:-zero-shot,in-domain}"

python scripts/evaluation/run_checkpoint_evaluation.py \
  --run_dir "${RUN_DIR}" \
  --model_name tiny_v100 \
  --checkpoints "${CHECKPOINTS}" \
  --suites "${QUICK_SUITES}" \
  --device "${EVAL_DEVICE:-cuda:0}" \
  --torch_dtype "${EVAL_DTYPE:-float16}" \
  --batch_size "${EVAL_BATCH_SIZE:-16}"

if [[ "${RUN_FULL_FINAL_EVAL:-1}" = "1" ]]; then
  python scripts/evaluation/run_checkpoint_evaluation.py \
    --run_dir "${RUN_DIR}" \
    --model_name tiny_v100_final \
    --checkpoints "checkpoint-final" \
    --suites "${FINAL_SUITES}" \
    --device "${EVAL_DEVICE:-cuda:0}" \
    --torch_dtype "${EVAL_DTYPE:-float16}" \
    --batch_size "${EVAL_BATCH_SIZE:-16}"
fi
