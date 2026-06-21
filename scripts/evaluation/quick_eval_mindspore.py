#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MindSpore adaptation of Chronos evaluation script (Tiny/Mini)

"""
Quick evaluation script for MindSpore Chronos models.
Computes MASE and WQL metrics on synthetic/given data.
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import Tensor, context

from chronos_mindspore import (
    ChronosConfig, ChronosPipeline, T5Config, get_t5_config,
)


def generate_test_data(n_series: int = 200, pred_length: int = 64, seed: int = 42):
    """Generate synthetic test data."""
    np.random.seed(seed)
    contexts = []
    futures = []
    for i in range(n_series):
        t = np.cumsum(np.random.randn(512))
        contexts.append(Tensor(t[:256], ms.float32))
        futures.append(Tensor(t[256:256 + pred_length], ms.float32))
    return contexts, futures


def compute_metrics(predictions, ground_truth, contexts):
    """Compute MASE and WQL."""
    n_series = len(predictions)
    mas_values = []
    wql_values = []

    for i in range(n_series):
        pred = predictions[i]  # (pred_len,)
        truth = ground_truth[i]  # (pred_len,)
        ctx = contexts[i]

        # Seasonal naive using last observed value
        naive_errors = np.abs(truth.asnumpy() - ctx.asnumpy()[-1])
        model_errors = np.abs(truth.asnumpy() - pred.asnumpy())
        in_sample_mad = np.mean(np.abs(np.diff(ctx.asnumpy())))

        if in_sample_mad > 0:
            mas_values.append(np.mean(model_errors) / in_sample_mad)
        wql = np.mean(model_errors / (np.abs(truth.asnumpy()) + 1e-8))
        wql_values.append(wql)

    mase = np.mean(mas_values)
    wql = np.mean(wql_values)
    return mase, wql


def main():
    parser = argparse.ArgumentParser(description="Quick evaluation for MindSpore Chronos")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--n_series", type=int, default=200, help="Number of test series")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "mini", "small"])
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms.set_seed(args.seed)

    # Chronos config (default values matching training)
    chronos_config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        context_length=512,
        prediction_length=args.prediction_length,
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        num_samples=args.num_samples,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    t5_config = get_t5_config(args.model_size)
    t5_config.vocab_size = 4096
    t5_config.pad_token_id = 0
    t5_config.eos_token_id = 1
    t5_config.decoder_start_token_id = 0

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    pipeline = ChronosPipeline.from_pretrained(args.checkpoint, chronos_config, t5_config)

    # Generate test data
    print(f"Generating {args.n_series} test series...")
    contexts, futures = generate_test_data(args.n_series, args.prediction_length, args.seed)

    # Predict
    print("Predicting...")
    start = time.time()
    quantiles, mean = pipeline.predict_quantiles(
        inputs=contexts,
        prediction_length=args.prediction_length,
        num_samples=args.num_samples,
    )
    elapsed = time.time() - start

    # Compute metrics
    predictions_list = [mean[i] for i in range(mean.shape[0])]
    mase, wql = compute_metrics(predictions_list, futures, contexts)

    print("=" * 40)
    print(f"  Series evaluated: {args.n_series}")
    print(f"  MASE = {mase:.4f}")
    print(f"  WQL  = {wql:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    print("=" * 40)

    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "n_series": args.n_series,
        "prediction_length": args.prediction_length,
        "num_samples": args.num_samples,
        "MASE": float(mase),
        "WQL": float(wql),
        "time_seconds": elapsed,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
