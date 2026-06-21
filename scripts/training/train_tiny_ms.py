#!/usr/bin/env python3
"""
Self-contained MindSpore Chronos Tiny training script.
No external dependencies beyond mindspore + numpy.
Uses synthetic time series data (mixture of sine, trend, noise).
"""

import sys
import os
import time
import json
import math
import random
from pathlib import Path
from functools import partial

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chronos_mindspore import (
    ChronosConfig,
    ChronosT5Model,
    T5Config,
    get_t5_config,
    MeanScaleUniformBins,
)

# ==================== Config ====================

OUTPUT_DIR = Path("./outputs_mindspore/tiny_run")
SEQ_LEN = 256           # total time series length
CONTEXT_LEN = 128       # history
PRED_LEN = 32           # forecast horizon
N_SERIES = 10000        # total synthetic series
BATCH_SIZE = 8
GRAD_ACCUM = 1          # effective batch = 8
MAX_STEPS = 5000
LOG_STEPS = 10
SAVE_STEPS = 500
LEARNING_RATE = 1e-3
SEED = 42

# ==================== Synthetic Data Generator ====================

class SyntheticTimeSeries:
    """Generate diverse synthetic time series for Chronos training."""
    
    def __init__(self, n_series: int, seq_len: int, seed: int = 42):
        self.n_series = n_series
        self.seq_len = seq_len
        rng = np.random.RandomState(seed)
        self.data = []
        
        for i in range(n_series):
            t = np.arange(seq_len, dtype=np.float32)
            pattern = rng.randint(0, 6)
            
            if pattern == 0:  # Random walk + drift
                ts = np.cumsum(rng.randn(seq_len) * 0.5)
                ts += np.linspace(0, rng.uniform(-2, 2), seq_len)
            elif pattern == 1:  # Sinusoid with noise
                freq = rng.uniform(0.02, 0.2)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.5, 3.0)
                ts = amp * np.sin(freq * t + phase) + rng.randn(seq_len) * 0.3
            elif pattern == 2:  # AR-like
                ts = np.zeros(seq_len, dtype=np.float32)
                ts[:5] = rng.randn(5)
                for j in range(5, seq_len):
                    ts[j] = 0.6 * ts[j-1] + 0.2 * ts[j-2] - 0.1 * ts[j-3] + rng.randn() * 0.5
            elif pattern == 3:  # Piecewise trend
                ts = np.zeros(seq_len)
                n_pieces = rng.randint(1, 5)
                points = sorted(rng.choice(seq_len - 1, n_pieces, replace=False) + 1)
                vals = rng.randn(n_pieces + 1) * 3
                start = 0
                for p, v in zip(points, vals[:-1]):
                    ts[start:p] = np.linspace(v, vals[points.index(p)+1], p-start) if p > start else v
                    start = p
                ts[start:] = np.linspace(vals[-1], vals[-1] + rng.randn()*2, seq_len-start)
                ts += rng.randn(seq_len) * 0.2
            elif pattern == 4:  # Exponential growth/decay
                rate = rng.uniform(-0.02, 0.02)
                noise_level = rng.uniform(0.1, 1.0)
                ts = np.exp(rate * t) * (1 + rng.randn(seq_len) * noise_level * 0.1)
            else:  # Chaos / mixture
                ts = np.cumsum(rng.randn(seq_len) * 0.3)
                ts += 2 * np.sin(0.05 * t) * np.cos(0.02 * t + 1)
                ts += 0.5 * np.sin(0.15 * t + 3)
            
            # Normalize to reasonable range
            ts = ts - np.mean(ts)
            std = np.std(ts)
            if std > 0:
                ts = ts / std * rng.uniform(0.5, 5.0)
            self.data.append(ts.astype(np.float32))

    def __len__(self):
        return self.n_series
    
    def get_batch(self, batch_size: int):
        """Return a batch of (context, future) tensors."""
        indices = np.random.choice(self.n_series, batch_size, replace=True)
        contexts = []
        futures = []
        for idx in indices:
            ts = self.data[idx]
            start = np.random.randint(0, self.seq_len - CONTEXT_LEN - PRED_LEN + 1)
            ctx = ts[start:start + CONTEXT_LEN]
            fut = ts[start + CONTEXT_LEN:start + CONTEXT_LEN + PRED_LEN]
            contexts.append(Tensor(ctx, ms.float32))
            futures.append(Tensor(fut, ms.float32))
        return ops.stack(contexts), ops.stack(futures)


# ==================== Training ====================

def main():
    # Setup
    random.seed(SEED)
    np.random.seed(SEED)
    ms.set_seed(SEED)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "train.log"
    
    def log(msg):
        with open(log_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
            f.flush()
        print(msg, flush=True)
    
    # Configs
    chronos_config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        context_length=CONTEXT_LEN,
        prediction_length=PRED_LEN,
        n_tokens=4096, n_special_tokens=2,
        pad_token_id=0, eos_token_id=1,
        use_eos_token=True, model_type="seq2seq",
        num_samples=20, temperature=1.0, top_k=50, top_p=1.0,
    )
    tokenizer = chronos_config.create_tokenizer()
    
    t5_cfg = get_t5_config("tiny")
    t5_cfg.model_type = "seq2seq"
    t5_cfg.vocab_size = 4096
    t5_cfg.pad_token_id = 0
    t5_cfg.eos_token_id = 1
    t5_cfg.decoder_start_token_id = 0
    
    log(f"Creating Tiny model...")
    model = ChronosT5Model(t5_cfg)
    n_params = sum(p.size for p in model.get_parameters())
    log(f"  Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = nn.AdamWeightDecay(
        params=model.trainable_params(),
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
    )
    
    # Data
    log(f"Generating {N_SERIES} synthetic series (seq_len={SEQ_LEN})...")
    dataset = SyntheticTimeSeries(N_SERIES, SEQ_LEN, SEED)
    log(f"  context={CONTEXT_LEN}, prediction={PRED_LEN}")
    log(f"  batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    log(f"  max_steps={MAX_STEPS}, lr={LEARNING_RATE}")
    
    # Save config
    config_dict = {
        "model": "chronos_t5_tiny", "framework": "mindspore",
        "n_params": n_params, "n_series": N_SERIES,
        "context_length": CONTEXT_LEN, "prediction_length": PRED_LEN,
        "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "max_steps": MAX_STEPS, "learning_rate": LEARNING_RATE,
        "tokenizer": "MeanScaleUniformBins", "n_tokens": 4096,
        "seed": SEED, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Training loop
    model.set_train(True)
    losses = []
    start_time = time.time()
    step_time = time.time()
    
    log(f"\n{'='*60}")
    log(f"Starting training: {MAX_STEPS} steps")
    log(f"{'='*60}")
    
    for step in range(1, MAX_STEPS + 1):
        # Accumulate gradients
        total_loss = 0.0
        
        for accum_step in range(GRAD_ACCUM):
            ctx_batch, fut_batch = dataset.get_batch(BATCH_SIZE)
            
            # Tokenize
            input_ids, attn_mask, scale = tokenizer.context_input_transform(ctx_batch)
            labels, labels_mask = tokenizer.label_input_transform(fut_batch, scale)
            # Mask padding (-100 for cross-entropy ignore)
            labels = ops.where(
                labels_mask.astype(ms.bool_),
                labels,
                ops.fill(labels.dtype, labels.shape, -100)
            )
            
            def compute_loss(inp_ids, mask, labs):
                return model(input_ids=inp_ids, attention_mask=mask, labels=labs)
            
            grad_fn = ops.value_and_grad(
                compute_loss, grad_position=None,
                weights=model.trainable_params()
            )
            loss_val, grads = grad_fn(input_ids, attn_mask, labels)
            
            # Scale loss for gradient accumulation
            scaled_grads = tuple(
                g / GRAD_ACCUM if g is not None else None for g in grads
            )
            optimizer(scaled_grads)
            total_loss += float(loss_val)
        
        avg_loss = total_loss / GRAD_ACCUM
        losses.append(avg_loss)
        
        # Logging
        if step % LOG_STEPS == 0:
            elapsed = time.time() - step_time
            recent_avg = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            total_elapsed = time.time() - start_time
            log(
                f"Step {step:5d}/{MAX_STEPS} | "
                f"Loss: {avg_loss:.4f} (avg100: {recent_avg:.4f}) | "
                f"{LOG_STEPS/elapsed:.1f} steps/s | "
                f"Elapsed: {total_elapsed/60:.1f}min"
            )
            step_time = time.time()
        
        # Save checkpoint
        if step % SAVE_STEPS == 0 or step == MAX_STEPS:
            ckpt_name = f"checkpoint-{step}" if step < MAX_STEPS else "checkpoint-final"
            ckpt_dir = OUTPUT_DIR / ckpt_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ms.save_checkpoint(model, str(ckpt_dir / "model.ckpt"))
            
            # Save training state
            state = {
                "step": step,
                "loss": avg_loss,
                "avg_loss_100": recent_avg,
                "total_time_s": time.time() - start_time,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(OUTPUT_DIR / "training_progress.json", "w") as f:
                json.dump(state, f, indent=2)
            log(f"  Checkpoint saved: {ckpt_name}")
    
    # Done
    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"Training complete!")
    log(f"  Total time: {elapsed/60:.1f}min ({elapsed:.0f}s)")
    log(f"  Final loss: {avg_loss:.4f}")
    log(f"  Checkpoint: {OUTPUT_DIR / 'checkpoint-final'}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
