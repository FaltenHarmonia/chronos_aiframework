# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MindSpore adaptation of Chronos training script (Tiny/Mini)

import ast
import json
import logging
import os
import random
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Dict, List, Optional

import typer
from typer_config import use_yaml_config
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train import Model as MSTrainer, LossMonitor, TimeMonitor
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint

from gluonts.dataset.common import FileDataset
from gluonts.itertools import Map, Filter
from gluonts.transform import LastValueImputation

from chronos_mindspore.tokenizer import ChronosConfig, MeanScaleUniformBins
from chronos_mindspore.model import ChronosT5Model, ChronosModel, T5Config, get_t5_config
from chronos_mindspore.dataset import ChronosDataset

app = typer.Typer(pretty_exceptions_enable=False)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# ====================== Utilities ======================

def atomic_write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as fp:
        json.dump(payload, fp, indent=4, default=str)
    os.replace(tmp_path, path)


def get_training_job_info() -> Dict:
    info = {
        "mindspore_version": ms.__version__,
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "device": str(context.get_context("device_target")),
        "device_id": context.get_context("device_id"),
    }
    return info


def save_preflight_info(output_dir: Path, training_config: Dict, data_paths: List[str]):
    descriptions = []
    for data_path in data_paths:
        path = Path(data_path)
        item = {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
        if path.exists() and path.is_file():
            item["bytes"] = path.stat().st_size
        descriptions.append(item)
    atomic_write_json(
        output_dir / "training_preflight.json",
        {
            "training_config": training_config,
            "job_info": get_training_job_info(),
            "data_paths": descriptions,
            "cwd": os.getcwd(),
            "argv": sys.argv,
        },
    )


def get_next_run_path(base_dir: Path) -> Path:
    """Get next available run directory."""
    existing = list(base_dir.glob("run-*"))
    nums = [int(d.name.split("-")[-1]) for d in existing if d.is_dir()] + [-1]
    next_num = max(nums) + 1
    run_dir = base_dir / f"run-{next_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def has_enough_observations(entry: dict, min_length: int = 0, max_missing_prop: float = 1.0) -> bool:
    if len(entry["target"]) >= min_length and np.isnan(entry["target"]).mean() <= max_missing_prop:
        return True
    return False


# ====================== Training Callback ======================

class TrainingProgressCallback(Callback):
    """Logs training progress and saves checkpoints."""

    def __init__(self, output_dir: Path, total_steps: int, log_steps: int = 100):
        super().__init__()
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.log_steps = log_steps
        self.start_time = None
        self.step_start_time = None

    def on_train_begin(self, run_context):
        atomic_write_json(self.output_dir / "training_progress.json", {
            "stage": "training_started",
            "max_steps": self.total_steps,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self.start_time = time.time()
        self.step_start_time = time.time()
        logger.info(f"Training started: {self.total_steps} steps")

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        loss = cb_params.net_outputs
        if isinstance(loss, tuple):
            loss = loss[0]
        if step % self.log_steps == 0:
            elapsed = time.time() - self.step_start_time
            steps_per_sec = self.log_steps / elapsed if elapsed > 0 else 0
            logger.info(
                f"Step {step}/{self.total_steps} | Loss: {float(loss):.4f} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            atomic_write_json(self.output_dir / "training_progress.json", {
                "stage": "training",
                "step": step,
                "max_steps": self.total_steps,
                "loss": float(loss),
                "steps_per_second": steps_per_sec,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            self.step_start_time = time.time()

    def on_train_end(self, run_context):
        elapsed = time.time() - self.start_time
        atomic_write_json(self.output_dir / "training_progress.json", {
            "stage": "complete",
            "total_time_seconds": elapsed,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        logger.info(f"Training finished in {elapsed:.1f}s")


# ====================== Main Training Function ======================

@app.command()
@use_yaml_config(param_name="config")
def main(
    # Data
    training_data_paths: str,
    probability: Optional[str] = None,
    dataset_name: str = "unspecified",
    data_level: str = "unspecified",
    # Model config
    model_id: str = "tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    # Training
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 50000,
    save_steps: int = 10000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 16,
    learning_rate: float = 1e-3,
    gradient_accumulation_steps: int = 8,
    # Tokenizer
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    # Sampling
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    # Scheduler
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    # Misc
    drop_prob: float = 0.2,
    max_missing_prop: float = 0.9,
    shuffle_buffer_length: int = 100,
    output_dir: str = "./outputs_mindspore/",
    seed: Optional[int] = None,
    fp16: bool = False,
):
    # ===== Setup =====
    if seed is None:
        seed = random.randint(0, 2**32)
    logger.info(f"Using SEED: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)

    # MindSpore context
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if fp16:
        logger.info("Using mixed precision (float16)")
        # MindSpore automatic mixed precision
        from mindspore import amp
        context.set_context(enable_graph_kernel=True)

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)

    try:
        # ===== Parse configs =====
        training_data_paths = ast.literal_eval(training_data_paths)
        assert isinstance(training_data_paths, list)

        if isinstance(probability, str):
            probability = ast.literal_eval(probability)
        elif probability is None:
            probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
        assert isinstance(probability, list)
        assert len(training_data_paths) == len(probability)

        if isinstance(tokenizer_kwargs, str):
            tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
        assert isinstance(tokenizer_kwargs, dict)

        assert model_type in ["seq2seq", "causal"]

        output_dir = get_next_run_path(output_dir)
        logger.info(f"Output directory: {output_dir}")
        save_preflight_info(output_dir, raw_training_config, training_data_paths)

        # ===== Create ChronosConfig =====
        chronos_config = ChronosConfig(
            tokenizer_class=tokenizer_class,
            tokenizer_kwargs=tokenizer_kwargs,
            n_tokens=n_tokens,
            n_special_tokens=n_special_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_eos_token=use_eos_token,
            model_type=model_type,
            context_length=context_length,
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # ===== Load data =====
        logger.info(f"Loading {len(training_data_paths)} datasets: {training_data_paths}")
        missing_paths = [p for p in training_data_paths if not Path(p).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Data files not found: {missing_paths}")

        train_datasets = [
            Filter(
                partial(has_enough_observations,
                        min_length=min_past + prediction_length,
                        max_missing_prop=max_missing_prop),
                FileDataset(path=Path(data_path), freq="h"),
            )
            for data_path in training_data_paths
        ]
        logger.info(f"Datasets loaded: {len(train_datasets)}")

        # ===== Create model =====
        logger.info(f"Creating model: {model_id}")
        t5_config = get_t5_config(model_id)
        t5_config.model_type = model_type
        t5_config.vocab_size = n_tokens
        t5_config.pad_token_id = pad_token_id
        t5_config.eos_token_id = eos_token_id
        t5_config.decoder_start_token_id = pad_token_id

        inner_model = ChronosT5Model(t5_config)
        model = ChronosModel(chronos_config, t5_config)
        # Set inner model ref correctly
        model.model = inner_model

        total_params = sum(p.size for p in model.get_parameters())
        logger.info(f"Model parameters: {total_params:,}")

        # ===== Create tokenizer and dataset =====
        tokenizer = chronos_config.create_tokenizer()
        dataset = ChronosDataset(
            datasets=train_datasets,
            probabilities=probability,
            tokenizer=tokenizer,
            context_length=context_length,
            prediction_length=prediction_length,
            drop_prob=drop_prob,
            min_past=min_past,
            model_type=model_type,
            imputation_method=LastValueImputation() if model_type == "causal" else None,
            mode="training",
            batch_size=per_device_train_batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
        )

        # ===== Optimizer =====
        optimizer = nn.AdamWeightDecay(
            params=model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=0.01,
        )

        # ===== Loss =====
        # Loss is computed inside model.construct() when labels are provided

        # ===== Training =====
        logger.info(f"Starting training: {max_steps} steps")

        def train_step(data):
            input_ids, attention_mask, labels = data
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            return loss

        # Manual training loop for full control
        model.set_train(True)
        step = 0
        dataloader = iter(dataset)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        start_time = time.time()
        step_time = time.time()

        while step < max_steps:
            try:
                batch = next(dataloader)
            except StopIteration:
                # Reshuffle for infinite training
                dataloader = iter(dataset)
                batch = next(dataloader)

            # Forward + backward
            def compute_loss(batch_data):
                return train_step(batch_data)

            grad_fn = ops.value_and_grad(compute_loss, grad_position=None,
                                         weights=model.trainable_params())
            loss_val, grads = grad_fn(batch)
            optimizer(grads)

            step += 1

            if step % log_steps == 0:
                elapsed = time.time() - step_time
                logger.info(
                    f"Step {step:6d}/{max_steps} | Loss: {float(loss_val):.4f} | "
                    f"{log_steps / elapsed:.1f} steps/s"
                )
                step_time = time.time()
                atomic_write_json(output_dir / "training_progress.json", {
                    "stage": "training",
                    "step": step,
                    "max_steps": max_steps,
                    "loss": float(loss_val),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

            if save_steps > 0 and step % save_steps == 0:
                ckpt_path = checkpoint_dir / f"checkpoint-{step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                ms.save_checkpoint(model, str(ckpt_path / "model.ckpt"))
                logger.info(f"Checkpoint saved at step {step}: {ckpt_path}")

        # ===== Save final checkpoint =====
        final_path = output_dir / "checkpoint-final"
        final_path.mkdir(parents=True, exist_ok=True)
        ms.save_checkpoint(model, str(final_path / "model.ckpt"))
        atomic_write_json(
            final_path / "training_info.json",
            {"training_config": raw_training_config, "job_info": get_training_job_info()},
        )
        atomic_write_json(output_dir / "training_progress.json", {
            "stage": "complete",
            "total_steps": step,
            "final_checkpoint": str(final_path),
            "total_time_seconds": time.time() - start_time,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        logger.info(f"Training complete. Final checkpoint: {final_path}")

    except Exception as exc:
        atomic_write_json(output_dir / "training_failure.json", {
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    app()
