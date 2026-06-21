# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MindSpore adaptation of ChronosDataset for Tiny/Mini training

import itertools
import logging
from functools import partial
from typing import Iterator, List, Optional

import numpy as np
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    LeavesMissingValues,
    LastValueImputation,
)

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

from chronos_mindspore.tokenizer import ChronosTokenizer

logger = logging.getLogger(__file__)


class ChronosDataset:
    """
    Dataset wrapper for Chronos Tiny/Mini training.

    Turns time series data into token IDs for T5 training.
    Uses GluonTS transforms for window slicing; outputs MindSpore Tensors.

    Parameters
    ----------
    datasets
        List of GluonTS FileDataset instances.
    probabilities
        Sampling probabilities for each dataset.
    tokenizer
        ChronosTokenizer instance (MeanScaleUniformBins).
    context_length
        Maximum context window length.
    prediction_length
        Prediction horizon length.
    drop_prob
        Probability of dropping observations during training.
    min_past
        Minimum historical observations required.
    model_type
        "seq2seq" or "causal".
    imputation_method
        Missing value imputation (used for causal models).
    mode
        "training", "validation", or "test".
    np_dtype
        Numpy float data type.
    batch_size
        Number of items per batch for MindSpore GeneratorDataset.
    shuffle_buffer_length
        Buffer size for pseudo-shuffling.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        imputation_method: Optional[MissingValueImputation] = None,
        mode: str = "training",
        np_dtype=np.float32,
        batch_size: int = 16,
        shuffle_buffer_length: int = 100,
    ) -> None:
        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype
        self.batch_size = batch_size
        self.shuffle_buffer_length = shuffle_buffer_length

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if self.model_type == "causal":
            entry["target"] = self.imputation_method(entry["target"])

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice([True, False], size=len(target), p=[drop_p, 1 - drop_p])
            target[mask] = np.nan
            entry["target"] = target

        return entry

    def _create_instance_splitter(self, mode: str):
        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0, min_instances=1,
                min_past=self.min_past, min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter("training") + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        return split_transform.apply(data, is_train=True)

    def create_test_data(self, data):
        return self._create_instance_splitter("test").apply(data, is_train=False)

    def create_validation_data(self, data):
        return self._create_instance_splitter("validation").apply(data, is_train=False)

    def to_hf_format(self, entry: dict) -> dict:
        """Convert GluonTS entry to token IDs."""
        past_target = Tensor(entry["past_target"], ms.float32).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(past_target)

        future_target = Tensor(entry["future_target"], ms.float32).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        # Set masked labels to -100 (HF standard ignore index)
        labels = ops.where(labels_mask, labels, ops.fill(ms.int32, labels.shape, -100))

        if self.model_type == "causal":
            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            # Move padding to right for causal models
            padded_input_ids, obs_input_ids = input_ids[:, pad_start_idx:], input_ids[:, :pad_start_idx]
            padded_attention_mask, obs_attention_mask = attention_mask[:, pad_start_idx:], attention_mask[:, :pad_start_idx]
            # ... simplified: skip the complex causal reordering for now

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def create_data_stream(self):
        """Create the underlying GluonTS data stream."""
        preprocessed_datasets = [
            Map(partial(self.preprocess_entry, mode=self.mode), dataset)
            for dataset in self.datasets
        ]
        if self.mode == "training":
            iterables = [self.create_training_data(d) for d in preprocessed_datasets]
        elif self.mode == "test":
            iterables = [self.create_test_data(d) for d in preprocessed_datasets]
        else:
            iterables = [self.create_validation_data(d) for d in preprocessed_datasets]
        return iterables

    def __iter__(self) -> Iterator:
        """Yields batches of tokenized data."""
        iterables = self.create_data_stream()
        probs = list(self.probabilities)
        probs = [p / sum(probs) for p in probs]
        iterators = list(map(iter, iterables))

        if self.mode == "training":
            while True:
                batch_input_ids = []
                batch_attention_mask = []
                batch_labels = []

                for _ in range(self.batch_size):
                    idx = np.random.choice(range(len(iterators)), p=probs)
                    try:
                        sample = self.to_hf_format(next(iterators[idx]))
                        batch_input_ids.append(sample["input_ids"])
                        batch_attention_mask.append(sample["attention_mask"])
                        batch_labels.append(sample["labels"])
                    except StopIteration:
                        probs[idx] = 0
                        if sum(probs) == 0:
                            return
                        probs = [p / sum(probs) for p in probs]
                        continue

                # Stack to uniform length
                max_len = max(t.shape[0] for t in batch_input_ids)
                padded_input_ids = []
                padded_attention_mask = []
                padded_labels = []
                for i in range(len(batch_input_ids)):
                    pad_len = max_len - batch_input_ids[i].shape[0]
                    if pad_len > 0:
                        pad = ops.fill(ms.int32, (pad_len,), self.tokenizer.config.pad_token_id)
                        input_id = ops.concat([pad, batch_input_ids[i]], axis=0)
                        attn = ops.concat([ops.fill(ms.bool_, (pad_len,), False), batch_attention_mask[i]], axis=0)
                        label = ops.concat([ops.fill(ms.int32, (pad_len,), -100), batch_labels[i]], axis=0)
                    else:
                        input_id = batch_input_ids[i]
                        attn = batch_attention_mask[i]
                        label = batch_labels[i]
                    padded_input_ids.append(input_id)
                    padded_attention_mask.append(attn)
                    padded_labels.append(label)

                yield (
                    ops.stack(padded_input_ids),
                    ops.stack(padded_attention_mask),
                    ops.stack(padded_labels),
                )
        else:
            # Validation/test: sequential
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            for entry in itertools.chain(*iterators):
                sample = self.to_hf_format(entry)
                batch_input_ids.append(sample["input_ids"])
                batch_attention_mask.append(sample["attention_mask"])
                batch_labels.append(sample["labels"])
                if len(batch_input_ids) >= self.batch_size:
                    # Pad and yield batch
                    max_len = max(t.shape[0] for t in batch_input_ids)
                    padded_input_ids, padded_attention_mask, padded_labels = [], [], []
                    for i in range(len(batch_input_ids)):
                        pad_len = max_len - batch_input_ids[i].shape[0]
                        if pad_len > 0:
                            pad = ops.fill(ms.int32, (pad_len,), self.tokenizer.config.pad_token_id)
                            input_id = ops.concat([pad, batch_input_ids[i]], axis=0)
                            attn = ops.concat([ops.fill(ms.bool_, (pad_len,), False), batch_attention_mask[i]], axis=0)
                            label = ops.concat([ops.fill(ms.int32, (pad_len,), -100), batch_labels[i]], axis=0)
                        else:
                            input_id = batch_input_ids[i]
                            attn = batch_attention_mask[i]
                            label = batch_labels[i]
                        padded_input_ids.append(input_id)
                        padded_attention_mask.append(attn)
                        padded_labels.append(label)

                    yield (
                        ops.stack(padded_input_ids),
                        ops.stack(padded_attention_mask),
                        ops.stack(padded_labels),
                    )
                    batch_input_ids, batch_attention_mask, batch_labels = [], [], []

            # Last incomplete batch
            if batch_input_ids:
                yield (
                    ops.stack(batch_input_ids),
                    ops.stack(batch_attention_mask),
                    ops.stack(batch_labels),
                )
