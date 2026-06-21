# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MindSpore adaptation of Chronos tokenizer (MeanScaleUniformBins)

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

logger = logging.getLogger(__file__)


@dataclass
class ChronosConfig:
    """
    Holds all configuration parameters for ChronosTokenizer and ChronosModel.
    """
    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    context_length: int
    prediction_length: int
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert self.pad_token_id < self.n_special_tokens
        assert self.eos_token_id < self.n_special_tokens

    def create_tokenizer(self) -> "ChronosTokenizer":
        if self.tokenizer_class == "MeanScaleUniformBins":
            return MeanScaleUniformBins(**self.tokenizer_kwargs, config=self)
        raise ValueError(f"Unknown tokenizer class: {self.tokenizer_class}")


class ChronosTokenizer:
    """Base class for Chronos tokenizers."""
    config: Any  # Expected to be ChronosConfig; set by subclasses

    def context_input_transform(self, context: Tensor) -> Tuple:
        raise NotImplementedError()

    def label_input_transform(self, label: Tensor, tokenizer_state: Any) -> Tuple:
        raise NotImplementedError()

    def output_transform(self, samples: Tensor, tokenizer_state: Any) -> Tensor:
        raise NotImplementedError()


class MeanScaleUniformBins(ChronosTokenizer):
    """Maps time series values to token IDs via mean-scaling + uniform binning."""

    def __init__(self, low_limit: float, high_limit: float, config: ChronosConfig) -> None:
        self.config = config
        self.centers = ops.linspace(
            Tensor(low_limit, ms.float32),
            Tensor(high_limit, ms.float32),
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = ops.concat([
            Tensor([-1e20], ms.float32),
            (self.centers[1:] + self.centers[:-1]) / 2,
            Tensor([1e20], ms.float32),
        ], axis=0)

    def _input_transform(
        self, context: Tensor, scale: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        context = context.astype(ms.float32)
        attention_mask = ~ops.isnan(context)

        if scale is None:
            # scale = sum(|x| * mask) / sum(mask), per batch
            ctx_clean = ops.where(attention_mask, context, ops.zeros_like(context))
            abs_sum = ops.ReduceSum(keep_dims=False)(ops.abs(ctx_clean), -1)
            cnt = ops.maximum(
                ops.ReduceSum(keep_dims=False)(attention_mask.astype(ms.float32), -1),
                Tensor(1.0, ms.float32),
            )
            scale = abs_sum / cnt
            scale = ops.where(scale > 0, scale, ops.ones_like(scale))

        scaled_context = context / scale.unsqueeze(1)
        # Equivalent to torch.bucketize with right=True using searchsorted
        token_ids = ops.searchsorted(self.boundaries, scaled_context, right=True) + self.config.n_special_tokens

        # Clamp to valid range (use same dtype as token_ids, which is int64 from searchsorted)
        token_ids = ops.clamp(token_ids,
                             ops.scalar_to_tensor(0, token_ids.dtype),
                             ops.scalar_to_tensor(self.config.n_tokens - 1, token_ids.dtype))

        # Set padding positions to pad_token_id
        token_ids = ops.where(attention_mask, token_ids,
                             ops.fill(token_ids.dtype, token_ids.shape, self.config.pad_token_id))

        return token_ids, attention_mask, scale

    def _append_eos_token(
        self, token_ids: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size = token_ids.shape[0]
        eos_tokens = ops.fill(token_ids.dtype, (batch_size, 1), self.config.eos_token_id)
        token_ids = ops.concat((token_ids, eos_tokens), axis=1)
        eos_mask = ops.fill(ms.bool_, (batch_size, 1), True)
        attention_mask = ops.concat((attention_mask, eos_mask), axis=1)
        return token_ids, attention_mask

    def context_input_transform(self, context: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        length = context.shape[-1]
        if length > self.config.context_length:
            context = context[..., -self.config.context_length:]

        token_ids, attention_mask, scale = self._input_transform(context=context)

        if self.config.use_eos_token and self.config.model_type == "seq2seq":
            token_ids, attention_mask = self._append_eos_token(token_ids=token_ids, attention_mask=attention_mask)

        return token_ids, attention_mask, scale

    def label_input_transform(self, label: Tensor, scale: Tensor) -> Tuple[Tensor, Tensor]:
        length = label.shape[-1]
        assert length == self.config.prediction_length
        token_ids, attention_mask, _ = self._input_transform(context=label, scale=scale)

        if self.config.use_eos_token:
            token_ids, attention_mask = self._append_eos_token(token_ids=token_ids, attention_mask=attention_mask)

        return token_ids, attention_mask

    def output_transform(self, samples: Tensor, scale: Tensor) -> Tensor:
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = ops.clamp(
            samples - self.config.n_special_tokens - 1,
            Tensor(0, ms.int32),
            Tensor(len(self.centers) - 1, ms.int32),
        )
        return self.centers[indices] * scale_unsqueezed
