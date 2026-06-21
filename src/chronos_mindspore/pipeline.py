# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MindSpore adaptation of Chronos Pipeline for Tiny/Mini inference

from pathlib import Path
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

from chronos_mindspore.tokenizer import ChronosConfig, MeanScaleUniformBins
from chronos_mindspore.model import ChronosModel, ChronosT5Model, T5Config
from chronos_mindspore.utils import left_pad_and_stack_1D


class ChronosPipeline:
    """
    Pipeline for Chronos time series forecasting with MindSpore.

    Use ``predict`` to get sample forecasts.
    Use ``predict_quantiles`` to get quantile forecasts.
    """

    def __init__(self, model: ChronosModel, config: ChronosConfig):
        self.model = model
        self.config = config
        self.tokenizer = config.create_tokenizer()

    @classmethod
    def from_pretrained(cls, checkpoint_path: Union[str, Path],
                        chronos_config: ChronosConfig,
                        t5_config: T5Config) -> "ChronosPipeline":
        """Load a model from a MindSpore checkpoint."""
        inner_model = ChronosT5Model(t5_config)
        model = ChronosModel(chronos_config, t5_config)
        model.model = inner_model

        ckpt_path = Path(checkpoint_path)
        if ckpt_path.is_dir():
            ckpt_file = str(ckpt_path / "model.ckpt")
        else:
            ckpt_file = str(ckpt_path)

        param_dict = ms.load_checkpoint(ckpt_file)
        ms.load_param_into_net(model, param_dict)
        return cls(model=model, config=chronos_config)

    def _prepare_context(self, context: Union[Tensor, List[Tensor]]) -> Tensor:
        """Prepare and validate context tensor."""
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
        return context

    def predict(
        self,
        inputs: Union[Tensor, List[Tensor]],
        prediction_length: Optional[int] = None,
        num_samples: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> Tensor:
        """
        Generate sample forecasts.

        Returns tensor of shape (batch_size, num_samples, prediction_length).
        """
        context_tensor = self._prepare_context(inputs)

        if prediction_length is None:
            prediction_length = self.config.prediction_length

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            token_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)

            samples = self.model.model.generate(
                input_ids=token_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(remaining, self.config.prediction_length),
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=self.config.pad_token_id,
                num_return_sequences=num_samples,
            )

            batch_size = token_ids.shape[0]
            samples = samples.view(batch_size, num_samples, -1)

            prediction = self.tokenizer.output_transform(samples, scale)
            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            # Use median as next context for autoregressive prediction
            context_tensor = ops.concat(
                [context_tensor, prediction.median(axis=1)[0]], axis=-1
            )

        return ops.concat(predictions, axis=-1).astype(ms.float32)

    def predict_quantiles(
        self,
        inputs: Union[Tensor, List[Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        num_samples: int = 20,
        **predict_kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get quantile forecasts.

        Returns (quantiles, mean), shapes:
        - quantiles: (batch_size, prediction_length, len(quantile_levels))
        - mean: (batch_size, prediction_length)
        """
        samples = self.predict(
            inputs,
            prediction_length=prediction_length,
            num_samples=num_samples,
            **predict_kwargs,
        )

        # samples: (batch, num_samples, pred_len)
        # Swap for per-time quantile computation
        samples_transposed = samples.swapaxes(1, 2)  # (batch, pred_len, num_samples)

        mean = samples_transposed.mean(axis=-1)

        q_tensor = Tensor(quantile_levels, ms.float32)
        # Compute quantiles across sample axis
        quantiles = ops.quantile(samples_transposed, q_tensor, axis=-1).swapaxes(0, 1).swapaxes(1, 2)
        # quantiles: (batch, pred_len, num_quantiles)

        return quantiles, mean

    def embed(
        self,
        context: Union[Tensor, List[Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        """Get encoder embeddings for the given context."""
        context_tensor = self._prepare_context(context)
        token_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)
        embeddings = self.model.encode(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )
        return embeddings, scale
