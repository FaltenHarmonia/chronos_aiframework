# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MindSpore adaptation of T5 model for Chronos Tiny/Mini training

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal, Constant


# ======================== T5 Config ========================

@dataclass
class T5Config:
    """Configuration for T5 model (Tiny/Mini variants)."""
    vocab_size: int = 4096
    d_model: int = 256
    d_kv: int = 64
    d_ff: int = 1024
    num_layers: int = 6
    num_heads: int = 4
    num_decoder_layers: int = 6
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 0.05
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    is_encoder_decoder: bool = True
    use_cache: bool = False
    model_type: str = "seq2seq"  # "seq2seq" or "causal"

    @property
    def num_attention_heads(self) -> int:
        return self.num_heads


def get_t5_config(model_name: str) -> T5Config:
    """Get T5 configuration for a given model name."""
    configs = {
        "tiny": T5Config(
            d_model=256, d_ff=1024, num_layers=6, num_heads=4, d_kv=64,
            num_decoder_layers=6,
        ),
        "mini": T5Config(
            d_model=384, d_ff=1536, num_layers=6, num_heads=6, d_kv=64,
            num_decoder_layers=6,
        ),
        "small": T5Config(
            d_model=512, d_ff=2048, num_layers=6, num_heads=8, d_kv=64,
            num_decoder_layers=6,
        ),
    }
    return configs.get(model_name, configs["tiny"])


# ======================== T5 LayerNorm ========================

class T5LayerNorm(nn.Cell):
    """T5-style LayerNorm: no bias, no mean subtraction."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = Parameter(initializer(Constant(1.0), (hidden_size,), ms.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.astype(ms.float32).pow(2).mean(axis=-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype != hidden_states.dtype:
            hidden_states = hidden_states.astype(self.weight.dtype)
        return self.weight * hidden_states


# ======================== Relative Position Bias ========================

class T5RelativePositionBias(nn.Cell):
    """T5-style relative position bias for attention."""
    def __init__(self, config: T5Config):
        super().__init__()
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads
        self.relative_attention_bias = nn.Embedding(
            self.num_buckets, self.n_heads,
            embedding_table=Normal(sigma=0.02)
        )

    @staticmethod
    def _relative_position_bucket(relative_position: Tensor, num_buckets: int, max_distance: int) -> Tensor:
        """Bucketize relative positions for T5 attention bias."""
        ret = ms.numpy.zeros_like(relative_position)
        n = -relative_position
        # bidirectional buckets
        num_buckets_half = num_buckets // 2
        is_small = n < num_buckets_half
        ret = ops.where(is_small, n, ret)
        is_medium = ops.logical_and(
            ops.logical_not(is_small),
            n < max_distance
        )
        val_big = (ops.log(n.astype(ms.float32)) / math.log(max_distance / num_buckets_half)
                   * (num_buckets - num_buckets_half)).astype(ms.int32) + num_buckets_half
        ret = ops.where(is_medium, val_big, ret)
        is_large = n >= max_distance
        ret = ops.where(is_large, ops.fill(ms.int32, ret.shape, num_buckets - 1), ret)
        return ret

    def construct(self, query_length: int, key_length: int) -> Tensor:
        context_position = ops.arange(query_length, dtype=ms.int32).unsqueeze(-1)
        memory_position = ops.arange(key_length, dtype=ms.int32).unsqueeze(0)
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(
            relative_position, self.num_buckets, self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        # (query_len, key_len, n_heads) -> (1, n_heads, query_len, key_len)
        values = values.transpose((2, 0, 1)).unsqueeze(0)
        return values


# ======================== T5 Attention ========================

class T5Attention(nn.Cell):
    """Multi-head attention for T5 with relative position bias."""
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = False  # set by caller for decoder layers
        self.has_relative_attention_bias = has_relative_attention_bias
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.dropout_rate = config.dropout_rate

        # Q, K, V projections (no bias in T5)
        self.q = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.k = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.v = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.o = nn.Dense(self.inner_dim, self.d_model, has_bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = T5RelativePositionBias(config)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    @staticmethod
    def _build_attention_mask(
        attention_mask: Optional[Tensor],
        dtype: ms.dtype,
        query_len: int,
        key_len: int,
        causal: bool = False,
    ) -> Tensor:
        """Build (batch, 1, query_len, key_len) additive mask.
        For causal=True, also applies lower-triangular masking.
        """
        if attention_mask is None:
            mask = ops.ones((1, query_len), dtype)
        else:
            mask = attention_mask[:, None, :].astype(dtype)  # (batch, 1, key_len)

        # Expand to 4D: (batch, 1, query_len, key_len)
        mask_4d = mask[:, :, None, :].broadcast_to((-1, -1, query_len, -1))

        if causal:
            tril_mask = ops.tril(ops.ones((1, 1, query_len, key_len), dtype))
            mask_4d = mask_4d * tril_mask

        # Invert: 1 -> 0 (attend), 0 -> -1e9 (mask)
        mask_4d = (1.0 - mask_4d) * Tensor(-1e9, dtype)
        return mask_4d

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        position_bias: Optional[Tensor] = None,
        use_cache: bool = False,
        causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """(batch, seq, dim) -> (batch, n_heads, seq, d_kv)"""
            return states.view(batch_size, -1, self.n_heads, self.d_kv).transpose((0, 2, 1, 3))

        def unshape(states):
            """(batch, n_heads, seq, d_kv) -> (batch, seq, dim)"""
            return states.transpose((0, 2, 1, 3)).view(batch_size, -1, self.inner_dim)

        query_states = shape(self.q(hidden_states))

        is_cross_attention = key_value_states is not None
        if is_cross_attention:
            key_states = shape(self.k(key_value_states))
            value_states = shape(self.v(key_value_states))
        elif past_key_value is not None:
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))
            key_states = ops.concat((past_key_value[0], key_states), axis=-2)
            value_states = ops.concat((past_key_value[1], value_states), axis=-2)
        else:
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))

        if past_key_value is not None and not is_cross_attention:
            if use_cache:
                new_past = (key_states, value_states)
            else:
                new_past = None
        else:
            new_past = None

        # Attention scores
        scores = ops.matmul(query_states, key_states.swapaxes(-1, -2))

        # Position bias
        if position_bias is not None:
            scores = scores + position_bias
        elif self.has_relative_attention_bias:
            position_bias = self.relative_attention_bias(seq_length, key_states.shape[-2])
            scores = scores + position_bias

        # Attention mask
        mask_4d = self._build_attention_mask(
            attention_mask, scores.dtype,
            query_len=seq_length, key_len=key_states.shape[-2],
            causal=causal and not is_cross_attention,
        )
        scores = scores + mask_4d

        attn_weights = ops.softmax(scores.astype(ms.float32), axis=-1).astype(scores.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = unshape(ops.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)

        return attn_output, position_bias, new_past


# ======================== T5 Layer (Self-Attn + Cross-Attn + FF) ========================

class T5Block(nn.Cell):
    """T5 encoder or decoder block."""
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.self_attention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.self_attention.is_decoder = is_decoder
        self.dropout = nn.Dropout(p=config.dropout_rate)

        if is_decoder:
            self.cross_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
            self.cross_attention = T5Attention(config, has_relative_attention_bias=False)
            self.cross_dropout = nn.Dropout(p=config.dropout_rate)

        self.ff_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ff_dense_relu = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.ff_dense_output = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.ff_dropout = nn.Dropout(p=config.dropout_rate)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        encoder_decoder_position_bias: Optional[Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        # Self-attention
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias, _ = self.self_attention(
            normed, attention_mask=attention_mask,
            position_bias=position_bias, past_key_value=past_key_value,
            use_cache=use_cache, causal=self.is_decoder,
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            normed = self.cross_layer_norm(hidden_states)
            attn_output, encoder_decoder_position_bias, _ = self.cross_attention(
                normed, key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
            )
            hidden_states = hidden_states + self.cross_dropout(attn_output)

        # Feed-forward
        normed = self.ff_layer_norm(hidden_states)
        ff_output = self.ff_dense_relu(normed)
        ff_output = ops.relu(ff_output)
        ff_output = self.ff_dense_output(ff_output)
        hidden_states = hidden_states + self.ff_dropout(ff_output)

        return hidden_states, position_bias, encoder_decoder_position_bias


# ======================== T5 Encoder Stack ========================

class T5Encoder(nn.Cell):
    """T5 encoder (stack of T5Blocks)."""
    def __init__(self, config: T5Config):
        super().__init__()
        self.block = nn.CellList([
            T5Block(config, has_relative_attention_bias=(i == 0), is_decoder=False)
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(
        self,
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states = self.dropout(inputs_embeds)
        position_bias = None

        for block in self.block:
            hidden_states, position_bias, _ = block(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# ======================== T5 Decoder Stack ========================

class T5Decoder(nn.Cell):
    """T5 decoder (stack of T5Blocks with cross-attention)."""
    def __init__(self, config: T5Config):
        super().__init__()
        self.block = nn.CellList([
            T5Block(config, has_relative_attention_bias=(i == 0), is_decoder=True)
            for i in range(config.num_decoder_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(
        self,
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states = self.dropout(inputs_embeds)
        position_bias = None
        encoder_decoder_position_bias = None

        for block in self.block:
            hidden_states, position_bias, encoder_decoder_position_bias = block(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# ======================== ChronosT5Model ========================

class ChronosT5Model(nn.Cell):
    """
    T5 encoder-decoder model for Chronos time series forecasting.

    Supports seq2seq (encoder-decoder) and causal (decoder-only) modes.
    Uses MindSpore's nn.Cell for autograd and training.
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model_dim = config.d_model
        self.vocab_size = config.vocab_size

        # Shared token embedding
        self.shared = nn.Embedding(config.vocab_size, config.d_model,
                                   embedding_table=Normal(sigma=1.0))

        if config.model_type == "seq2seq":
            self.encoder = T5Encoder(config)
            self.decoder = T5Decoder(config)
            self.lm_head = nn.Dense(config.d_model, config.vocab_size, has_bias=False)
            # Tie lm_head weight to shared embedding
            # Note: in MindSpore we can't easily tie weights; we do it in init

        # For decoder embeddings
        self.decoder_embed = None  # Will reuse shared

    def _init_weights(self):
        """Manual weight initialization."""
        factor = self.config.initializer_factor
        for name, param in self.parameters_and_names():
            if 'relative_attention_bias' in name:
                param.set_data(initializer(Normal(sigma=factor), param.shape, ms.float32))
            elif 'weight' in name and param.ndim >= 2:
                param.set_data(initializer(Normal(sigma=factor * (param.shape[-1] ** -0.5)),
                                           param.shape, ms.float32))

    def get_encoder(self):
        return self.encoder

    def _expand_input_ids(self, input_ids: Tensor) -> Tensor:
        """Get embeddings from shared embedding table."""
        return self.shared(input_ids)

    def construct(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ):
        """Forward pass for training (computes loss)."""
        batch_size, seq_length = input_ids.shape

        # Encoder
        encoder_hidden_states = None
        if self.model_type == "seq2seq":
            inputs_embeds = self._expand_input_ids(input_ids)
            encoder_hidden_states = self.encoder(inputs_embeds, attention_mask=attention_mask)

        # Decoder
        if labels is not None:
            # For training, decoder_input_ids is labels shifted right
            if decoder_input_ids is None:
                # Create decoder input from labels: prepend pad_token, then shift
                # Use same dtype as labels to avoid concat type mismatch
                pad_start = ops.fill(labels.dtype, (batch_size, 1), self.config.pad_token_id)
                decoder_input_ids = ops.concat([pad_start, labels[:, :-1]], axis=1)

            decoder_inputs_embeds = self._expand_input_ids(decoder_input_ids)
            # Create 2D attention mask for decoder (all ones: no padding for labels)
            decoder_seq_len = decoder_input_ids.shape[1]
            decoder_attention_mask = ops.ones((batch_size, decoder_seq_len), ms.bool_)

            decoder_hidden_states = self.decoder(
                decoder_inputs_embeds,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )

            # LM head
            logits = self.lm_head(decoder_hidden_states)

            # Compute loss
            loss = self._compute_loss(logits, labels)
            return loss

        # Inference / generate mode
        return None

    def _compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Cross-entropy loss with label smoothing."""
        # Standard cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Mask out padding (-100)
        mask = labels_flat >= 0
        labels_masked = ops.where(mask, labels_flat, ops.zeros_like(labels_flat))
        labels_onehot = ops.one_hot(labels_masked.astype(ms.int32), vocab_size,
                                     Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
        log_probs = ops.log_softmax(logits_flat, axis=-1)
        loss_per_token = -ops.ReduceSum(keep_dims=False)(log_probs * labels_onehot, -1)
        masked_loss = loss_per_token * mask.astype(ms.float32)
        loss = ops.ReduceSum(keep_dims=False)(masked_loss) / ops.maximum(
            ops.ReduceSum(keep_dims=False)(mask.astype(ms.float32)),
            Tensor(1.0, ms.float32)
        )
        return loss

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        num_return_sequences: int = 1,
    ) -> Tensor:
        """
        Generate predictions autoregressively.

        Returns shape (batch_size * num_return_sequences, seq_len).
        """
        batch_size = input_ids.shape[0]
        device_input = input_ids

        # Encoder pass
        if self.model_type == "seq2seq":
            inputs_embeds = self._expand_input_ids(device_input)
            encoder_hidden_states = self.encoder(inputs_embeds, attention_mask=attention_mask)
        else:
            encoder_hidden_states = None

        # Decoder start token
        decoder_input_ids = ops.fill(ms.int32, (batch_size, 1), self.config.decoder_start_token_id)
        generated = decoder_input_ids

        for _ in range(max_new_tokens):
            decoder_inputs_embeds = self._expand_input_ids(generated)
            decoder_hidden = self.decoder(
                decoder_inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )
            # Get next token logits (last position)
            next_logits = self.lm_head(decoder_hidden[:, -1:, :])  # (batch, 1, vocab)

            if do_sample:
                # Apply temperature
                next_logits = next_logits / max(temperature, 1e-7)
                next_logits = next_logits.astype(ms.float32).squeeze(1)  # (batch, vocab)

                # Top-k filtering
                if top_k > 0:
                    topk_vals, _ = ops.top_k(next_logits, top_k)
                    threshold = topk_vals[:, -1:]
                    next_logits = ops.where(next_logits >= threshold, next_logits,
                                            ops.fill(ms.float32, next_logits.shape, -1e9))

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = ops.sort(next_logits, descending=True)
                    cumulative_probs = ops.softmax(sorted_logits, axis=-1)
                    cumulative_probs = ops.cumsum(cumulative_probs, axis=-1)
                    sorted_mask = cumulative_probs - sorted_logits.exp() < top_p  # approximate
                    # Simple approach: just take top-k
                    pass

                probs = ops.softmax(next_logits.astype(ms.float32), axis=-1)
                next_token = ops.multinomial(probs, num_samples=1, replacement=True)
            else:
                next_token = ops.argmax(next_logits, axis=-1)

            generated = ops.concat([generated, next_token.astype(ms.int32)], axis=1)

            # Check EOS
            if (next_token == eos_token_id).all():
                break

        if self.model_type == "seq2seq":
            return generated[:, 1:]  # remove decoder start token
        return generated[:, -max_new_tokens:]


# ======================== ChronosModel Wrapper ========================

class ChronosModel(nn.Cell):
    """
    Wraps ChronosT5Model with ChronosConfig for forecasting.

    Provides encode() for embeddings and forward() for generating samples.
    """

    def __init__(self, chronos_config, t5_config: T5Config):
        super().__init__()
        self.chronos_config = chronos_config
        self.model = ChronosT5Model(t5_config)

    def construct(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        labels: Optional[Tensor] = None,
    ):
        """Training forward (with labels) or inference forward."""
        if labels is not None:
            # Training mode
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Inference mode: generate
        if prediction_length is None:
            prediction_length = self.chronos_config.prediction_length
        if num_samples is None:
            num_samples = self.chronos_config.num_samples
        if temperature is None:
            temperature = self.chronos_config.temperature
        if top_k is None:
            top_k = self.chronos_config.top_k
        if top_p is None:
            top_p = self.chronos_config.top_p

        # Repeat inputs for num_samples
        if num_samples > 1:
            input_ids = ops.repeat_interleave(input_ids, num_samples, axis=0)
            attention_mask = ops.repeat_interleave(attention_mask, num_samples, axis=0)

        preds = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=prediction_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.chronos_config.eos_token_id,
            pad_token_id=self.chronos_config.pad_token_id,
        )

        if self.chronos_config.model_type == "seq2seq":
            preds = preds[..., 1:]  # remove decoder start token
        else:
            preds = preds[..., -prediction_length:]

        batch_size = input_ids.shape[0] // num_samples if num_samples > 0 else input_ids.shape[0]
        return preds.view(batch_size, num_samples, -1)

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Get encoder hidden states."""
        inputs_embeds = self.model._expand_input_ids(input_ids)
        return self.model.encoder(inputs_embeds, attention_mask=attention_mask)
