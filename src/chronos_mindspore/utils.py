# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MindSpore adaptation of Chronos utility functions

from typing import List

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor


def left_pad_and_stack_1D(tensors: List[Tensor]) -> Tensor:
    """Left-pad a list of 1D tensors with NaN and stack into 2D."""
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert c.ndim == 1
        padding = ops.fill(ms.float32, (max_len - len(c),), float('nan'))
        padded.append(ops.concat((padding, c), axis=-1))
    return ops.stack(padded)
