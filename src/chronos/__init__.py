# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .__about__ import __version__
from .base import BaseChronosPipeline, ForecastType
from .chronos import (
    ChronosConfig,
    ChronosModel,
    ChronosPipeline,
    ChronosTokenizer,
    MeanScaleUniformBins,
)

# Chronos-2 and Chronos-Bolt are optional (need newer transformers/peft)
try:
    from .chronos2 import Chronos2ForecastingConfig, Chronos2Model, Chronos2Pipeline
except ImportError:
    Chronos2ForecastingConfig = None  # type: ignore
    Chronos2Model = None  # type: ignore
    Chronos2Pipeline = None  # type: ignore

try:
    from .chronos_bolt import ChronosBoltConfig, ChronosBoltPipeline
except ImportError:
    ChronosBoltConfig = None  # type: ignore
    ChronosBoltPipeline = None  # type: ignore

__all__ = [
    "__version__",
    "BaseChronosPipeline",
    "ForecastType",
    "ChronosConfig",
    "ChronosModel",
    "ChronosPipeline",
    "ChronosTokenizer",
    "MeanScaleUniformBins",
    "ChronosBoltConfig",
    "ChronosBoltPipeline",
    "Chronos2ForecastingConfig",
    "Chronos2Model",
    "Chronos2Pipeline",
]
