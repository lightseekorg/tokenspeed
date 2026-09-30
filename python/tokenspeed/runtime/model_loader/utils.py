# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Utilities for selecting and loading models."""

import contextlib
from collections.abc import Generator

import torch
from torch import nn

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.hf_transformers_utils import model_loader_architectures

logger = get_colorful_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype) -> Generator[None]:
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    from tokenspeed.runtime.models.registry import ModelRegistry

    architectures = model_loader_architectures(model_config.hf_config)
    # Mixtral only supports the quantization backends listed here in the
    # current model registry and loader stack.
    mixtral_supported = ["fp8", "compressed-tensors"]

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    model_cls, architecture = ModelRegistry.resolve_model_cls(architectures)
    profiled = model_config.model_profile_architecture
    if profiled is not None and architecture != profiled:
        # An earlier architecture in the list resolved to another class, so
        # the plugin profile that configured attention and cache geometry
        # describes a model that is not the one being built.
        raise ValueError(
            f"the model loader resolves architecture {architecture!r}, but the "
            f"plugin profile was taken from {profiled!r}; list the plugin "
            "architecture first in config.architectures"
        )
    return model_cls, architecture
