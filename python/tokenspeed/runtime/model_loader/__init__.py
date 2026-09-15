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

from torch import nn

from tokenspeed.runtime.configs.device_config import DeviceConfig
from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.model_loader.loader import BaseModelLoader, get_model_loader
from tokenspeed.runtime.model_loader.utils import get_model_architecture


def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    """Load and return a model for the given runtime configuration."""
    loader = get_model_loader(load_config)
    model = loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    if global_server_args_dict.get("enable_dsv4_vision_instrumentation", False):
        from tokenspeed.runtime.metrics.dsv4_vision_instrumentation import (
            get_dsv4_vision_instrumentation,
        )

        instrumentation = get_dsv4_vision_instrumentation()
        instrumentation.record_model_parameters(
            name for name, _ in model.named_parameters()
        )
    return model


__all__ = [
    "get_model",
    "get_model_loader",
    "BaseModelLoader",
    "get_model_architecture",
]
