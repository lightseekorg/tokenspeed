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

from typing import Any

import torch

from tokenspeed.runtime.layers.quantization import QuantizationConfig


class W8A8Fp8Config(QuantizationConfig):
    """Config class for W8A8 FP8 Quantization.

    Weight Quantization:
    - Method: Static quantization
    - Granularity: Per-channel
    - Type: Symmetric

    Activation Quantization:
    - Method: Dynamic quantization
    - Granularity: Per-token
    - Type: Symmetric

    Note:
    - For models without offline quantization, weights will be quantized during model loading:
        - If CUTLASS is supported: Per-channel weight quantization is used
        - If CUTLASS is not supported: Falls back to per-tensor weight quantization
    """

    def __init__(self, is_checkpoint_fp8_serialized: bool = False):
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_name(self) -> str:
        return "w8a8_fp8"

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = (
            "compressed-tensors" in quant_method or "w8a8_fp8" in quant_method
        )
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized)

    def get_scaled_act_names(self) -> list[str]:
        return []
