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


from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig


def _is_amd_quark_w_mxfp4_a_fp8(config: dict) -> bool:
    if not isinstance(config, dict):
        return False
    if not current_platform().is_amd:
        return False
    if str(config.get("quant_method", "")).lower() != "quark":
        return False
    g = config.get("global_quant_config") or {}
    weight = g.get("weight") or {}
    inputs = g.get("input_tensors") or {}
    if str(weight.get("dtype", "")).lower() not in {"fp4", "mxfp4"}:
        return False
    if int(weight.get("group_size", 0)) != 32:
        return False
    in_dtype = str(inputs.get("dtype", "")).lower()
    if "fp8" not in in_dtype:
        return False
    return True


class Mxfp4Config(QuantizationConfig):

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        is_checkpoint_mxfp4_serialized: bool = False,
        is_w4a8_fp8: bool = False,
    ):
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.ignored_layers = ignored_layers
        self.is_w4a8_fp8 = is_w4a8_fp8

    @classmethod
    def from_config(cls, config):
        quant_method = str(config.get("quant_method", "")).lower()
        is_w4a8_fp8 = _is_amd_quark_w_mxfp4_a_fp8(config)
        is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method or is_w4a8_fp8
        return cls(
            is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized,
            is_w4a8_fp8=is_w4a8_fp8,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        """Promote AMD Quark w_mxfp4_a_fp8 checkpoints to mxfp4."""
        if user_quant in {"mxfp4", None} and _is_amd_quark_w_mxfp4_a_fp8(hf_quant_cfg):
            return "mxfp4"
        return None

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def is_static_cfg(self):
        return self.is_checkpoint_mxfp4_serialized

    def get_scaled_act_names(self) -> list[str]:
        return []
