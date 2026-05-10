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

from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig


def _is_quark_w_mxfp4_a_fp8(config: dict) -> bool:
    """Detect AMD-Quark exported W=MXFP4 / A=FP8 (per-tensor static) checkpoints.

    The HF ``quantization_config`` for these checkpoints looks like::

        {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "fp4", "group_size": 32,
                           "scale_format": "e8m0", "qscheme": "per_group"},
                "input_tensors": {"dtype": "fp8_e4m3", "qscheme": "per_tensor",
                                  "is_dynamic": false},
                ...
            },
            ...
        }

    See https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8 for a reference.
    """
    if not isinstance(config, dict):
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
        excluded_layers: list[str] | None = None,
    ):
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.ignored_layers = ignored_layers
        # When True, the checkpoint stores per-expert MXFP4 weights with a
        # per-tensor static FP8 activation scale (Quark "w_mxfp4_a_fp8").
        # The MoE backend uses FP8 x MXFP4 GEMMs in this case.
        self.is_w4a8_fp8 = is_w4a8_fp8
        # Names listed under Quark's ``exclude`` field; these layers are kept
        # in the original (bf16/fp16) precision and bypass the MoE backend.
        self.excluded_layers = excluded_layers or []

    @classmethod
    def from_config(cls, config):
        quant_method = str(config.get("quant_method", "")).lower()
        is_w4a8_fp8 = _is_quark_w_mxfp4_a_fp8(config)
        is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method or is_w4a8_fp8
        excluded = list(config.get("exclude", []) or [])
        return cls(
            is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized,
            is_w4a8_fp8=is_w4a8_fp8,
            excluded_layers=excluded,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        """Promote AMD Quark ``w_mxfp4_a_fp8`` checkpoints to mxfp4."""
        if user_quant in {"mxfp4", None} and _is_quark_w_mxfp4_a_fp8(hf_quant_cfg):
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
