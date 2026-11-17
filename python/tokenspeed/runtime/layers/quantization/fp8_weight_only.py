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

"""Weight-only FP8 for a 16-bit decode model's bandwidth-bound projections."""

from __future__ import annotations

import re
from typing import Any

import torch

from tokenspeed.runtime.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)

# The dense MLP projections and the attention output projection: at decode
# rows the MLP GEMVs stream most of a layer's weight bytes, and the output
# projection's GEMV is short enough that half the bytes still show. The QKV
# and other projections keep 16-bit weights: their K = hidden GEMVs measured
# no faster with FP8 weights at 8-16 rows (latency bound), and QKV also runs
# prefill rows when a draft writes its context.
_QUANTIZED_LAYER = re.compile(
    r"(^|\.)(mlp\.(gate_up_proj|down_proj)|self_attn\.o_proj)$"
)


class Fp8WeightOnlyConfig(QuantizationConfig):
    """Weight-only FP8 E4M3 for the MLP and attention output projections.

    Each such weight of a 16-bit checkpoint becomes E4M3 at load, with one
    FP32 scale per output row (the row's absolute maximum over 448).
    Activations stay 16-bit: the GEMM upcasts the weight exactly, sums in
    FP32 and applies the row scale once
    (``tokenspeed_kernel.ops.gemm.mv_fp8_weight``). Meant for a decode-only
    model, a speculative draft, through
    ``--speculative-draft-model-quantization fp8_weight_only``: that GEMV
    serves up to 32 rows a launch, and larger calls run as 32-row chunks
    that each re-read the weight.
    """

    @classmethod
    def get_name(cls) -> str:
        return "fp8_weight_only"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Fp8WeightOnlyConfig:
        return cls()

    def quantizes(self, prefix: str) -> bool:
        """Whether the linear at ``prefix`` takes FP8 weights."""
        return _QUANTIZED_LAYER.search(prefix) is not None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase:
        from tokenspeed.runtime.layers.dense.fp8_weight_only import (
            Fp8WeightOnlyLinearMethod,
        )
        from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod

        if self.quantizes(prefix):
            return Fp8WeightOnlyLinearMethod()
        return UnquantizedLinearMethod()

    def get_scaled_act_names(self) -> list[str]:
        return []
