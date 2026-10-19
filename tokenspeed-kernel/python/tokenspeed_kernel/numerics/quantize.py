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

from typing import Any

import torch
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance

# Quantize kernels under test return ``qweight.float()`` — the fp8 values cast
# to fp32. Two correct implementations following the same round-to-nearest-even
# rule on the same group statistics produce *bit-identical* fp8 values, so the
# tolerance only needs to absorb a single fp8 ulp (~ 1/8 at magnitude 1) for
# values near the quantization boundary, where different scale-rounding paths
# could land one ulp apart.


def tolerance(
    dtype: torch.dtype,
    *,
    inputs: dict[str, Any] | None = None,
    **_: Any,
) -> Tolerance:
    # 1 fp8 e4m3 ulp at the values we actually compare. fp8_e4m3 has 3 mantissa
    # bits, so the relative gap between adjacent representable values is 2^-3
    # = 0.125. Allow that plus a small safety margin.
    return Tolerance(atol=0.2, rtol=0.2)


set_family_tolerance("quantize", tolerance)


class QuantizeInputGenerator(InputGenerator):
    """Generates a 2D activation tensor [M, K] for fp8 quantize kernels."""

    def generate(self, M: int, K: int) -> dict[str, Any]:
        x = torch.randn(
            M, K, dtype=torch.float32, device=self.device, generator=self.rng
        ).to(self.dtype)
        return {"x": x}


set_input_generator("quantize", "fp8_token_group_128", QuantizeInputGenerator)
set_input_generator("quantize", "fp8_token", QuantizeInputGenerator)
set_input_generator("quantize", "fp8_tensor", QuantizeInputGenerator)

# K must be divisible by the group size (128). Cover (a) prefill batches, (b)
# decode (M = 1), and (c) DSv3 hidden / fused-A widths.

_QUANTIZE_STANDARD_SHAPES: list[dict[str, int]] = [
    {"M": 1, "K": 128},
    {"M": 1, "K": 7168},
    {"M": 8, "K": 128},
    {"M": 8, "K": 7168},
    {"M": 32, "K": 1536},
    {"M": 128, "K": 4096},
    {"M": 128, "K": 7168},
    {"M": 512, "K": 4096},
]

for _mode in ("fp8_token_group_128", "fp8_token", "fp8_tensor"):
    set_standard_shapes("quantize", _mode, _QUANTIZE_STANDARD_SHAPES)
    set_benchmark_shapes("quantize", _mode, _QUANTIZE_STANDARD_SHAPES)
