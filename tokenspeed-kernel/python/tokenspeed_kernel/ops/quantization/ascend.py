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

"""Ascend (NPU) FP8 quantization under the ``npu`` solution.

CANN 9.0 + torch_npu 2.10 cannot materialize FP8 (E4M3/E5M2) tensor values
(casts and in-place copies into FP8 output fail with ``aclnnInplaceCopy``;
``torch_npu.npu_rms_norm_dynamic_quant`` is not present in this build), so the
FP8 *cast* semantics are simulated numerically: values are clamped to the E4M3
finite range (and divided by the optional scale) and returned in the input
dtype. The output is NOT an FP8 tensor — callers on NPU must keep FP8 data in
the dequantized BF16/FP16 representation that the ``npu`` GEMM solutions
consume. The numeric result matches a clamped scaled cast; grid rounding is
not applied (documented deviation).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = ["npu_quantize_fp8"]

_NPU_CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))

_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


@register_kernel(
    "quantization",
    "fp8",
    name="npu_quantize_fp8",
    solution="npu",
    capability=_NPU_CAPABILITY,
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={"has_scale": frozenset({True, False})},
    priority=Priority.PORTABLE + 1,
    tags={"portability"},
)
def npu_quantize_fp8(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Simulate an FP8 cast on Ascend NPU (dequantized BF16/FP16 output).

    When ``scale`` is provided the input is divided by it first; the result is
    clamped to the E4M3 finite range and returned in ``x``'s dtype.
    """
    del enable_pdl  # NPU path has no PDL launch.
    if scale is not None:
        scale_value = (
            float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)
        )
        if scale_value == 0:
            raise ValueError("npu_quantize_fp8 scale must be non-zero")
        output = (x.float() / scale_value).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    else:
        output = x.float().clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return output.to(x.dtype)
