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

"""TRT-LLM GEMM kernels exposed via the numerics registry.

- ``cublaslt_mm_nvfp4`` wraps the cuBLASLt NVFP4 GEMM runner (heuristic algo 0).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.signature import ScaleFormat, format_signature, tensor_format
from tokenspeed_kernel_nvidia.registration import Priority, register_kernel

# Re-exported
# dsv3_fused_a_gemm supports specific shapes only (see python/tokenspeed/runtime/models/deepseek_v3.py);
# call such kernels manually rather than by register_kernel.
from tokenspeed_kernel_nvidia.thirdparty.trtllm import dsv3_fused_a_gemm  # noqa: F401

_fp4_dtypes: frozenset[torch.dtype] = frozenset({torch.uint8, torch.float4_e2m1fn_x2})
_NVFP4_SCALE_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float32, torch.uint8, torch.float8_e4m3fn}
)
_NVFP4_FORMAT_SIGNATURES = frozenset(
    format_signature(
        a=tensor_format(
            "nvfp4",
            storage_dtype,
            scale=ScaleFormat(
                storage_dtype=a_scale_dtype, granularity="block", block_shape=(16,)
            ),
        ),
        b=tensor_format(
            "nvfp4",
            storage_dtype,
            scale=ScaleFormat(
                storage_dtype=b_scale_dtype, granularity="block", block_shape=(16,)
            ),
        ),
    )
    for storage_dtype in _fp4_dtypes
    for a_scale_dtype in _NVFP4_SCALE_DTYPES
    for b_scale_dtype in _NVFP4_SCALE_DTYPES
)

# One stateful torchbind instance per output dtype. Each holds its own
# per-shape algo cache inside C++.
_runner_cache: dict[torch.dtype, object] = {}
_CUBLASLT_HEURISTIC_ALGO = 0


def _get_runner(out_dtype: torch.dtype):
    runner = _runner_cache.get(out_dtype)
    if runner is None:
        runner = torch.classes.trtllm.CublasLtFP4GemmRunner(out_dtype)
        _runner_cache[out_dtype] = runner
    return runner


@register_kernel(
    "gemm",
    "mm",
    name="cublaslt_mm_nvfp4",
    solution="cublas",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=_NVFP4_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED + 3,
)
def cublaslt_mm_nvfp4(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor,
    block_size: list[int] | None = None,
) -> torch.Tensor:
    runner = _get_runner(out_dtype)
    return runner.run_gemm(
        A,
        B.T,
        A_scales,
        B_scales.T,
        alpha,
        False,
        _CUBLASLT_HEURISTIC_ALGO,
    )
