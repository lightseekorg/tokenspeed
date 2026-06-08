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

"""DeepSeek-V3 router GEMM kernel wrapper.

Provides dsv3_router_gemm with cuBLAS fallback for num_tokens > 16.
"""

import functools
from pathlib import Path

import torch


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_dsv3_gemm_module():
    """Load the pre-compiled dsv3_gemm shared library via TVM FFI."""
    import tvm_ffi

    so_path = _objs_dir() / "dsv3_gemm" / "dsv3_gemm.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel dsv3_gemm library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Router GEMM for DeepSeek-V3 MoE gating.

    Args:
        hidden_states: bf16 CUDA tensor [num_tokens, hidden_dim]
        router_weights: bf16 or fp32 CUDA tensor [num_experts, hidden_dim]
        out_dtype: output dtype (must be torch.float32)

    Returns:
        float32 CUDA tensor [num_tokens, num_experts]
    """
    if out_dtype is not torch.float32:
        raise ValueError("dsv3_router_gemm only supports out_dtype=torch.float32")
    output = torch.empty(
        (hidden_states.shape[0], router_weights.shape[0]),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    _load_dsv3_gemm_module().dsv3_router_gemm(
        output, hidden_states, router_weights, bool(enable_pdl)
    )
    return output
