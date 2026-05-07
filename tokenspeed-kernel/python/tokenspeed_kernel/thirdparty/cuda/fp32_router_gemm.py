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

"""FP32 router GEMM kernel wrapper for MiniMax M2.

Fused kernel for activation(bf16/fp32) x weight(fp32) -> fp32.
Custom kernel for M<=32 (E=256, H=3072), cuBLAS fallback for larger M.

Ported from the TensorRT LLM router GEMM implementation.
"""

import functools
from pathlib import Path

import torch

_FP32_ROUTER_GEMM_FAST_PATH_MAX_TOKENS = 32
_FP32_ROUTER_GEMM_HIDDEN_DIM = 3072
_FP32_ROUTER_GEMM_NUM_EXPERTS = 256


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_fp32_router_gemm_module():
    """Load the pre-compiled fp32_router_gemm shared library via TVM FFI."""
    import tvm_ffi

    so_path = _objs_dir() / "fp32_router_gemm" / "fp32_router_gemm.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel fp32_router_gemm library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def supports_fp32_router_gemm_fast_path(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
) -> bool:
    """Return whether the custom small-M router GEMM path can be used."""
    return (
        hidden_states.is_cuda
        and router_weights.is_cuda
        and hidden_states.ndim == 2
        and router_weights.ndim == 2
        and hidden_states.dtype in (torch.bfloat16, torch.float32)
        and router_weights.dtype == torch.float32
        and 0 < hidden_states.shape[0] <= _FP32_ROUTER_GEMM_FAST_PATH_MAX_TOKENS
        and hidden_states.shape[1] == _FP32_ROUTER_GEMM_HIDDEN_DIM
        and router_weights.shape
        == (
            _FP32_ROUTER_GEMM_NUM_EXPERTS,
            _FP32_ROUTER_GEMM_HIDDEN_DIM,
        )
    )


def fp32_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    """Router GEMM for MiniMax M2 MoE gating.

    Args:
        hidden_states: bf16 or fp32 CUDA tensor [num_tokens, hidden_dim]
        router_weights: fp32 CUDA tensor [num_experts, hidden_dim]

    Returns:
        float32 CUDA tensor [num_tokens, num_experts]
    """
    output = torch.empty(
        (hidden_states.shape[0], router_weights.shape[0]),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    _load_fp32_router_gemm_module().fp32_router_gemm(
        output, hidden_states, router_weights
    )
    return output
