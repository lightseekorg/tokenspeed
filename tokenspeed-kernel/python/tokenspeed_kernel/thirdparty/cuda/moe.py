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

"""MoE kernels: fused finalize + shared-output residual."""

import functools
from pathlib import Path
from typing import Optional

import torch


@functools.cache
def _load_moe_finalize_fuse_shared_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "moe_finalize_fuse_shared"
    so_path = objs_dir / "moe_finalize_fuse_shared.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel moe_finalize_fuse_shared library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def moe_finalize_fuse_shared(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    top_k: int,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Fused MoE finalize + optional shared-output residual (bf16, SM>=90).

    Computes, per token ``t``::

        out[t] = Σ_k expert_weights[t, k] * gemm2_out[permuted_idx(t, k)]
               + shared_output[t]                      # if non-null

    Replaces the flashinfer built-in finalize kernel + the native
    ``routed + shared`` tensor add. The caller is responsible for ensuring
    ``shared_output`` is ready on the current stream (e.g. via
    ``current_stream.wait_stream(alt_stream)``).

    Expert-weight scale convention: ``expert_weights`` are read verbatim.
    In the DSv3/K2.5 path they already carry ``routed_scaling_factor``
    because ``apply_routed_scaling_factor_on_output=True`` folds it in at
    topk time — so this kernel does not apply any additional scale.

    Args:
        gemm2_out: ``[total_num_padded_tokens, hidden_dim_padded]`` bf16 —
            raw permuted MoE output when the flashinfer runner was called
            with ``do_finalize=False``.
        expanded_idx_to_permuted_idx: ``[num_tokens * top_k]`` int32 —
            permute map (``-1`` means "drop this slot").
        expert_weights: ``[num_tokens, top_k]`` float32 or bfloat16 — per-token
            topk weights, already scaled. DSv3/K2.5 trtllm backends use
            float32 (``_routing_logits_dtype = torch.float32``); other
            backends use bf16. The kernel is templated on this dtype.
        shared_output: ``[num_tokens, hidden_dim]`` bf16 or ``None`` —
            per-token residual to fold into the finalize.
        top_k: top-k count (must be ``<= 64``).
        enable_pdl: honor upstream/downstream PDL if True.

    Returns:
        ``[num_tokens, hidden_dim]`` bf16.
    """
    assert gemm2_out.dtype == torch.bfloat16
    assert expert_weights.dtype in (torch.float32, torch.bfloat16)
    assert expanded_idx_to_permuted_idx.dtype == torch.int32
    assert gemm2_out.dim() == 2
    assert expert_weights.dim() == 2
    num_tokens, top_k_check = expert_weights.shape
    assert top_k_check == top_k
    hidden_dim = gemm2_out.shape[1]
    # hiddenDim = out.shape[-1]; caller may want a trimmed hidden_dim if
    # padding was applied on the permuted side.
    if shared_output is not None:
        assert shared_output.dtype == torch.bfloat16
        assert shared_output.dim() == 2
        assert shared_output.shape[0] == num_tokens
        hidden_dim = shared_output.shape[1]
        assert hidden_dim <= gemm2_out.shape[1]

    out = torch.empty(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device=gemm2_out.device
    )
    # The C++ side uses numel()==0 to mean "no shared bias"; pass an empty
    # placeholder when the caller didn't provide one. Avoids optional-tensor
    # plumbing through tvm_ffi.
    if shared_output is None:
        shared_output = gemm2_out.new_empty((0, 0), dtype=torch.bfloat16)

    mod = _load_moe_finalize_fuse_shared_module()
    mod.moe_finalize_fuse_shared(
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        int(top_k),
        bool(enable_pdl),
    )
    return out
