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


"""End-to-end entry points for gfx950 A4W4 MoE."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.decode_stage1 import (
    invoke_stage1_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.decode_stage2 import (
    invoke_stage2_mxfp4_mfma_decode_gluon,
)


def gluon_mxfp4_moe_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    block_n1: int | None = None,
    block_d2: int = 16,
    *,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
) -> torch.Tensor:
    """Run the small-M direct MXFP4-activation x MXFP4-weight MoE.

    BF16 is only the entry/intermediate/output storage contract.  Both GEMM
    inputs are dynamically quantized to packed E2M1 plus E8M0 block scales
    before the direct CDNA4 MFMA kernels are launched.
    """
    from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
        _extract_gluon_raw_s,
        _extract_gluon_raw_w,
        _quantize_mxfp4_activation,
    )

    assert hidden_states.dtype == torch.bfloat16
    w1_runtime = _extract_gluon_raw_w(w1)
    w2_runtime = _extract_gluon_raw_w(w2)
    w1_scale_runtime = _extract_gluon_raw_s(w1_scale)
    w2_scale_runtime = _extract_gluon_raw_s(w2_scale)
    assert all(
        isinstance(t, torch.Tensor)
        for t in (w1_runtime, w2_runtime, w1_scale_runtime, w2_scale_runtime)
    )

    num_tokens, hidden_dim = hidden_states.shape
    topk = int(topk_ids.shape[1])
    inter_dim = int(w1_runtime.shape[2]) // 2
    out_dim = int(getattr(w2_runtime, "original_n", int(w2_runtime.shape[2])))
    assert out_dim == hidden_dim
    if block_n1 is None:
        block_n1 = 16 if num_tokens <= 2 else 32

    hidden_mxfp4, hidden_scale = _quantize_mxfp4_activation(hidden_states)
    inter = torch.empty(
        (num_tokens * topk, inter_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    invoke_stage1_mxfp4_mfma_decode_gluon(
        hidden_mxfp4,
        hidden_scale,
        w1_runtime,
        w1_scale_runtime,
        topk_ids,
        inter,
        topk,
        BLOCK_N=block_n1,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        swiglu_beta=swiglu_beta,
    )

    inter_mxfp4, inter_scale = _quantize_mxfp4_activation(inter)
    out = torch.empty(
        (num_tokens, out_dim), dtype=torch.bfloat16, device=hidden_states.device
    )
    invoke_stage2_mxfp4_mfma_decode_gluon(
        inter_mxfp4,
        inter_scale,
        w2_runtime,
        w2_scale_runtime,
        topk_ids,
        topk_weights,
        out,
        topk,
        BLOCK_N=block_d2,
    )
    return out


__all__ = ["gluon_mxfp4_moe_decode"]
