# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""FlashInfer Blackwell (sm100) gated delta net chunked prefill.

Fast-path for GDN prefill on Blackwell (sm100/sm103) + CUDA 13 + bf16 +
head_dim==128. The caller gates on ``is_supported`` and must fail fast when the
fast-path is unavailable;
this module has no Triton fallback.

Convention vs the Triton FLA path (verified equal to bf16 on B200):
- q, k must be L2-normalized by the caller (the sm100 kernel ignores its own
  use_qk_l2norm flag).
- g is the FLA log-space forget gate; the sm100 kernel takes log internally, so
  we pass alpha = exp(g).
- beta is cast to float32 (flashinfer passes it through without casting).
- the recurrent state is stored transposed (K<->V) vs FLA, so we transpose the
  initial state in and the final state out.
- scale defaults to head_dim**-0.5 (FLA default).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform

SUPPORTED_HEAD_DIM = 128

_chunk_gated_delta_rule = None
_AVAILABLE = False

if current_platform().is_nvidia:
    try:
        from flashinfer.gdn_kernels import (
            _has_blackwell_prefill as _fi_has_blackwell_prefill,
        )
        from flashinfer.gdn_prefill import chunk_gated_delta_rule as _fi_chunk

        _chunk_gated_delta_rule = _fi_chunk
        _p = current_platform()
        _cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        # flashinfer's gdn_prefill treats any compute-capability major 10 as the
        # Blackwell path (sm100 B200/GB200, sm103 B300), gated on CUDA>=13 and the
        # prefill kernel being present; it raises NotImplementedError otherwise.
        # Mirror that here so the caller does not commit to a crashing fast-path.
        _AVAILABLE = (
            _p.arch_version.major == 10
            and _cuda_major >= 13
            and _fi_has_blackwell_prefill
        )
    except ImportError:
        _AVAILABLE = False


def is_available() -> bool:
    """Whether the sm100 GDN kernel can run on this platform."""
    return _AVAILABLE


def is_supported(
    head_dim: int, dtype: torch.dtype, num_q_heads: int, num_v_heads: int
) -> bool:
    # bf16 is the verified path; fp16 is rejected (caller fails fast).
    # flashinfer reads g/beta/state with max(num_q, num_v) heads; the runtime
    # supplies them with num_v heads, so num_v < num_q (e.g. Hk=32, Hv=16) reads
    # out of bounds. Only num_v >= num_q is safe (GVA or equal heads).
    return (
        _AVAILABLE
        and head_dim == SUPPORTED_HEAD_DIM
        and dtype == torch.bfloat16
        and num_v_heads >= num_q_heads
    )


def gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one chunked GDN prefill on the sm100 kernel.

    q, k, v are [B, T, H, D] (B==1) or [T, H, D]; q/k already L2-normalized.
    g, beta are [B, T, H] or [T, H]; g in log space. initial_state is the FLA
    [N, H, K, V] recurrent state. Returns (out, final_state) matching the FLA
    layout: out [B, T, Hv, D] in q.dtype, final_state [N, H, K, V].
    """
    batched = q.dim() == 4
    q3 = q.squeeze(0) if batched else q
    k3 = k.squeeze(0) if batched else k
    v3 = v.squeeze(0) if batched else v
    g2 = g.squeeze(0) if g.dim() == 3 else g
    beta2 = beta.squeeze(0) if beta.dim() == 3 else beta

    head_dim = q3.shape[-1]
    if scale is None:
        scale = head_dim**-0.5

    out, final_state = _chunk_gated_delta_rule(
        q3.contiguous(),
        k3.contiguous(),
        v3.contiguous(),
        g=torch.exp(g2).float().contiguous(),
        beta=beta2.float().contiguous(),
        scale=scale,
        # flashinfer requires fp32 state; runtime ssm dtype may be bf16.
        initial_state=initial_state.float().transpose(-1, -2).contiguous(),
        output_final_state=True,
        # flashinfer casts cu_seqlens per path internally (int32 for sm100), so
        # pass it through rather than forcing a dtype.
        cu_seqlens=cu_seqlens,
    )

    out = out.to(q.dtype)
    if batched:
        out = out.unsqueeze(0)
    return out, final_state.transpose(-1, -2)
