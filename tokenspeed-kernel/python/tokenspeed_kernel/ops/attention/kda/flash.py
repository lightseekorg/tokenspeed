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

"""FlashKDA drop-in for the chunked KDA prefill scan.

Mirrors ``triton.linear.kda.kda_chunk_prefill``'s signature and state
convention (FLA-native ``[N, HV, K, V]`` states) so the runtime can swap the
two calls behind a policy flag. FlashKDA (MoonshotAI, two CUTLASS kernels)
applies sigmoid(beta), the safe gate, and QK L2 normalization in-kernel like
the FLA path, but has an order of magnitude less fixed overhead at short
extend lengths and no shape specialization. Its state ABI is ``[N, HV, V, K]``,
so this wrapper transposes the square 128x128 state on entry and returns the
final state transposed back to the FLA convention (a per-layer ~1 MB copy,
microseconds against a multi-ms forward).
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.thirdparty.flash_kda import (
    flash_kda_fwd,
    is_flash_kda_installed,
)

__all__ = ["flash_kda_chunk_prefill", "is_flash_kda_installed"]


def flash_kda_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    lower_bound: float | None = None,
    beta_is_logit: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked prefill KDA scan through FlashKDA (varlen via ``cu_seqlens``).

    Args:
        q: Query ``[B, T, H, K]`` (bfloat16; raw or pre-normalized — the
            kernel L2-normalizes, which is idempotent).
        k: Key, same shape/dtype rules as ``q``.
        v: Value ``[B, T, HV, V]`` bfloat16.
        g_raw: Raw per-channel decay logits ``[B, T, HV, K]``.
        beta: Raw beta logits ``[B, T, HV]``; sigmoid is applied in-kernel.
        A_log: Per-head FP32 decay parameter ``[HV]``.
        dt_bias: FP32 gate bias with ``HV * K`` elements.
        initial_state: Optional FP32 recurrent state per packed sequence in
            the FLA-native ``[N, HV, K, V]`` convention; ``None`` starts from
            zero.
        cu_seqlens: Cumulative sequence boundaries ``[N + 1]`` (``B`` must
            be 1); ``None`` treats each batch row as one sequence.
        cu_seqlens_cpu: Host copy of ``cu_seqlens`` (unified prefill-op
            signature). FlashKDA plans entirely on device and does not read
            it.
        lower_bound: Safe-gate lower bound; required (FlashKDA applies the
            safe gate unconditionally).
        beta_is_logit: Must be True; FlashKDA always applies sigmoid.

    Returns:
        ``(o [B, T, HV, V], final_state [N, HV, K, V])`` matching the FLA
        wrapper's convention.
    """
    if not beta_is_logit:
        raise ValueError("flash_kda_chunk_prefill requires raw beta logits")
    if lower_bound is None:
        raise ValueError("flash_kda_chunk_prefill requires a safe-gate bound")
    if dt_bias is None:
        raise ValueError("flash_kda_chunk_prefill requires dt_bias")
    key_dim = q.shape[-1]
    num_value_heads = v.shape[2]
    if cu_seqlens is not None:
        num_sequences = cu_seqlens.numel() - 1
        boundaries = cu_seqlens.to(dtype=torch.int64)
    else:
        num_sequences = q.shape[0]
        boundaries = None
    # FLA-native [N, HV, K, V] -> FlashKDA [N, HV, V, K].
    state_in = (
        initial_state.transpose(-1, -2).contiguous()
        if initial_state is not None
        else None
    )
    final_state = torch.empty(
        num_sequences,
        num_value_heads,
        v.shape[-1],
        key_dim,
        dtype=torch.float32,
        device=q.device,
    )
    out = torch.empty_like(v)
    flash_kda_fwd()(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g_raw.contiguous(),
        beta.contiguous(),
        1.0 / math.sqrt(key_dim),
        out,
        A_log.contiguous(),
        dt_bias.reshape(num_value_heads, key_dim).contiguous(),
        float(lower_bound),
        initial_state=state_in,
        final_state=final_state,
        cu_seqlens=boundaries,
    )
    return out, final_state.transpose(-1, -2)
