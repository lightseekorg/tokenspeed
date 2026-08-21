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

"""Kimi Delta Attention (KDA) gated-delta-rule op backed by ``fla``.

KDA's applied decay gate is **per-channel**: a per-head log-decay
``A_log[num_heads]`` modulates a per-(head, channel) gate ``g [B, T, HV, K]`` --
unlike GDN's scalar-per-head decay. ``fla``'s ``chunk_kda`` /
``fused_recurrent_kda`` implement the gated-delta scan. The optional dependency
is isolated in
``tokenspeed_kernel.thirdparty.fla``.

The gate is computed inside ``fla``'s kernel (``use_gate_in_kernel=True``): we
pass the raw ``g`` plus ``A_log`` and per-(head, channel) ``dt_bias``, and ``fla``
applies the safe gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``. The
checkpoint stores ``A_log`` in a ``[head_dim]``-sized buffer zero-padded past
``num_heads``; the model loads only the real ``[num_heads]`` per-head entries.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.thirdparty import fla as _fla


def kda_chunk_prefill(
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
    lower_bound: float | None = None,
    recurrent_layout: str = "k_major",
    beta_is_logit: bool = True,
):
    """Chunked prefill KDA scan (varlen via ``cu_seqlens``).

    q/k: ``[B, T, H, K]``; v: ``[B, T, HV, V]``; g_raw: ``[B, T, HV, K]``;
    beta: ``[B, T, HV]`` (raw logits if ``beta_is_logit``); returns
    ``(o [B, T, HV, V], final_state [N, HV, K, V])``. QK are L2-normalized and the
    safe gate is applied inside the kernel (``use_gate_in_kernel``).
    """
    # ``use_beta_sigmoid_in_kernel`` is a backend-extension kwarg that FLA's
    # native chunk_kda silently swallows via **kwargs — passing raw logits
    # with that flag makes the native path consume the LOGIT as the delta
    # coefficient (beta can then be negative or exceed 1). Apply the sigmoid
    # here instead; against a token-by-token recurrent-decode trajectory this
    # takes the chunk path's max state error from ~4e-1 down to ~8e-4.
    if beta_is_logit:
        beta = beta.float().sigmoid()
    return _fla.chunk_kda(
        q,
        k,
        v,
        g_raw,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )


def kda_recurrent_decode_pool(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    h_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    lower_bound: float | None = None,
    recurrent_layout: str = "k_major",
) -> torch.Tensor:
    """Single-step decode KDA update with in-kernel state-pool addressing.

    Same math as :func:`kda_recurrent_decode`, but reads ``h_pool[read_indices]``
    and writes ``h_pool[write_indices]`` inside the kernel (negative write ids
    skip the store), eliminating the python-side gather/scatter round-trip.
    ``h_pool`` is ``[num_pages, HV, K, V]`` fp32 (``[num_pages, HV, V, K]``
    for ``recurrent_layout="v_major"``) and is updated in place.
    Returns ``o [B, T, HV, V]``.
    """
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_recurrent_kda_pool,
    )

    return fused_recurrent_kda_pool(
        q,
        k,
        v,
        g_raw,
        beta,
        A_log,
        dt_bias,
        h_pool,
        read_indices,
        write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        recurrent_layout=recurrent_layout,
    )


def kda_recurrent_decode(
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
    lower_bound: float | None = None,
    beta_is_logit: bool = True,
):
    """Single-step decode KDA update.

    Returns ``(o [B, T, HV, V], final_state)``; ``final_state`` is a fresh tensor
    (not in-place), so the caller must write it back. Same layout as
    :func:`kda_chunk_prefill`; safe gate applied in-kernel.
    """
    # Unlike ``chunk_kda``, ``fused_recurrent_kda`` has no ``safe_gate`` flag: it
    # applies the safe gate whenever ``lower_bound`` is set.
    return _fla.fused_recurrent_kda(
        q,
        k,
        v,
        g_raw,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=beta_is_logit,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
