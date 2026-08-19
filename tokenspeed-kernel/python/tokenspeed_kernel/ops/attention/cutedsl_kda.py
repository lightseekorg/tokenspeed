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

"""CuteDSL KDA drop-in for the chunked KDA prefill scan.

Mirrors ``triton.linear.kda.kda_chunk_prefill``'s signature and state
convention (FLA-native ``[N, HV, K, V]`` states) so the runtime can swap the
calls behind the same policy flag as FlashKDA. The **native
token-major** build reads the runtime's ``[B, T, H, D]`` activations and the
``[1, T, H]`` beta directly, so the only per-call data movement is the FP32
gate cast and one 128x128 state transpose per sequence (the state is the sole
canonical-layout operand). Sigmoid(beta), the safe gate, and QK L2
normalization run in-kernel, like the FLA and FlashKDA paths. The safe-gate
lower bound is baked into the CUBIN and validated on every call.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.thirdparty.cutedsl_kda import (
    DEFAULT_SCALE,
    cutedsl_kda_check_config,
    cutedsl_kda_forward,
    cutedsl_kda_workspace_size,
    is_cutedsl_kda_installed,
)

__all__ = ["cutedsl_kda_chunk_prefill", "is_cutedsl_kda_installed"]


def cutedsl_kda_chunk_prefill(
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
    cu_seqlens_cpu=None,
    lower_bound: float | None = None,
    beta_is_logit: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked prefill KDA scan through the CuteDSL KDA kernel (varlen native).

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
        cu_seqlens_cpu: Optional CPU copy of ``cu_seqlens`` (tuple, list, or
            CPU tensor) whose contents MUST equal ``cu_seqlens``. The kernel
            wrapper plans launch grids, routing, and workspace partitioning on
            the host from the boundary values; without this hint it reads them
            back with a stream-synchronizing D2H copy on every call (the
            per-call ``.to(int64)`` below allocates a fresh boundaries tensor,
            so the wrapper's identity memo never hits across layers).
        lower_bound: Safe-gate lower bound; required, and must match the
            value baked into the CUBIN (validated).
        beta_is_logit: Must be True; the kernel always applies sigmoid.

    Returns:
        ``(o [B, T, HV, V], final_state [N, HV, K, V])`` matching the FLA
        wrapper's convention.
    """
    if not beta_is_logit:
        raise ValueError("cutedsl_kda_chunk_prefill requires raw beta logits")
    if lower_bound is None:
        raise ValueError("cutedsl_kda_chunk_prefill requires a safe-gate bound")
    if dt_bias is None:
        raise ValueError("cutedsl_kda_chunk_prefill requires dt_bias")
    # The bound is a compile-time CUBIN constant; mismatches must fail
    # loudly rather than silently mis-gate.
    cutedsl_kda_check_config(float(lower_bound))
    batch, tokens, num_heads, key_dim = q.shape
    num_value_heads, value_dim = v.shape[2], v.shape[-1]
    if cu_seqlens is not None:
        num_sequences = cu_seqlens.numel() - 1
        boundaries = cu_seqlens.to(dtype=torch.int64)
        if cu_seqlens_cpu is not None and len(cu_seqlens_cpu) != num_sequences + 1:
            # A wrong hint would silently corrupt the host-side chunk plan;
            # a length mismatch means the caller wired the wrong tensor.
            raise ValueError(
                f"cu_seqlens_cpu has {len(cu_seqlens_cpu)} entries, "
                f"cu_seqlens has {num_sequences + 1}"
            )
    else:
        # The kernel is varlen-only with a unit batch dim; token-major
        # memory lets batch rows flatten to packed sequences as pure views.
        num_sequences = batch
        boundaries = torch.arange(
            0, (batch + 1) * tokens, tokens, device=q.device, dtype=torch.int64
        )
        cu_seqlens_cpu = tuple(range(0, (batch + 1) * tokens, tokens))
        q, k, v = (t.reshape(1, batch * tokens, -1, t.shape[-1]) for t in (q, k, v))
        g_raw = g_raw.reshape(1, batch * tokens, num_value_heads, key_dim)
        beta = beta.reshape(1, batch * tokens, num_value_heads)
    # Native token-major ABI: q/k/v/gate stay [1, T, H, D] and beta stays
    # [1, T, H]; the kernel reads them directly (no head-major re-layout, no
    # beta copy). gate must be fp32; contiguous() pins the token-major memory
    # the TMA descriptors index.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g_f32 = g_raw.float().contiguous()
    beta = beta.contiguous()
    dt_bias = dt_bias.reshape(num_value_heads, key_dim).contiguous()
    A_log = A_log.contiguous()
    # States: FLA-native [N, HV, K, V] -> canonical [N, HV, V, K].
    if initial_state is not None:
        state_in = initial_state.transpose(-1, -2).contiguous()
    else:
        state_in = torch.zeros(
            num_sequences,
            num_value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
    # Decomposition-route scratch (0 bytes on the engine route); preallocated
    # here so the wrapper does not allocate on the hot path.
    ws_bytes = cutedsl_kda_workspace_size(
        boundaries, num_value_heads, cu_seqlens_cpu=cu_seqlens_cpu
    )
    workspace = (
        torch.empty(ws_bytes, dtype=torch.uint8, device=q.device) if ws_bytes else None
    )
    out, final_state = cutedsl_kda_forward(
        q,
        k,
        v,
        g_f32,
        A_log,
        dt_bias,
        beta,
        boundaries,
        state_in,
        scale=DEFAULT_SCALE,
        workspace=workspace,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    # out is token-major [1, T, HV, V] already; state VK -> FLA-native KV.
    return out.view(batch, tokens, num_value_heads, value_dim), final_state.transpose(
        -1, -2
    )
