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

"""CuteDSL KDA adapter for the chunked KDA prefill scan.

States use the native ``[N, HV, V, K]`` convention, matching the NVIDIA
cache and the CuteDSL ABI. The dispatch facade adapts other cache layouts;
this wrapper must not transpose the state a second time. The native
token-major build reads ``[B, T, H, D]`` activations directly. Capacity
execution fuses padding cleanup, FP32 gate conversion and chunk planning.
Exact-length execution retains the public wrapper's preparation.
Sigmoid(beta), the safe gate, and QK L2 normalization run in-kernel, like the
FLA and FlashKDA paths. The safe-gate lower bound is baked into the CUBIN
and validated on every call.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult
from tokenspeed_kernel.ops.attention.kda._triton.prefill_scan_inputs import (
    prepare_capacity_scan,
)
from tokenspeed_kernel.ops.attention.kda.triton import (
    _DENSE_HALF_SIGNATURES,
    _nvidia_kda_prefill,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.thirdparty.cutedsl_kda import (
    cutedsl_kda_forward_with_prepared_plan,
    cutedsl_kda_supports_prepared_plan,
)

_SUPPORTED_ARCHES = frozenset({ArchVersion(10, 0), ArchVersion(10, 3)})


def cutedsl_kda_supported() -> bool:
    """Whether the current platform supports the packaged CuteDSL KDA kernel."""
    platform = current_platform()
    return platform.is_nvidia and platform.arch_version in _SUPPORTED_ARCHES


if cutedsl_kda_supported():
    from tokenspeed_cutedsl_kda import (
        cutedsl_kda_check_config,
        cutedsl_kda_forward,
        cutedsl_kda_workspace_size,
    )


__all__ = ["cutedsl_kda_chunk_prefill", "cutedsl_kda_supported"]


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="cutedsl_kda_nvidia_paged_prefill",
    solution="cutedsl_kda",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "recurrent_layout": frozenset({"v_major"}),
        "prefill_capacity": frozenset({True}),
    },
    tags={"nvidia", "paged_cache"},
)
def cutedsl_kda_nvidia_paged_prefill(**kwargs) -> KdaPrefillResult:
    # Capacity admission is checked against the actual CPU mirror by the
    # facade. Only native host planning receives the larger synthetic bounds;
    # device boundaries continue to identify the real packed token ranges.
    capacity = kwargs.pop("capacity", None)
    inputs_packed = kwargs.pop("inputs_packed", False)
    if (
        capacity is not None
        and kwargs["q"].is_cuda
        and kwargs["q"].dtype == torch.bfloat16
        and kwargs["q"].shape[-1] == kwargs["v"].shape[-1] == 128
        and all(
            t.stride(-1) == 1 and t.stride(-2) == 128
            for t in (kwargs["q"], kwargs["k"], kwargs["v"], kwargs["g_raw"])
        )
        and kwargs["beta_logits"].stride(-1) == 1
        and cutedsl_kda_supports_prepared_plan()
    ):
        if kwargs["lower_bound"] is None:
            raise ValueError("CuteDSL KDA requires a safe-gate bound")
        cutedsl_kda_check_config(float(kwargs["lower_bound"]))
        q, k, v, gate, beta, chunks, chunk_rows = prepare_capacity_scan(
            kwargs["q"],
            kwargs["k"],
            kwargs["v"],
            kwargs["g_raw"],
            kwargs["beta_logits"],
            kwargs["cu_seqlens"],
            inputs_packed,
        )
        out, state = cutedsl_kda_forward_with_prepared_plan(
            q,
            k,
            v,
            gate,
            kwargs["A_log"].contiguous(),
            kwargs["dt_bias"].reshape(q.shape[2], 128).contiguous(),
            beta,
            kwargs["cu_seqlens"].to(torch.int64),
            kwargs["initial_state"].contiguous(),
            token_capacity=capacity.token_capacity,
            cu_chunks=chunks,
            chunk_to_seq=chunk_rows,
            scale=1.0 / math.sqrt(q.shape[-1]),
        )
        return KdaPrefillResult(out, state)
    if capacity is not None:
        # Capacity descriptors make padding physically addressable to native
        # full-tile loads. Conv output padding is undefined, so scrub all scan
        # inputs from the live device boundary before normalization can see NaN.
        padding = (
            torch.arange(capacity.token_capacity, device=kwargs["q"].device)
            >= kwargs["cu_seqlens"][-1]
        )
        for name in ("q", "k", "v", "g_raw", "beta_logits"):
            tensor = kwargs[name]
            mask = padding.view(1, -1, *([1] * (tensor.ndim - 2)))
            kwargs[name] = tensor.masked_fill(mask, 0)
        kwargs["cu_seqlens_cpu"] = capacity.boundaries_cpu()
    return _nvidia_kda_prefill(cutedsl_kda_chunk_prefill, **kwargs)


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
    cu_seqlens_cpu: torch.Tensor | None = None,
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
            the native ``[N, HV, V, K]`` convention; ``None`` starts from
            zero.
        cu_seqlens: Cumulative sequence boundaries ``[N + 1]`` (``B`` must
            be 1); ``None`` treats each batch row as one sequence.
        cu_seqlens_cpu: Host int64 copy of ``cu_seqlens``, REQUIRED whenever
            ``cu_seqlens`` is given. Direct callers must supply equal contents.
            The registered adapter may instead supply validated per-sequence
            capacity bounds after explicit facade capacity admission. The
            kernel wrapper plans launch grids, routing, and workspace
            partitioning on the host from the boundary values; reading them
            back instead would be a stream-synchronizing D2H copy on every
            call. Runtime callers share a device int64 boundary tensor across
            layers, making the conversion below a no-op. Standalone callers
            may still supply int32 boundaries.
        lower_bound: Safe-gate lower bound; required, and must match the
            value baked into the CUBIN (validated).
        beta_is_logit: Must be True; the kernel always applies sigmoid.

    Returns:
        ``(o [B, T, HV, V], final_state [N, HV, V, K])`` in native layout.
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
    if key_dim != 128:
        raise ValueError(f"CuteDSL KDA requires key_dim=128, got {key_dim}")
    num_value_heads, value_dim = v.shape[2], v.shape[-1]
    if cu_seqlens is not None:
        num_sequences = cu_seqlens.numel() - 1
        boundaries = cu_seqlens.to(dtype=torch.int64)
        if cu_seqlens_cpu is None:
            raise ValueError(
                "cutedsl_kda_chunk_prefill requires cu_seqlens_cpu alongside "
                "cu_seqlens (host int64 copy with equal contents)"
            )
        if len(cu_seqlens_cpu) != num_sequences + 1:
            # A wrong copy would silently corrupt the host-side chunk plan;
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
        cu_seqlens_cpu = torch.arange(
            0, (batch + 1) * tokens, tokens, dtype=torch.int64
        )
        q, k, v = (t.reshape(1, batch * tokens, -1, t.shape[-1]) for t in (q, k, v))
        g_raw = g_raw.reshape(1, batch * tokens, num_value_heads, key_dim)
        beta = beta.reshape(1, batch * tokens, num_value_heads)
    # Native token-major ABI: no head-major re-layout. Gate must be FP32
    # and beta may be a strided slice of the merged projection; contiguous()
    # pins the token-major memory the kernel descriptors index.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g_f32 = g_raw.float().contiguous()
    beta = beta.contiguous()
    dt_bias = dt_bias.reshape(num_value_heads, key_dim).contiguous()
    A_log = A_log.contiguous()
    # The dispatch layout trait already matches the native [N, HV, V, K]
    # ABI. A second transpose here would undo the dispatcher's conversion.
    if initial_state is not None:
        state_in = initial_state.contiguous()
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
        scale=1.0 / math.sqrt(key_dim),
        workspace=workspace,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return out.view(batch, tokens, num_value_heads, value_dim), final_state
