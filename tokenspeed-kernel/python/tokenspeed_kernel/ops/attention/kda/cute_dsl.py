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

"""CuteDSL KDA adapters: the chunked prefill scan, the speculative verify and
the batched replay.

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
    _DENSE_BF16_SIGNATURES,
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


__all__ = [
    "cutedsl_kda_chunk_prefill",
    "cutedsl_kda_supported",
    "cutedsl_nvidia_kda_batched_replay_commit",
    "cutedsl_nvidia_kda_fused_paged_verify",
]


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
        "prefill_capacity": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def cutedsl_kda_nvidia_paged_prefill(**kwargs) -> KdaPrefillResult:
    # Capacity admission is checked against the actual CPU mirror by the
    # facade. Only native host planning receives the larger synthetic bounds;
    # device boundaries continue to identify the real packed token ranges.
    capacity = kwargs.pop("capacity", None)
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
            kwargs["initial_state"].float().contiguous(),
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
        initial_state: Optional BF16 or FP32 stored state, accumulated in FP32, per sequence in
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
        state_in = initial_state.float().contiguous()
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


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="cutedsl_nvidia_kda_fused_paged_verify",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=_DENSE_BF16_SIGNATURES,
    # Outranks the Triton split verify (SPECIALIZED + 1). Same coverage as the
    # raw-gate batched replay below, which reads the payload only this verify
    # writes.
    priority=Priority.SPECIALIZED + 2,
    traits={
        "head_dim": frozenset({128}),
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
        "split_producers": frozenset({False}),
        "store_states": frozenset({False}),
    },
)
def cutedsl_nvidia_kda_fused_paged_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    replay_mixed_qkv: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
) -> torch.Tensor:
    """Run target verify and fill the raw-gate replay payload; no rollback states.

    The gate GEMM writes the raw gate into ``replay_gate``; one persistent
    CuTe DSL kernel then convolves the raw projections from the committed
    window, runs the recurrence per draft position, writes the output and
    copies the raw QKV and beta rows into the payload. Per-position conv
    windows and states are not materialized (``conv_scratch``,
    ``state_scratch`` and ``write_indices`` are unused): the batched replay
    recomputes the accepted prefix.

    Args:
        mixed_qkv: ``[batch * draft_token_num, >= 3 * P]`` raw projections.
        conv_weights: ``[3 * P, 4]`` convolution taps.
        conv_states: ``[pages, 3 * P, 3]`` committed convolution windows.
        conv_scratch: Unused.
        f_a_out: Low-rank gate activation.
        f_b_weight: Second gate projection.
        beta_logits: Raw beta.
        A_log: Per-head FP32 decay parameter.
        dt_bias: FP32 gate bias.
        state_pool: ``[pages, num_heads, 128, 128]`` V-major states.
        state_scratch: Unused.
        read_indices: ``[batch]`` committed pages (-1 = fresh request).
        write_indices: Unused.
        num_heads: Local KDA heads.
        head_dim: 128.
        draft_token_num: Draft window tokens per request.
        lower_bound: Safe-gate lower bound.
        replay_mixed_qkv: Payload view for the raw QKV rows.
        replay_gate: BF16 payload view for the raw gate.
        replay_beta: Payload view for the raw beta.

    Returns:
        ``[1, batch * draft_token_num, num_heads, 128]`` attention output.
    """
    from tokenspeed_kernel.ops.attention.kda._cute_dsl.verify import (
        kda_fused_verify,
    )

    if head_dim != 128:
        raise ValueError("CuTe DSL KDA verify requires head_dim=128")
    if lower_bound is None:
        raise ValueError("CuTe DSL KDA verify requires a safe-gate bound")
    return kda_fused_verify(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        read_indices,
        replay_mixed_qkv,
        replay_gate,
        replay_beta,
        num_heads=num_heads,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
    ).view(1, -1, num_heads, head_dim)


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="cutedsl_nvidia_kda_batched_replay_commit",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=_DENSE_BF16_SIGNATURES,
    priority=Priority.PERFORMANT,
    traits={
        "batched_layers": frozenset({True}),
        "flat_state": frozenset({True}),
        "head_dim": frozenset({128}),
        "recurrent_layout": frozenset({"v_major"}),
        "replay_raw_gate": frozenset({True}),
    },
)
def cutedsl_nvidia_kda_batched_replay_commit(
    descriptors: torch.Tensor,
    *,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    f_a_dim: int,
    qkv_stride: int,
    conv_stride: int,
    f_a_stride: int,
    beta_stride: int,
    state_stride: int,
    state_dtype: torch.dtype,
    gate_stride: int,
    conv_width: int,
    lower_bound: float,
) -> None:
    """Replay every KDA layer described by stable device pointer tables.

    Raw-gate contract: the gate descriptor column holds the BF16 raw gate that
    ``cutedsl_nvidia_kda_fused_paged_verify`` wrote, and the recurrence applies
    the decay gate in-kernel as one persistent CuTe DSL kernel. The
    convolution-window publication is the shared Triton launch.

    Args:
        descriptors: Device pointers for every layer's inputs, weights, and state.
        group_indices: Cache-group row for each descriptor.
        read_indices: Source page indices, shaped ``[groups, batch]``.
        write_indices: Destination page indices, shaped ``[groups, batch]``.
        accepted_length: Accepted draft-token count for each request.
        draft_token_num: Maximum number of draft tokens in the replay window.
        num_heads: Number of local KDA value heads.
        head_dim: Per-head key and value dimension (128).
        f_a_dim: Unused (the raw gate is already projected).
        qkv_stride: Token stride of the packed QKV payload.
        conv_stride: Page stride of the convolution state.
        f_a_stride: Unused.
        beta_stride: Token stride of beta.
        state_stride: Page stride of the recurrent state.
        state_dtype: Storage dtype of every recurrent-state descriptor.
        gate_stride: Token stride of the raw-gate payload.
        conv_width: Width of the depthwise convolution kernel (4).
        lower_bound: Lower bound used by the KDA decay gate.

    Returns:
        None.
    """
    from tokenspeed_kernel.ops.attention.kda._cute_dsl.replay import (
        kda_batched_replay,
    )
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        batched_kda_replay_conv_window,
        check_batched_replay_args,
    )

    check_batched_replay_args(
        descriptors, group_indices, head_dim=head_dim, state_dtype=state_dtype
    )
    if conv_width != 4:
        raise ValueError("CuTe DSL KDA replay requires conv_width=4")
    kda_batched_replay(
        descriptors.view(torch.int64),
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        draft_token_num=draft_token_num,
        num_heads=num_heads,
        qkv_stride=qkv_stride,
        conv_stride=conv_stride,
        beta_stride=beta_stride,
        state_stride=state_stride,
        state_dtype=state_dtype,
        gate_stride=gate_stride,
        lower_bound=lower_bound,
    )
    batched_kda_replay_conv_window(
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        draft_token_num=draft_token_num,
        num_heads=num_heads,
        head_dim=head_dim,
        qkv_stride=qkv_stride,
        conv_stride=conv_stride,
    )
