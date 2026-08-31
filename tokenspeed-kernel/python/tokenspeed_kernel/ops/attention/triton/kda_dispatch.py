# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang,
# Zhiyuan Li
#
# The adapters in this file preserve the NVIDIA KDA implementations behind one
# public kernel contract.

"""Registered adapters for KDA implementations."""

from __future__ import annotations

from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.attention import KdaPrefillResult
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_DENSE_HALF_SIGNATURES = format_signatures(
    ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
)
# Descriptor addresses are dereferenced as bf16, so registrations exclude fp16.
_DENSE_BF16_SIGNATURES = format_signatures(("q", "k", "v"), "dense", {torch.bfloat16})


@register_kernel(
    "attention",
    "kda_fused_paged_decode",
    name="triton_nvidia_kda_fused_paged_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "fused_output_norm": frozenset({False, True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion"},
)
def triton_nvidia_kda_fused_paged_decode(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> torch.Tensor:
    """Adapt dev's NVIDIA conv/GEMV/recurrent megafusion."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_recurrent_kda_megafuse,
    )

    return fused_recurrent_kda_megafuse(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        h_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        output_gate=output_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    ).view(1, -1, num_heads, head_dim)


def _nvidia_fused_verify(
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
    store_states: bool,
    split_producers: bool = False,
) -> torch.Tensor:
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_kda_verify_conv_update,
        fused_recurrent_kda_verify_megafuse,
    )

    conv_qkv = None
    g_raw = None
    if split_producers:
        conv_qkv = fused_kda_verify_conv_update(
            mixed_qkv,
            conv_weights,
            conv_states,
            read_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            draft_token_num=draft_token_num,
        )
        g_raw = torch.mm(f_a_out, f_b_weight.t())

    return fused_recurrent_kda_verify_megafuse(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        # Aliasing the pool is only safe when nothing is stored into it.
        state_pool if state_scratch is None and not store_states else state_scratch,
        read_indices,
        write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=store_states,
        g_raw=g_raw,
        conv_qkv=conv_qkv,
    ).view(1, -1, num_heads, head_dim)


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "store_states": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion", "speculative"},
)
def triton_nvidia_kda_fused_paged_verify(
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
    state_scratch: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run fused target verify, storing per-position rollback states."""
    return _nvidia_fused_verify(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=True,
    )


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify_no_store",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "store_states": frozenset({False}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion", "speculative"},
)
def triton_nvidia_kda_fused_paged_verify_no_store(
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
) -> torch.Tensor:
    """Run fused target verify without materializing rollback states."""
    return _nvidia_fused_verify(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=False,
    )


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify_split",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_BF16_SIGNATURES,
    # Outranks the inline-producer twin so bf16 gets it without asking; fp16
    # is filtered out by signature and falls back to that twin.
    priority=Priority.SPECIALIZED + 1,
    traits={
        "paged_state": frozenset({True}),
        "store_states": frozenset({False}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion", "speculative"},
)
def triton_nvidia_kda_fused_paged_verify_split(
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
) -> torch.Tensor:
    """Run target verify with split convolution and gate producers."""
    return _nvidia_fused_verify(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=False,
        split_producers=True,
    )


@register_kernel(
    "attention",
    "kda_paged_decode",
    name="triton_nvidia_kda_paged_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.PERFORMANT,
    traits={
        "indexed_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph"},
)
def triton_nvidia_kda_paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Adapt dev's NVIDIA indexed recurrent decode kernel."""
    from tokenspeed_kernel.ops.attention.triton.linear.kda import (
        kda_recurrent_decode_pool,
    )

    return kda_recurrent_decode_pool(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        h_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )


def _nvidia_kda_prefill(
    implementation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    lower_bound: float | None,
) -> KdaPrefillResult:
    out, final_state = implementation(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        lower_bound=lower_bound,
        beta_is_logit=True,
    )
    return KdaPrefillResult(out, final_state)


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="triton_nvidia_kda_replay_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "flat_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "flat_kv", "fusion", "speculative"},
)
def triton_nvidia_kda_replay_commit(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_out: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    gate_scratch: torch.Tensor | None = None,
) -> None:
    """Replay the accepted prefix of a verified window into the state pool."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_recurrent_kda_replay_commit,
    )

    fused_recurrent_kda_replay_commit(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_out,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        state_out,
        read_indices,
        write_indices,
        accepted_length,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        gate_scratch=gate_scratch,
    )


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="triton_nvidia_kda_batched_replay_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_BF16_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "flat_state": frozenset({True}),
        "batched_layers": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "flat_kv", "fusion", "speculative", "batched_layers"},
)
def triton_nvidia_kda_batched_replay_commit(
    descriptors: torch.Tensor,
    *,
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
    gate_stride: int,
    conv_width: int,
    layers_per_group: int,
    lower_bound: float,
) -> None:
    """Replay every KDA layer described by stable device pointer tables."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        batched_recurrent_kda_replay_commit,
    )

    batched_recurrent_kda_replay_commit(
        descriptors,
        read_indices,
        write_indices,
        accepted_length,
        draft_token_num=draft_token_num,
        num_heads=num_heads,
        head_dim=head_dim,
        f_a_dim=f_a_dim,
        qkv_stride=qkv_stride,
        conv_stride=conv_stride,
        f_a_stride=f_a_stride,
        beta_stride=beta_stride,
        state_stride=state_stride,
        gate_stride=gate_stride,
        conv_width=conv_width,
        layers_per_group=layers_per_group,
        lower_bound=lower_bound,
    )


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="triton_nvidia_kda_paged_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.PERFORMANT,
    traits={"recurrent_layout": frozenset({"k_major"})},
    tags={"nvidia", "paged_cache"},
)
def triton_nvidia_kda_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.triton.linear.kda import (
        kda_chunk_prefill,
    )

    # The host boundaries feed FLA's chunk-index prep so it plans without a
    # stream-synchronizing D2H read of the varlen boundaries.
    return _nvidia_kda_prefill(kda_chunk_prefill, **kwargs)


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="flashkda_nvidia_kda_paged_prefill",
    solution="flashkda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={"recurrent_layout": frozenset({"k_major"})},
    tags={"nvidia", "paged_cache"},
)
def flashkda_nvidia_kda_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.flash_kda import flash_kda_chunk_prefill

    return _nvidia_kda_prefill(flash_kda_chunk_prefill, **kwargs)


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="cutedsl_kda_nvidia_paged_prefill",
    solution="cutedsl_kda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={"recurrent_layout": frozenset({"k_major"})},
    tags={"nvidia", "paged_cache"},
)
def cutedsl_kda_nvidia_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.cutedsl_kda import cutedsl_kda_chunk_prefill

    return _nvidia_kda_prefill(cutedsl_kda_chunk_prefill, **kwargs)
