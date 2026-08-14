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

import torch
from tokenspeed_kernel.ops.attention.kda_utils import KdaPrefillResult
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_DENSE_HALF_SIGNATURES = format_signatures(
    ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
)


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
        "fused_output_norm": frozenset({False}),
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
    del output_gate, norm_weight, norm_eps
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
    ).view(1, -1, num_heads, head_dim)


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={"paged_state": frozenset({True})},
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion", "speculative"},
)
def triton_nvidia_kda_fused_paged_verify(
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
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    prev_qkv: torch.Tensor | None = None,
    prev_f_a: torch.Tensor | None = None,
    prev_beta: torch.Tensor | None = None,
    prev_base: torch.Tensor | None = None,
    prev_steps: torch.Tensor | None = None,
    commit_indices: torch.Tensor | None = None,
    enable_pdl: bool = False,
    gate_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adapt the NVIDIA conv/GEMV/recurrent megafusion to target verify.

    Writes no state of its own; with the ``prev_*`` args staged it also
    replays and commits the previous round's accepted prefix on the way in
    (the deferred lazy commit -- see the kernel docstring).
    """
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_recurrent_kda_verify_megafuse,
    )

    return fused_recurrent_kda_verify_megafuse(
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
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        prev_qkv=prev_qkv,
        prev_f_a=prev_f_a,
        prev_beta=prev_beta,
        prev_base=prev_base,
        prev_steps=prev_steps,
        commit_indices=commit_indices,
        enable_pdl=enable_pdl,
        gate_scratch=gate_scratch,
    ).view(1, -1, num_heads, head_dim)


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="triton_nvidia_kda_replay_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={"flat_state": frozenset({True})},
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
    "kda_paged_decode",
    name="triton_nvidia_kda_paged_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.PERFORMANT,
    traits={"indexed_state": frozenset({True})},
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
    implementation,
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
        lower_bound=lower_bound,
        beta_is_logit=True,
    )
    return KdaPrefillResult(out, final_state)


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="triton_nvidia_kda_paged_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.PERFORMANT,
    tags={"nvidia", "paged_cache"},
)
def triton_nvidia_kda_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.triton.linear.kda import (
        kda_chunk_prefill,
    )

    return _nvidia_kda_prefill(kda_chunk_prefill, **kwargs)


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="flashkda_nvidia_kda_paged_prefill",
    solution="flashkda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
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
    tags={"nvidia", "paged_cache"},
)
def cutedsl_kda_nvidia_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.cutedsl_kda import cutedsl_kda_chunk_prefill

    return _nvidia_kda_prefill(cutedsl_kda_chunk_prefill, **kwargs)
