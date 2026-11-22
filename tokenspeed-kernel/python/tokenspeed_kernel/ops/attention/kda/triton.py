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
from tokenspeed_kernel.ops.attention.kda import (
    KdaOutputNorm,
    KdaPrefillResult,
    KdaRetainedRecords,
    apply_kda_output_norm,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_DENSE_HALF_SIGNATURES = format_signatures(
    ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
)


@register_kernel(
    "attention",
    "kda_verify_conv_update",
    name="triton_kda_verify_conv_update",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
        "split_producers": frozenset({True}),
    },
)
def triton_kda_verify_conv_update(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    read_indices: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    query_start_loc: torch.Tensor,
    bias: torch.Tensor | None = None,
    original_qkv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the shared split-verify convolution producer."""
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        fused_kda_verify_conv_update,
    )

    return fused_kda_verify_conv_update(
        mixed_qkv,
        conv_weights,
        conv_states,
        read_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        query_start_loc=query_start_loc,
        bias=bias,
        original_qkv=original_qkv,
    )


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify_split",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    # Serving enables this producer arrangement only with compatible recovery.
    priority=Priority.SPECIALIZED + 1,
    traits={
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
        "split_producers": frozenset({True}),
        "retained_history": frozenset({True}),
    },
)
def triton_nvidia_kda_fused_paged_verify_split(
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
    retained_records: KdaRetainedRecords,
    output_norm: KdaOutputNorm | None,
    g_raw: torch.Tensor | None = None,
    conv_qkv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run target verify with split convolution and gate producers."""
    records = retained_records
    from tokenspeed_kernel.ops.attention.kda import _attention_format_signature
    from tokenspeed_kernel.selection import select_kernel

    if g_raw is None:
        g_raw = torch.nn.functional.linear(f_a_out, f_b_weight)
    if conv_qkv is None:
        conv_qkv = triton_kda_verify_conv_update(
            mixed_qkv,
            conv_weights,
            conv_states,
            read_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            draft_token_num=draft_token_num,
            query_start_loc=records.query_start_loc,
            bias=None,
            original_qkv=None,
        )
    destination = records.original_qkv
    if destination is not None:
        if (
            destination.shape != mixed_qkv.shape
            or destination.dtype != mixed_qkv.dtype
            or destination.device != mixed_qkv.device
            or destination.stride(-1) != 1
            or destination.stride(0) < destination.shape[-1]
        ):
            raise ValueError("Original QKV destination must match the input rows")
        storage = destination.untyped_storage()
        begin, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        for source in (mixed_qkv, g_raw, beta_logits):
            source_storage = source.untyped_storage()
            if (
                begin < source_storage.data_ptr() + source_storage.nbytes()
                and source_storage.data_ptr() < end
            ):
                raise ValueError("Original QKV destination must not overlap producers")
    q, k, v = (
        part.view(1, -1, num_heads, head_dim) for part in conv_qkv.chunk(3, dim=-1)
    )
    verify = select_kernel(
        "attention",
        "kda_recoverssm_verify",
        _attention_format_signature(q=q, k=k, v=v),
        traits={
            "paged_state": True,
            "recurrent_layout": "v_major",
            "head_dim": head_dim,
            "spec_query_len": draft_token_num,
            "state_dtype": state_pool.dtype,
        },
    )
    out = verify(
        q,
        k,
        v,
        g_raw.view_as(q),
        beta_logits.view(1, -1, num_heads),
        A_log,
        dt_bias,
        checkpoint_state=state_pool[:, :num_heads],
        retained_history=retained_records.history,
        query_start_loc=records.query_start_loc,
        state_indices=read_indices,
        spec_query_len=draft_token_num,
        lower_bound=lower_bound,
        out=torch.empty_like(v),
        record_metadata=records.metadata,
        record_stamps=records.stamps,
    )
    if destination is not None:
        destination.copy_(mixed_qkv)
    return apply_kda_output_norm(out, output_norm)


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
    "kda_paged_prefill",
    name="triton_nvidia_kda_paged_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.PERFORMANT,
    traits={"recurrent_layout": frozenset({"k_major"})},
)
def triton_nvidia_kda_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.kda._triton.fla import (
        kda_chunk_prefill,
    )

    # The host boundaries feed FLA's chunk-index prep so it plans without a
    # stream-synchronizing D2H read of the varlen boundaries.
    return _nvidia_kda_prefill(kda_chunk_prefill, **kwargs)


from tokenspeed_kernel.ops.attention.kda._triton.state_pages import (  # noqa: E402
    commit_state_groups,
    commit_state_pages,
    verify_state_blocks,
    verify_state_groups,
)


@register_kernel(
    "attention",
    "kda_retained_commit",
    name="triton_nvidia_kda_batched_retained_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "batched_layers": frozenset({True}),
        "retained_history": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def triton_nvidia_kda_batched_retained_commit(**kwargs) -> None:
    """Commit packed convolution windows across layers, then publish lengths."""
    from tokenspeed_kernel.ops.attention.kda._retained import batched_commit

    batched_commit(**kwargs)


@register_kernel(
    "attention",
    "kda_retained_prepare",
    name="triton_kda_batched_retained_prepare",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "batched_layers": frozenset({True}),
        "retained_history": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def triton_kda_batched_retained_prepare(**kwargs) -> None:
    """Fold accepted records into live pages across layers, then clear lengths."""
    from tokenspeed_kernel.ops.attention.kda._retained import batched_prepare

    batched_prepare(**kwargs)
