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
from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult, KdaReplayRecords
from tokenspeed_kernel.platform import CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_DENSE_HALF_SIGNATURES = format_signatures(
    ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
)


def kda_recurrent_decode_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    h_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    h_pool_out: torch.Tensor,
    lower_bound: float | None,
    recurrent_layout: str,
) -> torch.Tensor:
    """Run multi-token KDA decode against explicit read and write state pools."""
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        fused_recurrent_kda_mtp,
    )

    return fused_recurrent_kda_mtp(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        h_pool,
        read_indices,
        write_indices,
        h_pool_out=h_pool_out,
        scale=None,
        lower_bound=lower_bound,
        recurrent_layout=recurrent_layout,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
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
        "recurrent_layout": frozenset({"v_major"}),
    },
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
) -> torch.Tensor:
    """Adapt dev's NVIDIA conv/GEMV/recurrent megafusion."""
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
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
    g_raw: torch.Tensor | None = None,
    conv_qkv: torch.Tensor | None = None,
) -> torch.Tensor:
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        fused_kda_verify_conv_update,
        fused_recurrent_kda_verify_megafuse,
    )

    if split_producers:
        if conv_qkv is None:
            conv_qkv = fused_kda_verify_conv_update(
                mixed_qkv,
                conv_weights,
                conv_states,
                read_indices,
                num_heads=num_heads,
                head_dim=head_dim,
                draft_token_num=draft_token_num,
            )
        if g_raw is None:
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
        enable_pdl=pdl_enabled(),
    ).view(1, -1, num_heads, head_dim)


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
        bias=bias,
        original_qkv=original_qkv,
    )


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
        "recurrent_layout": frozenset({"v_major"}),
        "split_producers": frozenset({False}),
        "store_states": frozenset({True}),
    },
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
        split_producers=False,
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
        "recurrent_layout": frozenset({"v_major"}),
        "split_producers": frozenset({False}),
        "store_states": frozenset({False}),
    },
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
        split_producers=False,
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
        "store_states": frozenset({False}),
    },
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
    replay_records: KdaReplayRecords | None,
    g_raw: torch.Tensor | None = None,
    conv_qkv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run target verify with split convolution and gate producers."""
    if replay_records is not None:
        from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
            vllm_triton_kda_recoverssm_verify,
        )

        if conv_qkv is None or g_raw is None:
            raise ValueError("record verify requires precomputed conv and gates")
        if replay_records.original_qkv is not None:
            destination = replay_records.original_qkv
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
                    raise ValueError(
                        "Original QKV destination must not overlap producers"
                    )
            destination.copy_(mixed_qkv)
        q, k, v = (
            part.view(1, -1, num_heads, head_dim) for part in conv_qkv.chunk(3, dim=-1)
        )
        return vllm_triton_kda_recoverssm_verify(
            q,
            k,
            v,
            g_raw.view_as(q),
            beta_logits.view(1, -1, num_heads),
            A_log,
            dt_bias,
            checkpoint_state=state_pool[:, :num_heads],
            correction_cache=replay_records.correction,
            kg_cache=replay_records.key_gate,
            query_start_loc=replay_records.query_start_loc,
            state_indices=read_indices,
            spec_query_len=draft_token_num,
            lower_bound=lower_bound,
            out=torch.empty_like(v),
            record_metadata=replay_records.metadata,
            record_stamps=replay_records.stamps,
        )
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
        g_raw=g_raw,
        conv_qkv=conv_qkv,
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
    from tokenspeed_kernel.ops.attention.kda._triton.fla import (
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
    name="triton_nvidia_kda_batched_replay_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "batched_layers": frozenset({True}),
        "flat_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def triton_nvidia_kda_batched_replay_commit(
    descriptors: torch.Tensor,
    *,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    activation_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    qkv_stride: int,
    conv_stride: int,
    correction_stride: int,
    key_gate_stride: int,
    state_stride: int,
    conv_width: int,
    lower_bound: float | None,
    record_metadata: torch.Tensor,
    record_stamps: torch.Tensor,
) -> None:
    """Recover every KDA layer from FP32 U and saved K/raw-gate records.

    Args:
        descriptors: Device pointers for every layer's inputs, weights, and state.
        group_indices: Cache-group row for each descriptor.
        read_indices: Source page indices, shaped ``[groups, batch]``.
        write_indices: Destination page indices, shaped ``[groups, batch]``.
        accepted_length: Accepted draft-token count for each request.
        activation_dtype: Storage dtype of raw QKV and saved K/raw-gate records.
        conv_state_dtype: Storage dtype of the bound convolution history pool.
        draft_token_num: Maximum number of draft tokens in the replay window.
        num_heads: Number of local KDA value heads.
        head_dim: Per-head key and value dimension.
        qkv_stride: Token stride of the packed QKV payload.
        conv_stride: Page stride of the convolution state.
        correction_stride: Token stride of the FP32 correction payload.
        key_gate_stride: Token stride of unnormalized key/raw-gate records.
        state_stride: Page stride of the recurrent state.
        conv_width: Width of the depthwise convolution kernel.
        lower_bound: Bound used to transform saved raw gates during commit.

    Returns:
        None.
    """
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        batched_recoverssm_commit,
    )

    batched_recoverssm_commit(
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        activation_dtype=activation_dtype,
        conv_state_dtype=conv_state_dtype,
        draft_token_num=draft_token_num,
        num_heads=num_heads,
        head_dim=head_dim,
        qkv_stride=qkv_stride,
        conv_stride=conv_stride,
        correction_stride=correction_stride,
        key_gate_stride=key_gate_stride,
        state_stride=state_stride,
        conv_width=conv_width,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
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
)
