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

"""Registration shims for AMD Gluon KDA attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.kda import (
    KdaPrefillResult,
    KdaReplayRecords,
    KdaRetainedHistory,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_decode_gfx950 as _kda_fused_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_verify_gfx950 as _kda_fused_verify_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_recurrent_decode_gfx950 as _kda_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        launch_gluon_kda_fused_replay_gfx950 as _kda_fused_replay_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        launch_gluon_kda_retained_commit_gfx950 as _retained_commit_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        launch_gluon_kda_retained_verify_gfx950 as _retained_verify_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.prefill import (
        launch_gluon_kda_paged_prefill_gfx950 as _kda_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        gluon_kda_fused_decode_gfx1250 as _kda_fused_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        gluon_kda_fused_verify_gfx1250 as _kda_fused_verify_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        gluon_kda_recurrent_decode_gfx1250 as _kda_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        launch_gluon_kda_fused_replay_gfx1250 as _kda_fused_replay_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        launch_gluon_kda_retained_commit_gfx1250 as _retained_commit_gfx1250,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.decode import (
        launch_gluon_kda_retained_verify_gfx1250 as _retained_verify_gfx1250,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.prefill import (
        launch_gluon_kda_paged_prefill_gfx1250 as _kda_prefill_gfx1250_impl,
    )

    @register_kernel(
        "attention",
        "kda_paged_prefill",
        name="gluon_kda_paged_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
    )
    def gluon_kda_paged_prefill_gfx950(**kwargs) -> KdaPrefillResult:
        """Run specialized gfx950 KDA prefill with V-major state."""
        # Host-boundary hint is consumed only by the CuteDSL wrapper.
        kwargs.pop("cu_seqlens_cpu", None)
        output, final_state = _kda_prefill_impl(**kwargs)
        return KdaPrefillResult(out=output, final_state=final_state)

    @register_kernel(
        "attention",
        "kda_paged_prefill",
        name="gluon_kda_paged_prefill_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
    )
    def gluon_kda_paged_prefill_gfx1250(**kwargs) -> KdaPrefillResult:
        """Run specialized gfx1250 KDA prefill with V-major state."""
        # Host-boundary hint is consumed only by the CuteDSL wrapper.
        kwargs.pop("cu_seqlens_cpu", None)
        output, final_state = _kda_prefill_gfx1250_impl(**kwargs)
        return KdaPrefillResult(out=output, final_state=final_state)

    @register_kernel(
        "attention",
        "kda_paged_decode",
        name="gluon_kda_paged_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "indexed_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "single_token": frozenset({True}),
        },
    )
    def gluon_kda_paged_decode_gfx950(**kwargs):
        """Run specialized gfx950 KDA decode against the physical V-major pool."""
        return _kda_decode_impl(**kwargs)

    @register_kernel(
        "attention",
        "kda_fused_paged_decode",
        name="gluon_kda_fused_paged_decode_vmajor_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "conv_kernel_size": frozenset({4}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
        },
    )
    def gluon_kda_fused_paged_decode_vmajor_gfx950(
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
    ):
        """Run V-major gfx950 fused decode, decay projection included."""
        return _kda_fused_decode_impl(
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            conv_states=conv_states,
            f_a_out=f_a_out,
            f_b_weight=f_b_weight,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

    def _gluon_kda_fused_paged_verify_vmajor_gfx950(
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
        replay_records: KdaReplayRecords,
        g_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse gate, convolution and verify; a supplied raw gate takes precedence."""
        if replay_records is None:
            raise ValueError("AMD no-store verify requires recovery records")
        return _kda_fused_verify_impl(
            mixed_qkv=mixed_qkv,
            original_qkv=replay_records.original_qkv,
            conv_weights=conv_weights,
            conv_pool=conv_states,
            raw_g=g_raw,
            f_a_out=f_a_out if g_raw is None else None,
            f_b_weight=f_b_weight if g_raw is None else None,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            correction=replay_records.correction,
            key_gate=replay_records.key_gate,
            record_metadata=replay_records.metadata,
            record_stamps=replay_records.stamps,
            query_start_loc=replay_records.query_start_loc,
            num_heads=num_heads,
            head_dim=head_dim,
            draft_token_num=draft_token_num,
            lower_bound=lower_bound,
        )

    @register_kernel(
        "attention",
        "kda_fused_paged_verify",
        name="gluon_kda_fused_paged_verify_nostore_vmajor_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "store_states": frozenset({False}),
            "split_producers": frozenset({False}),
        },
    )
    def gluon_kda_fused_paged_verify_nostore_vmajor_gfx950(*args, **kwargs):
        return _gluon_kda_fused_paged_verify_vmajor_gfx950(*args, **kwargs)

    @register_kernel(
        "attention",
        "kda_replay_commit",
        name="gluon_kda_fused_replay_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "batched_layers": frozenset({True}),
            "flat_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
        },
    )
    def gluon_kda_fused_replay_gfx950(
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
        record_metadata: torch.Tensor | None,
        record_stamps: torch.Tensor | None,
    ) -> None:
        """Recover accepted prefixes from FP32 U and activation-dtype K/raw-g records."""
        _kda_fused_replay_impl(
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
            lower_bound=lower_bound,
            record_metadata=record_metadata,
            record_stamps=record_stamps,
        )

    @register_kernel(
        "attention",
        "kda_paged_decode",
        name="gluon_kda_paged_decode_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "indexed_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "single_token": frozenset({True}),
        },
    )
    def gluon_kda_paged_decode_gfx1250(**kwargs):
        """Run specialized gfx1250 KDA decode against the physical V-major pool."""
        return _kda_decode_gfx1250_impl(**kwargs)

    @register_kernel(
        "attention",
        "kda_fused_paged_decode",
        name="gluon_kda_fused_paged_decode_vmajor_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "conv_kernel_size": frozenset({4}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
        },
    )
    def gluon_kda_fused_paged_decode_vmajor_gfx1250(
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
    ):
        """Run V-major gfx1250 fused decode, decay projection included."""
        return _kda_fused_decode_gfx1250_impl(
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            conv_states=conv_states,
            f_a_out=f_a_out,
            f_b_weight=f_b_weight,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

    def _gluon_kda_fused_paged_verify_vmajor_gfx1250(
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
        replay_records: KdaReplayRecords,
        g_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse gate, convolution and verify; a supplied raw gate takes precedence."""
        if replay_records is None:
            raise ValueError("AMD no-store verify requires recovery records")
        return _kda_fused_verify_gfx1250_impl(
            mixed_qkv=mixed_qkv,
            original_qkv=replay_records.original_qkv,
            conv_weights=conv_weights,
            conv_pool=conv_states,
            raw_g=g_raw,
            f_a_out=f_a_out if g_raw is None else None,
            f_b_weight=f_b_weight if g_raw is None else None,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            correction=replay_records.correction,
            key_gate=replay_records.key_gate,
            record_metadata=replay_records.metadata,
            record_stamps=replay_records.stamps,
            query_start_loc=replay_records.query_start_loc,
            num_heads=num_heads,
            head_dim=head_dim,
            draft_token_num=draft_token_num,
            lower_bound=lower_bound,
        )

    @register_kernel(
        "attention",
        "kda_fused_paged_verify",
        name="gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "store_states": frozenset({False}),
            "split_producers": frozenset({False}),
        },
    )
    def gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250(*args, **kwargs):
        return _gluon_kda_fused_paged_verify_vmajor_gfx1250(*args, **kwargs)

    @register_kernel(
        "attention",
        "kda_replay_commit",
        name="gluon_kda_fused_replay_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "batched_layers": frozenset({True}),
            "flat_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
        },
    )
    def gluon_kda_fused_replay_gfx1250(
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
        record_metadata: torch.Tensor | None,
        record_stamps: torch.Tensor | None,
    ) -> None:
        """Recover accepted prefixes from FP32 U and activation-dtype K/raw-g records."""
        _kda_fused_replay_gfx1250_impl(
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
            lower_bound=lower_bound,
            record_metadata=record_metadata,
            record_stamps=record_stamps,
        )


def _validate_retained_request(
    history, reads, starts, metadata, stamps, *, heads, dim, window, pages, device
):
    """Check current request geometry; immutable storage was checked at construction."""
    batch = reads.numel()
    if (
        history.correction.shape[:2] != (pages, heads)
        or history.correction.shape[-1] != dim
        or history.key.shape[-1] != dim
    ):
        raise ValueError("retained history must match checkpoint page/head geometry")
    if (
        history.live_state_indices.shape != (batch,)
        or reads.shape != (batch,)
        or starts.shape != (batch + 1,)
    ):
        raise ValueError("retained history indices must match request/query boundaries")
    if any(x.dtype not in (torch.int32, torch.int64) for x in (reads, starts)):
        raise ValueError("retained KDA request indices must be integer tensors")
    if window <= 0 or window > history.correction.shape[2]:
        raise ValueError("retained KDA candidate window exceeds history capacity")
    if any(x.device != device for x in (history.correction, reads, starts)):
        raise ValueError("retained history and requests must use the checkpoint device")
    if (metadata is None) != (stamps is None):
        raise ValueError("retained KDA metadata and stamps must be supplied together")
    if metadata is not None:
        if (
            metadata.shape != (batch, 3)
            or stamps.ndim != 3
            or stamps.shape[0] < batch
            or (stamps.shape[1] < window or stamps.shape[2] != 2)
            or any(
                x.dtype != torch.int64 or not x.is_contiguous() or x.device != device
                for x in (metadata, stamps)
            )
        ):
            raise ValueError("retained KDA metadata must match requests and capacity")


def gluon_kda_retained_verify_gfx950(
    *,
    mixed_qkv: torch.Tensor,
    original_qkv: torch.Tensor | None,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    raw_g: torch.Tensor | None,
    f_a_out: torch.Tensor | None,
    f_b_weight: torch.Tensor | None,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    retained_history: KdaRetainedHistory,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Verify packed pre-convolution QKV using page-owned accepted history.

    conv_states and state_pool contain source checkpoints. retained_history
    selects private live pages and FP32 U/normalized-K/decay storage. Verify
    writes candidates after the accepted prefix, without publishing length.
    original_qkv optionally retains input rows for commit. raw_g excludes
    f_a_out/f_b_weight; otherwise both projection inputs are required.
    record_metadata and record_stamps are both tensors or both None.
    num_heads/head_dim describe the checkpoint; draft_token_num bounds each
    packed interval in query_start_loc. lower_bound selects the decay formula.
    Return attention output with shape [1, tokens, heads, dim].
    """
    if (
        state_pool.ndim != 4
        or state_pool.shape[1:] != (num_heads, head_dim, head_dim)
        or state_pool.dtype != torch.float32
    ):
        raise ValueError("retained KDA checkpoint must be FP32 [page, head, V, K]")
    _validate_retained_request(
        retained_history,
        read_indices,
        query_start_loc,
        record_metadata,
        record_stamps,
        heads=num_heads,
        dim=head_dim,
        window=draft_token_num,
        pages=state_pool.shape[0],
        device=state_pool.device,
    )
    return _retained_verify_gfx950(
        mixed_qkv=mixed_qkv,
        original_qkv=original_qkv,
        conv_weights=conv_weights,
        conv_pool=conv_states,
        raw_g=raw_g,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        query_start_loc=query_start_loc,
        correction=retained_history.correction,
        retained_key=retained_history.key,
        retained_decay=retained_history.decay,
        retained_length=retained_history.length,
        live_indices=retained_history.live_state_indices,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
    )


def gluon_kda_retained_commit_gfx950(
    *,
    original_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    read_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_length: torch.Tensor,
    retained_history: KdaRetainedHistory,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
) -> None:
    """Commit accepted packed original_qkv rows to live convolution pages.

    conv_states contains source taps. read_indices and query_start_loc map
    request rows; accepted_length is clamped to each row's actual token count.
    retained_history identifies private live pages and the length to publish.
    Metadata/stamps must be supplied together; stale accepted records suppress
    both convolution updates and length publication. num_heads, head_dim and
    draft_token_num specify fixed geometry. Return None.
    """
    _validate_retained_request(
        retained_history,
        read_indices,
        query_start_loc,
        record_metadata,
        record_stamps,
        heads=num_heads,
        dim=head_dim,
        window=draft_token_num,
        pages=conv_states.shape[0],
        device=conv_states.device,
    )
    batch = read_indices.numel()
    if (
        accepted_length.shape != (batch,)
        or accepted_length.dtype not in (torch.int32, torch.int64)
        or accepted_length.device != conv_states.device
    ):
        raise ValueError("retained KDA accepted lengths must match the requests")
    if (
        head_dim != 128
        or conv_states.shape[1:] != (3 * num_heads * head_dim, 3)
        or original_qkv.ndim != 2
        or original_qkv.shape[1] != 3 * num_heads * head_dim
        or original_qkv.shape[0] > batch * draft_token_num
        or original_qkv.stride(-1) != 1
        or original_qkv.device != conv_states.device
    ):
        raise ValueError(
            "retained KDA commit needs packed pre-convolution QKV and matching taps"
        )
    _retained_commit_gfx950(
        original_qkv=original_qkv,
        conv_pool=conv_states,
        read_indices=read_indices,
        live_indices=retained_history.live_state_indices,
        query_start_loc=query_start_loc,
        accepted_length=accepted_length,
        retained_length=retained_history.length,
        history_capacity=retained_history.correction.shape[2],
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
    )


def gluon_kda_retained_verify_gfx1250(
    *,
    mixed_qkv: torch.Tensor,
    original_qkv: torch.Tensor | None,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    raw_g: torch.Tensor | None,
    f_a_out: torch.Tensor | None,
    f_b_weight: torch.Tensor | None,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    retained_history: KdaRetainedHistory,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Verify packed pre-convolution QKV using page-owned accepted history.

    conv_states and state_pool contain source checkpoints. retained_history
    selects private live pages and FP32 U/normalized-K/decay storage. Verify
    writes candidates after the accepted prefix, without publishing length.
    original_qkv optionally retains input rows for commit. raw_g excludes
    f_a_out/f_b_weight; otherwise both projection inputs are required.
    record_metadata and record_stamps are both tensors or both None.
    num_heads/head_dim describe the checkpoint; draft_token_num bounds each
    packed interval in query_start_loc. lower_bound selects the decay formula.
    Return attention output with shape [1, tokens, heads, dim].
    """
    if (
        state_pool.ndim != 4
        or state_pool.shape[1:] != (num_heads, head_dim, head_dim)
        or state_pool.dtype != torch.float32
    ):
        raise ValueError("retained KDA checkpoint must be FP32 [page, head, V, K]")
    _validate_retained_request(
        retained_history,
        read_indices,
        query_start_loc,
        record_metadata,
        record_stamps,
        heads=num_heads,
        dim=head_dim,
        window=draft_token_num,
        pages=state_pool.shape[0],
        device=state_pool.device,
    )
    return _retained_verify_gfx1250(
        mixed_qkv=mixed_qkv,
        original_qkv=original_qkv,
        conv_weights=conv_weights,
        conv_pool=conv_states,
        raw_g=raw_g,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        query_start_loc=query_start_loc,
        correction=retained_history.correction,
        retained_key=retained_history.key,
        retained_decay=retained_history.decay,
        retained_length=retained_history.length,
        live_indices=retained_history.live_state_indices,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
    )


def gluon_kda_retained_commit_gfx1250(
    *,
    original_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    read_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_length: torch.Tensor,
    retained_history: KdaRetainedHistory,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
) -> None:
    """Commit accepted packed original_qkv rows to live convolution pages.

    conv_states contains source taps. read_indices and query_start_loc map
    request rows; accepted_length is clamped to each row's actual token count.
    retained_history identifies private live pages and the length to publish.
    Metadata/stamps must be supplied together; stale accepted records suppress
    both convolution updates and length publication. num_heads, head_dim and
    draft_token_num specify fixed geometry. Return None.
    """
    _validate_retained_request(
        retained_history,
        read_indices,
        query_start_loc,
        record_metadata,
        record_stamps,
        heads=num_heads,
        dim=head_dim,
        window=draft_token_num,
        pages=conv_states.shape[0],
        device=conv_states.device,
    )
    batch = read_indices.numel()
    if (
        accepted_length.shape != (batch,)
        or accepted_length.dtype not in (torch.int32, torch.int64)
        or accepted_length.device != conv_states.device
    ):
        raise ValueError("retained KDA accepted lengths must match the requests")
    if (
        head_dim != 128
        or conv_states.shape[1:] != (3 * num_heads * head_dim, 3)
        or original_qkv.ndim != 2
        or original_qkv.shape[1] != 3 * num_heads * head_dim
        or original_qkv.shape[0] > batch * draft_token_num
        or original_qkv.stride(-1) != 1
        or original_qkv.device != conv_states.device
    ):
        raise ValueError(
            "retained KDA commit needs packed pre-convolution QKV and matching taps"
        )
    _retained_commit_gfx1250(
        original_qkv=original_qkv,
        conv_pool=conv_states,
        read_indices=read_indices,
        live_indices=retained_history.live_state_indices,
        query_start_loc=query_start_loc,
        accepted_length=accepted_length,
        retained_length=retained_history.length,
        history_capacity=retained_history.correction.shape[2],
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
    )
