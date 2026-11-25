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
    KdaOutputNorm,
    KdaPrefillResult,
    KdaRetainedHistory,
    KdaRetainedRecords,
    apply_kda_output_norm,
)
from tokenspeed_kernel.ops.tuning import is_warming_up
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        launch_gluon_kda_retained_verify_gfx950 as _retained_verify_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.prefill import (
        launch_gluon_kda_paged_prefill_gfx950 as _kda_prefill_impl,
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
        output, final_state = _kda_prefill_impl(**kwargs, warm_variants=is_warming_up())
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
        output, final_state = _kda_prefill_gfx1250_impl(**kwargs)
        return KdaPrefillResult(out=output, final_state=final_state)

    def _gluon_kda_fused_paged_verify_vmajor_gfx950(
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
    ) -> torch.Tensor:
        """Fuse gate, convolution and verify; a supplied raw gate takes precedence."""
        result = gluon_kda_retained_verify_gfx950(
            mixed_qkv=mixed_qkv,
            original_qkv=retained_records.original_qkv,
            conv_weights=conv_weights,
            conv_states=conv_states,
            raw_g=g_raw,
            f_a_out=f_a_out if g_raw is None else None,
            f_b_weight=f_b_weight if g_raw is None else None,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            state_pool=state_pool[:, :num_heads],
            read_indices=read_indices,
            query_start_loc=retained_records.query_start_loc,
            retained_history=retained_records.history,
            record_metadata=retained_records.metadata,
            record_stamps=retained_records.stamps,
            num_heads=num_heads,
            head_dim=head_dim,
            draft_token_num=draft_token_num,
            lower_bound=lower_bound,
        )
        return apply_kda_output_norm(result, output_norm)

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
            "retained_history": frozenset({True}),
            "split_producers": frozenset({False}),
        },
    )
    def gluon_kda_fused_paged_verify_nostore_vmajor_gfx950(*args, **kwargs):
        return _gluon_kda_fused_paged_verify_vmajor_gfx950(*args, **kwargs)

    def _gluon_kda_fused_paged_verify_vmajor_gfx1250(
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
    ) -> torch.Tensor:
        """Fuse gate, convolution and verify; a supplied raw gate takes precedence."""
        result = gluon_kda_retained_verify_gfx1250(
            mixed_qkv=mixed_qkv,
            original_qkv=retained_records.original_qkv,
            conv_weights=conv_weights,
            conv_states=conv_states,
            raw_g=g_raw,
            f_a_out=f_a_out if g_raw is None else None,
            f_b_weight=f_b_weight if g_raw is None else None,
            beta_logits=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            state_pool=state_pool[:, :num_heads],
            read_indices=read_indices,
            query_start_loc=retained_records.query_start_loc,
            retained_history=retained_records.history,
            record_metadata=retained_records.metadata,
            record_stamps=retained_records.stamps,
            num_heads=num_heads,
            head_dim=head_dim,
            draft_token_num=draft_token_num,
            lower_bound=lower_bound,
        )
        return apply_kda_output_norm(result, output_norm)

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
            "retained_history": frozenset({True}),
            "split_producers": frozenset({False}),
        },
    )
    def gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250(*args, **kwargs):
        return _gluon_kda_fused_paged_verify_vmajor_gfx1250(*args, **kwargs)


def _validate_retained_request(
    history, reads, starts, metadata, stamps, *, heads, dim, window, device
):
    """Check current request geometry; immutable storage was checked at construction."""
    batch = reads.numel()
    if (
        history.correction.shape[1] != heads
        or history.correction.shape[-1] != dim
        or history.key.shape[-1] != dim
    ):
        raise ValueError("retained history must match checkpoint head geometry")
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
    """Verify packed pre-convolution QKV using request-owned accepted records.

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
        or state_pool.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError(
            "retained KDA checkpoint must be FP32 or BF16 [page, head, V, K]"
        )
    _validate_retained_request(
        retained_history,
        read_indices,
        query_start_loc,
        record_metadata,
        record_stamps,
        heads=num_heads,
        dim=head_dim,
        window=draft_token_num,
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
        record_slots=retained_history.record_slots,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
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
    """Verify packed pre-convolution QKV using request-owned accepted records.

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
        or state_pool.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError(
            "retained KDA checkpoint must be FP32 or BF16 [page, head, V, K]"
        )
    _validate_retained_request(
        retained_history,
        read_indices,
        query_start_loc,
        record_metadata,
        record_stamps,
        heads=num_heads,
        dim=head_dim,
        window=draft_token_num,
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
        record_slots=retained_history.record_slots,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
    )


if current_platform().is_amd:

    @register_kernel(
        "attention",
        "kda_retained_commit",
        name="gluon_kda_batched_retained_commit",
        solution="gluon",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}), min_arch_version=ArchVersion(9, 5)
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "batched_layers": frozenset({True}),
            "retained_history": frozenset({True}),
            "head_dim": frozenset({128}),
            "recurrent_layout": frozenset({"v_major"}),
        },
    )
    def gluon_kda_batched_retained_commit(**kwargs) -> None:
        """Commit all AMD retained convolution windows, then publish lengths."""
        from tokenspeed_kernel.ops.attention.kda._retained import batched_commit

        batched_commit(**kwargs)
