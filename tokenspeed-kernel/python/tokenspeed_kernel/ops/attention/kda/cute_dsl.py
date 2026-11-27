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

"""FlashInfer CuTe KDA prefill and in-tree CuTe verify/commit kernels.

Prefill reads token-strided Q/K/V and native V-major FP32 state. Its device
plan uses live sequence boundaries for eager execution and capacity replay.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from tokenspeed_kernel._optional import has_optional_package
from tokenspeed_kernel.ops.attention.kda import (
    KdaOutputNorm,
    KdaPreconvRing,
    KdaPrefillCapacity,
    KdaPrefillResult,
    KdaRetainedHistory,
    KdaRetainedRecords,
    kda_verify_conv_update,
)
from tokenspeed_kernel.ops.attention.kda.triton import (
    _nvidia_kda_prefill,
    triton_nvidia_kda_fused_paged_verify_split,
)
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
    vllm_triton_kda_recoverssm_verify,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.flashinfer.kda import cutedsl_kda_prefill

_SUPPORTED_ARCHES = frozenset({ArchVersion(10, 0), ArchVersion(10, 3)})


def cutedsl_kda_supported() -> bool:
    """Whether the current platform supports the FlashInfer CuTe KDA kernel."""
    platform = current_platform()
    return (
        platform.is_nvidia
        and platform.arch_version in _SUPPORTED_ARCHES
        and has_optional_package("flashinfer", "cutlass", "quack")
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
    signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
    priority=Priority.SPECIALIZED,
    traits={
        "prefill_capacity": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def cutedsl_kda_nvidia_paged_prefill(
    *, capacity: KdaPrefillCapacity | None, **kwargs
) -> KdaPrefillResult:
    """Use tensor extent and live boundaries after the facade admits capacity."""
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
    """Scan BF16 Q/K/V views with FlashInfer's CuTe kernels.

    Inputs use [B, T, H, 128]; gate is BF16/FP32 and beta contains raw
    logits. State uses FP32 [sequences, H, V, K]. Packed inputs have B=1
    and device sequence boundaries; fixed batches flatten without copying.
    The CPU boundary mirror is retained for the shared facade contract;
    this backend generates its plan from device boundaries without a D2H.
    Returns BF16 output and a separate FP32 final state.
    """
    if not beta_is_logit:
        raise ValueError("cutedsl_kda_chunk_prefill requires raw beta logits")
    if lower_bound is None or lower_bound >= 0:
        raise ValueError("CuteDSL KDA requires a negative safe-gate bound")
    if dt_bias is None:
        raise ValueError("cutedsl_kda_chunk_prefill requires dt_bias")
    batch, tokens, heads, dim = q.shape
    if dim != 128 or v.shape != q.shape or q.dtype != torch.bfloat16:
        raise ValueError("FlashInfer CuTe KDA requires matching BF16 Q/K/V with D=128")
    if cu_seqlens is None:
        cu_seqlens = (
            torch.arange(batch + 1, device=q.device, dtype=torch.int64) * tokens
        )
        q, k, v, g_raw = (
            t.view(1, batch * tokens, heads, dim) for t in (q, k, v, g_raw)
        )
        beta = beta.reshape(1, batch * tokens, heads)
    sequences = cu_seqlens.numel() - 1
    if initial_state is None:
        initial_state = torch.zeros(
            sequences, heads, dim, dim, dtype=torch.float32, device=q.device
        )
    out, final_state = cutedsl_kda_prefill(
        q,
        k,
        v,
        g_raw,
        beta,
        A_log,
        dt_bias,
        initial_state=initial_state.float().contiguous(),
        cu_seqlens=cu_seqlens.to(torch.int64),
        lower_bound=lower_bound,
    )
    return out.view(batch, tokens, heads, dim), final_state


def _check_ring_storage(
    preconv: KdaPreconvRing,
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    f_a_out: torch.Tensor | None,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    g_raw: torch.Tensor | None,
    output_norm: KdaOutputNorm | None,
) -> None:
    """Raise ValueError when the ring shares storage with a verify input.

    Checked before either route runs, as the Triton producer and the AMD
    launchers do: the fused kernel writes the ring while it reads these.
    """
    preconv.check_separate_storage(
        mixed_qkv,
        conv_weights,
        f_a_out,
        f_b_weight,
        beta_logits,
        g_raw,
        None if output_norm is None else output_norm.gate,
    )


def _split_producer_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    f_a_out: torch.Tensor | None,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    g_raw: torch.Tensor | None,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    retained_records: KdaRetainedRecords,
    output_norm: KdaOutputNorm | None,
) -> torch.Tensor:
    """Run the split gate and convolution producers, then the split verify.

    The convolution producer reads and writes the pre-convolution ring, so
    the verify that follows takes the records without it.
    """
    if g_raw is None:
        g_raw = torch.nn.functional.linear(f_a_out, f_b_weight)
    conv_qkv = kda_verify_conv_update(
        mixed_qkv,
        conv_weights,
        read_indices,
        live_indices=retained_records.history.live_state_indices,
        preconv=retained_records.preconv,
        record_slots=retained_records.history.record_slots,
        page_count=state_pool.shape[0],
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        recurrent_layout="v_major",
        query_start_loc=retained_records.query_start_loc,
    )
    return triton_nvidia_kda_fused_paged_verify_split(
        mixed_qkv,
        conv_weights,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        retained_records=replace(retained_records, preconv=None),
        output_norm=output_norm,
        g_raw=g_raw,
        conv_qkv=conv_qkv,
    )


if current_platform().is_nvidia and has_optional_package("cutlass", "quack"):

    @register_kernel(
        "attention",
        "kda_recoverssm_verify",
        name="cute_dsl_kda_recoverssm_verify",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim_align": frozenset({32}),
            "head_dim_max": frozenset({128}),
            "spec_query_len_max": frozenset({16}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "state_dtype": frozenset({torch.float32, torch.bfloat16}),
        },
    )
    def cute_dsl_kda_recoverssm_verify(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_g: torch.Tensor,
        raw_beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        *,
        checkpoint_state: torch.Tensor,
        retained_history: KdaRetainedHistory,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        spec_query_len: int,
        lower_bound: float | None,
        out: torch.Tensor,
        record_metadata: torch.Tensor | None,
        record_stamps: torch.Tensor | None,
    ) -> torch.Tensor:
        """Verify candidates and save current-window recovery records.

        Same contract as ``vllm_triton_kda_recoverssm_verify``: q/k/v and raw_g
        pack tokens as [1, tokens, heads, dim] with dense heads, raw_beta omits
        the final dim, and A_log and dt_bias parameterize the gate. lower_bound
        selects the bounded gate, or None the softplus gate. checkpoint_state
        uses [page, head, value, key]; state_indices selects request pages,
        with pages <= 0 marking padding, and query_start_loc gives packed-token
        offsets for windows of at most spec_query_len tokens. retained_history
        owns FP32 correction, normalized key and multiplicative decay records
        and writes candidates after each request slot's accepted length.
        record_metadata is INT64 [batch, 3] (slot, generation,
        absolute start) and record_stamps INT64 [slots, capacity, 2]; both may
        be None only for unstamped diagnostics.

        Inputs the CuTe DSL kernel cannot address run the Triton verify
        instead: token rows or state not 16-byte aligned, state tiles that are
        not dense, strided indices or record channels, a strided one-element
        A_log, and positive lower bounds.

        Returns out after writing candidates. Dense pages and accepted lengths
        stay unchanged: the round's batched prepare readied the live bases.
        """
        from tokenspeed_kernel.ops.attention.kda._cute_dsl.recoverssm_verify import (
            recoverssm_verify,
            recoverssm_verify_supported,
        )

        if not recoverssm_verify_supported(
            q,
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            retained_history,
            query_start_loc,
            state_indices,
            spec_query_len,
            lower_bound,
            out,
        ):
            return vllm_triton_kda_recoverssm_verify(
                q,
                k,
                v,
                raw_g,
                raw_beta,
                A_log,
                dt_bias,
                checkpoint_state=checkpoint_state,
                retained_history=retained_history,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                spec_query_len=spec_query_len,
                lower_bound=lower_bound,
                out=out,
                record_metadata=record_metadata,
                record_stamps=record_stamps,
            )
        return recoverssm_verify(
            q,
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            lower_bound=lower_bound,
            checkpoint_state=checkpoint_state,
            retained_history=retained_history,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            spec_query_len=spec_query_len,
            out=out,
            record_metadata=record_metadata,
            record_stamps=record_stamps,
        )

    @register_kernel(
        "attention",
        "kda_fused_paged_verify",
        name="cute_dsl_kda_fused_paged_verify_step",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        # Above the window kernel, which also admits one-token windows.
        priority=Priority.SPECIALIZED + 3,
        traits={
            # A bound, so callers that do not state the window never select it.
            "head_dim": frozenset({128}),
            "head_dim": frozenset({128}),
            "spec_query_len_max": frozenset({1}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "retained_history": frozenset({True}),
            "split_producers": frozenset({False}),
        },
    )
    def cute_dsl_kda_fused_paged_verify_step(
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
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
        retained_records: KdaRetainedRecords | None,
        output_norm: KdaOutputNorm | None,
        g_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one-token record verify with the gate and convolution producers fused.

        Same results as the split producers (``f_a_out @ f_b_weight.T`` and
        ``kda_verify_conv_update``) followed by ``kda_recoverssm_verify``:
        outputs, correction and key/gate records, stamps, and the
        pre-convolution rows in the ring of ``retained_records.preconv``,
        whose rows before each request's head give the convolution history.
        The recurrent state is read only. Inputs the CuTe DSL kernel cannot
        address run those split producers instead. ``output_norm`` is
        applied before the store.

        Returns the verify output as ``[1, tokens, num_heads, head_dim]``, or
        with ``output_norm`` the normed ``[tokens, num_heads * head_dim]``.
        """
        from tokenspeed_kernel.ops.attention.kda._cute_dsl.fused_verify_step import (
            fused_verify_step,
            fused_verify_step_supported,
        )

        if retained_records is None or retained_records.preconv is None:
            raise ValueError(
                "fused verify step requires records with a pre-convolution ring"
            )
        if draft_token_num != 1:
            raise ValueError(
                f"fused verify step takes one-token windows, got {draft_token_num}"
            )
        _check_ring_storage(
            retained_records.preconv,
            mixed_qkv,
            conv_weights,
            f_a_out,
            f_b_weight,
            beta_logits,
            g_raw,
            output_norm,
        )
        batch = read_indices.shape[0]
        rows = batch * draft_token_num
        mixed_qkv = mixed_qkv[:rows]
        f_a_out = None if f_a_out is None else f_a_out[:rows]
        beta_logits = beta_logits[:rows]
        query_start_loc = retained_records.query_start_loc
        if g_raw is not None or not fused_verify_step_supported(
            mixed_qkv,
            conv_weights,
            retained_records.preconv,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            state_pool,
            retained_records.history,
            query_start_loc,
            read_indices,
            num_heads,
            head_dim,
            lower_bound,
        ):
            return _split_producer_verify(
                mixed_qkv,
                conv_weights,
                f_a_out,
                f_b_weight,
                beta_logits,
                A_log,
                dt_bias,
                g_raw=g_raw,
                state_pool=state_pool,
                read_indices=read_indices,
                num_heads=num_heads,
                head_dim=head_dim,
                draft_token_num=draft_token_num,
                lower_bound=lower_bound,
                retained_records=retained_records,
                output_norm=output_norm,
            )
        out = torch.empty(
            1, rows, num_heads, head_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )
        fused_verify_step(
            mixed_qkv,
            conv_weights,
            retained_records.preconv,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            lower_bound,
            state_pool[:, :num_heads],
            retained_records.history,
            query_start_loc[: batch + 1],
            read_indices,
            num_heads,
            head_dim,
            out.view(rows, num_heads * head_dim),
            retained_records.metadata,
            retained_records.stamps,
            output_norm,
        )
        return out if output_norm is None else out.view(rows, num_heads * head_dim)

    @register_kernel(
        "attention",
        "kda_fused_paged_verify",
        name="cute_dsl_kda_fused_paged_verify_window",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        # Below the one-token kernel, above the split producers.
        priority=Priority.SPECIALIZED + 2,
        traits={
            # Past 16 requests the split producers' GEMM outruns the per-head
            # CUDA-core projection (GB300 W=8: 0.86x at 16, 1.05x at 32).
            "head_dim": frozenset({128}),
            "head_dim": frozenset({128}),
            "batch_size_max": frozenset({16}),
            "spec_query_len_max": frozenset({16}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "retained_history": frozenset({True}),
            "split_producers": frozenset({False}),
        },
    )
    def cute_dsl_kda_fused_paged_verify_window(
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
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
        retained_records: KdaRetainedRecords | None,
        output_norm: KdaOutputNorm | None,
        g_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run multi-token record verify with the gate and convolution producers fused.

        Same results as the split producers (``f_a_out @ f_b_weight.T`` and
        ``kda_verify_conv_update``) followed by ``kda_recoverssm_verify`` for
        windows of 2 to 16 tokens: outputs, correction and key/gate records,
        stamps, and the pre-convolution rows in the ring of
        ``retained_records.preconv``, whose rows before each request's head
        give the convolution history. The recurrent state is read only.
        Inputs the CuTe DSL kernel cannot address run those split producers
        instead. ``output_norm`` is applied before the store.

        Returns the verify output as ``[1, tokens, num_heads, head_dim]``, or
        with ``output_norm`` the normed ``[tokens, num_heads * head_dim]``.
        """
        from tokenspeed_kernel.ops.attention.kda._cute_dsl.fused_verify_window import (
            fused_verify_window,
            fused_verify_window_supported,
        )

        if retained_records is None or retained_records.preconv is None:
            raise ValueError(
                "fused verify window requires records with a pre-convolution ring"
            )
        _check_ring_storage(
            retained_records.preconv,
            mixed_qkv,
            conv_weights,
            f_a_out,
            f_b_weight,
            beta_logits,
            g_raw,
            output_norm,
        )
        batch = read_indices.shape[0]
        rows = batch * draft_token_num
        mixed_qkv = mixed_qkv[:rows]
        f_a_out = None if f_a_out is None else f_a_out[:rows]
        beta_logits = beta_logits[:rows]
        query_start_loc = retained_records.query_start_loc[: batch + 1]
        if g_raw is not None or not fused_verify_window_supported(
            mixed_qkv,
            conv_weights,
            retained_records.preconv,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            state_pool,
            retained_records.history,
            query_start_loc,
            read_indices,
            num_heads,
            head_dim,
            draft_token_num,
            lower_bound,
        ):
            return _split_producer_verify(
                mixed_qkv,
                conv_weights,
                f_a_out,
                f_b_weight,
                beta_logits,
                A_log,
                dt_bias,
                g_raw=g_raw,
                state_pool=state_pool,
                read_indices=read_indices,
                num_heads=num_heads,
                head_dim=head_dim,
                draft_token_num=draft_token_num,
                lower_bound=lower_bound,
                retained_records=replace(
                    retained_records, query_start_loc=query_start_loc
                ),
                output_norm=output_norm,
            )
        out = torch.empty(
            1, rows, num_heads, head_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )
        fused_verify_window(
            mixed_qkv,
            conv_weights,
            retained_records.preconv,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            lower_bound,
            state_pool[:, :num_heads],
            retained_records.history,
            query_start_loc,
            read_indices,
            num_heads,
            head_dim,
            draft_token_num,
            out.view(rows, num_heads * head_dim),
            retained_records.metadata,
            retained_records.stamps,
            output_norm,
        )
        return out if output_norm is None else out.view(rows, num_heads * head_dim)
