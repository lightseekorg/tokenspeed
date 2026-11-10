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

It also registers the in-tree CuTe DSL RecoverSSM verify, which replaces the
Triton verify on Blackwell for BF16 windows of up to 16 tokens.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._optional import has_optional_package
from tokenspeed_kernel.ops.attention.kda import (
    KdaOutputNorm,
    KdaPrefillResult,
    KdaReplayRecords,
    kda_verify_conv_update,
)
from tokenspeed_kernel.ops.attention.kda._triton.prefill_scan_inputs import (
    prepare_capacity_scan,
)
from tokenspeed_kernel.ops.attention.kda.triton import (
    _DENSE_HALF_SIGNATURES,
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
from tokenspeed_kernel.thirdparty.cutedsl_kda import (
    cutedsl_kda_forward_with_prepared_plan,
    cutedsl_kda_supports_prepared_plan,
)

_SUPPORTED_ARCHES = frozenset({ArchVersion(10, 0), ArchVersion(10, 3)})


def cutedsl_kda_supported() -> bool:
    """Whether the current platform supports the packaged CuteDSL KDA kernel."""
    platform = current_platform()
    return (
        platform.is_nvidia
        and platform.arch_version in _SUPPORTED_ARCHES
        and has_optional_package("tokenspeed_cutedsl_kda")
    )


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
            "state_dtype": frozenset({torch.float32}),
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
        correction_cache: torch.Tensor,
        kg_cache: torch.Tensor,
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
        offsets for windows of at most spec_query_len tokens. correction_cache
        is FP32 [slot, head, position, channel] and kg_cache holds unnormalized
        K and raw gate. record_metadata is INT64 [batch, 3] (slot, generation,
        absolute start) and record_stamps INT64 [slots, capacity, 2]; both may
        be None only for unstamped diagnostics.

        Inputs the CuTe DSL kernel cannot address run the Triton verify
        instead: token rows or state not 16-byte aligned, state tiles that are
        not dense, strided indices or record channels, a strided one-element
        A_log, and positive lower bounds.

        Returns out after writing verify results; the checkpoint is unchanged.
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
            correction_cache,
            kg_cache,
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
                correction_cache=correction_cache,
                kg_cache=kg_cache,
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
            correction_cache=correction_cache,
            kg_cache=kg_cache,
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
            "spec_query_len_max": frozenset({1}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "replay_records": frozenset({True}),
            "split_producers": frozenset({False}),
            "store_states": frozenset({False}),
        },
    )
    def cute_dsl_kda_fused_paged_verify_step(
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
        output_norm: KdaOutputNorm | None,
    ) -> torch.Tensor:
        """Run one-token record verify with the gate and convolution producers fused.

        Same results as the split producers (``f_a_out @ f_b_weight.T`` and
        ``kda_verify_conv_update``) followed by ``kda_recoverssm_verify``:
        outputs, correction and key/gate records, stamps, and the
        pre-convolution rows in ``replay_records.original_qkv``. The recurrent
        and convolution states are read only. Inputs the CuTe DSL kernel
        cannot address run those split producers instead. ``output_norm``
        is applied before the store.

        Returns the verify output as ``[1, tokens, num_heads, head_dim]``, or
        with ``output_norm`` the normed ``[tokens, num_heads * head_dim]``.
        """
        from tokenspeed_kernel.ops.attention.kda._cute_dsl.fused_verify_step import (
            fused_verify_step,
            fused_verify_step_supported,
        )

        if replay_records is None or replay_records.original_qkv is None:
            raise ValueError("fused verify step requires records with original rows")
        if draft_token_num != 1:
            raise ValueError(
                f"fused verify step takes one-token windows, got {draft_token_num}"
            )
        batch = read_indices.shape[0]
        rows = batch * draft_token_num
        mixed_qkv = mixed_qkv[:rows]
        f_a_out = f_a_out[:rows]
        beta_logits = beta_logits[:rows]
        query_start_loc = replay_records.query_start_loc
        if not fused_verify_step_supported(
            mixed_qkv,
            conv_weights,
            conv_states,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            state_pool,
            num_heads,
            head_dim,
            lower_bound,
        ):
            g_raw = torch.nn.functional.linear(f_a_out, f_b_weight)
            conv_qkv = kda_verify_conv_update(
                mixed_qkv,
                conv_weights,
                conv_states,
                read_indices,
                num_heads=num_heads,
                head_dim=head_dim,
                draft_token_num=draft_token_num,
                recurrent_layout="v_major",
            )
            return triton_nvidia_kda_fused_paged_verify_split(
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
                replay_records=replay_records,
                output_norm=output_norm,
                g_raw=g_raw,
                conv_qkv=conv_qkv,
            )
        out = torch.empty(
            1, rows, num_heads, head_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )
        fused_verify_step(
            mixed_qkv,
            conv_weights,
            conv_states,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            lower_bound,
            state_pool[:, :num_heads],
            replay_records.correction,
            replay_records.key_gate,
            replay_records.original_qkv,
            query_start_loc[: batch + 1],
            read_indices,
            num_heads,
            head_dim,
            out.view(rows, num_heads * head_dim),
            replay_records.metadata,
            replay_records.stamps,
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
            "batch_size_max": frozenset({16}),
            "spec_query_len_max": frozenset({16}),
            "paged_state": frozenset({True}),
            "recurrent_layout": frozenset({"v_major"}),
            "replay_records": frozenset({True}),
            "split_producers": frozenset({False}),
            "store_states": frozenset({False}),
        },
    )
    def cute_dsl_kda_fused_paged_verify_window(
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
        output_norm: KdaOutputNorm | None,
    ) -> torch.Tensor:
        """Run multi-token record verify with the gate and convolution producers fused.

        Same results as the split producers (``f_a_out @ f_b_weight.T`` and
        ``kda_verify_conv_update``) followed by ``kda_recoverssm_verify`` for
        windows of 2 to 16 tokens: outputs, correction and key/gate records,
        stamps, and the pre-convolution rows in ``replay_records.original_qkv``.
        The recurrent and convolution states are read only. Inputs the CuTe
        DSL kernel cannot address run those split producers instead.
        ``output_norm`` is applied before the store.

        Returns the verify output as ``[1, tokens, num_heads, head_dim]``, or
        with ``output_norm`` the normed ``[tokens, num_heads * head_dim]``.
        """
        from tokenspeed_kernel.ops.attention.kda._cute_dsl.fused_verify_window import (
            fused_verify_window,
            fused_verify_window_supported,
        )

        if replay_records is None or replay_records.original_qkv is None:
            raise ValueError("fused verify window requires records with original rows")
        batch = read_indices.shape[0]
        rows = batch * draft_token_num
        mixed_qkv = mixed_qkv[:rows]
        f_a_out = f_a_out[:rows]
        beta_logits = beta_logits[:rows]
        query_start_loc = replay_records.query_start_loc[: batch + 1]
        if not fused_verify_window_supported(
            mixed_qkv,
            conv_weights,
            conv_states,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            state_pool,
            query_start_loc,
            read_indices,
            num_heads,
            head_dim,
            draft_token_num,
            lower_bound,
        ):
            g_raw = torch.nn.functional.linear(f_a_out, f_b_weight)
            conv_qkv = kda_verify_conv_update(
                mixed_qkv,
                conv_weights,
                conv_states,
                read_indices,
                num_heads=num_heads,
                head_dim=head_dim,
                draft_token_num=draft_token_num,
                recurrent_layout="v_major",
            )
            return triton_nvidia_kda_fused_paged_verify_split(
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
                replay_records=replay_records,
                output_norm=output_norm,
                g_raw=g_raw,
                conv_qkv=conv_qkv,
            )
        out = torch.empty(
            1, rows, num_heads, head_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )
        fused_verify_window(
            mixed_qkv,
            conv_weights,
            conv_states,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            lower_bound,
            state_pool[:, :num_heads],
            replay_records.correction,
            replay_records.key_gate,
            replay_records.original_qkv,
            query_start_loc,
            read_indices,
            num_heads,
            head_dim,
            draft_token_num,
            out.view(rows, num_heads * head_dim),
            replay_records.metadata,
            replay_records.stamps,
            output_norm,
        )
        return out if output_norm is None else out.view(rows, num_heads * head_dim)

    @register_kernel(
        "attention",
        "kda_replay_commit",
        name="cute_dsl_kda_batched_replay_commit",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        # Above the Triton commit, which longer windows and BF16 state keep.
        priority=Priority.SPECIALIZED + 1,
        traits={
            "batched_layers": frozenset({True}),
            "flat_state": frozenset({True}),
            "head_dim": frozenset({128}),
            "recurrent_layout": frozenset({"v_major"}),
            # The Triton commit's CUDA-core windows, which this kernel matches bit for bit.
            "spec_query_len_max": frozenset({15}),
            "state_dtype": frozenset({torch.float32}),
        },
    )
    def cute_dsl_kda_batched_replay_commit(
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
        state_dtype: torch.dtype,
    ) -> None:
        """Commit every KDA layer's accepted tokens, the states in one CuTe DSL launch.

        Same contract and results as ``batched_recoverssm_commit``: the
        recurrent states fold the accepted records bit for bit as its
        CUDA-core path does (``commit_states``), and the convolution
        histories take the same window commit. Inputs the CuTe DSL kernel
        does not take (the unbounded gate, other record or state dtypes,
        longer windows) run ``batched_recoverssm_commit`` instead.
        """
        from tokenspeed_kernel.ops.attention.kda._cute_dsl.commit_state import (
            commit_states,
            commit_states_supported,
        )
        from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
            batched_recoverssm_commit,
            check_batched_commit,
            commit_kda_conv_windows,
        )

        arguments = dict(
            group_indices=group_indices,
            read_indices=read_indices,
            write_indices=write_indices,
            accepted_length=accepted_length,
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
            state_dtype=state_dtype,
        )
        if not commit_states_supported(
            activation_dtype=activation_dtype,
            state_dtype=state_dtype,
            draft_token_num=draft_token_num,
            head_dim=head_dim,
            lower_bound=lower_bound,
            record_metadata=record_metadata,
        ):
            batched_recoverssm_commit(descriptors, **arguments)
            return
        if not check_batched_commit(
            descriptors,
            group_indices,
            read_indices,
            write_indices,
            accepted_length,
            draft_token_num=draft_token_num,
            head_dim=head_dim,
            conv_width=conv_width,
            record_metadata=record_metadata,
            record_stamps=record_stamps,
            state_dtype=state_dtype,
        ):
            return
        commit_states(
            descriptors,
            group_indices,
            read_indices,
            write_indices,
            accepted_length,
            draft_token_num=draft_token_num,
            num_heads=num_heads,
            correction_stride=correction_stride,
            key_gate_stride=key_gate_stride,
            state_stride=state_stride,
            lower_bound=lower_bound,
            record_metadata=record_metadata,
            record_stamps=record_stamps,
        )
        commit_kda_conv_windows(
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
        )
