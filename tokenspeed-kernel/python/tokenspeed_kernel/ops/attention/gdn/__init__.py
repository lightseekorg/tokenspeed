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

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


# One UE8M0 scale per 32 consecutive head_dim elements (MXFP8).
MXFP8_ATTENTION_BLOCK_SCALE = MXFP8_BLOCK_SCALE


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


def _mxfp8_attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{
            role: tensor_format(
                "mxfp8", tensor.dtype, scale=MXFP8_ATTENTION_BLOCK_SCALE
            )
            for role, tensor in roles.items()
        }
    )


def _blockscaled_signature_and_scales(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
):
    """Pick dense vs MXFP8 signature and build the scale kwargs splat.

    q_scale selects the block-scaled path; k_scale/v_scale must accompany it.
    Returns (signature, scale_kwargs) for the paged-KV-cache entry points.
    """
    if q_scale is not None:
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 attention requires q_scale, k_scale, and v_scale together"
        signature = _mxfp8_attention_format_signature(
            q=q, k_cache=k_cache, v_cache=v_cache
        )
    else:
        signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    return signature, dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)


LSE_LN = math.log2(math.e)


# ===-----------------------------------------------------------------------===#
# GDN Kernels
# ===-----------------------------------------------------------------------===#


class GdnCheckpointLayout(str, Enum):
    """Backend-native checkpoint layout returned by GDN chunk prefill."""

    NONE = "none"
    FLA = "fla"
    FLASHINFER = "flashinfer"


@dataclass(frozen=True)
class GdnChunkPrefillResult:
    """Structured result for GDN chunk prefill.

    Args:
        out: GDN output tensor.
        final_state: Final recurrent state, when requested.
        h: Optional backend-native intermediate recurrent checkpoints.
        h_cu_starts: Optional cumulative checkpoint starts for FlashInfer layout.
        h_layout: Layout of ``h``.
    """

    out: torch.Tensor
    final_state: torch.Tensor | None
    h: torch.Tensor | None = None
    h_cu_starts: torch.Tensor | None = None
    h_layout: GdnCheckpointLayout = GdnCheckpointLayout.NONE


@dataclass(frozen=True)
class GdnStateCheckpoints:
    """Recurrent states a chunk prefill stores from inside its scan.

    Sequence ``i`` stores the state after each whole multiple of
    ``every_n_tokens`` of its own tokens (counted from its ``cu_seqlens``
    start): ``len_i // every_n_tokens`` states, the ``j``-th (0-based) the
    state after ``(j + 1) * every_n_tokens`` tokens, at row
    ``cu_starts[i] + j`` of ``states``.

    Args:
        every_n_tokens: Interval in tokens; a positive multiple of
            ``GDN_STATE_CHECKPOINT_GRANULARITY``, the scan's chunk size.
        states: Caller-allocated K-last float32 buffer ``[total, num_v_heads,
            head_v_dim, head_dim]`` with ``total = sum(len_i //
            every_n_tokens)``; the kernel overwrites it.
        cu_starts: ``[num_seqs + 1]`` int32 cumulative per-sequence counts on
            the device of ``states``.
    """

    every_n_tokens: int
    states: torch.Tensor
    cu_starts: torch.Tensor


# Chunk-prefill kernels whose launch geometry depends only on the sequence count
# and which read the sequence bounds on device.
_DEVICE_BOUNDS_CHUNK_PREFILL = frozenset({"flashinfer_gdn_chunk_prefill"})


def _chunk_prefill_traits(
    head_dim: int,
    value_head_dim: int,
    num_q_heads: int,
    num_v_heads: int,
    output_h: bool,
    qk_l2norm: bool,
    state_checkpoints: bool,
) -> dict[str, int | bool]:
    return {
        "head_dim": head_dim,
        "value_head_dim": value_head_dim,
        "num_v_gte_num_q": num_v_heads >= num_q_heads,
        "output_h": output_h,
        "qk_l2norm": qk_l2norm,
        "state_checkpoints": state_checkpoints,
    }


def gdn_chunk_prefill_capturable(
    dtype: torch.dtype,
    *,
    head_dim: int,
    value_head_dim: int,
    num_q_heads: int,
    num_v_heads: int,
    qk_l2norm: bool,
) -> bool:
    """Whether a CUDA graph can capture ``gdn_chunk_prefill`` at a fixed sequence count.

    The kernel selected for this geometry must size its launch from the number
    of sequences alone and read ``cu_seqlens`` on device, so a graph replayed
    with rewritten bounds of the same sequence count stays valid.

    Args:
        dtype: Q/K/V dtype.
        head_dim: Query/key head dimension.
        value_head_dim: Value head dimension.
        num_q_heads: Query (and key) head count.
        num_v_heads: Value head count.
        qk_l2norm: Whether the scan L2-normalizes Q/K.

    Returns:
        ``True`` when the selected kernel can be captured that way.
    """
    probe = torch.empty(0, dtype=dtype, device="meta")
    try:
        kernel = select_kernel(
            "attention",
            "gdn_chunk_prefill",
            _attention_format_signature(q=probe, k=probe, v=probe),
            traits=_chunk_prefill_traits(
                head_dim,
                value_head_dim,
                num_q_heads,
                num_v_heads,
                False,
                qk_l2norm,
                False,
            ),
        )
    except NoKernelFoundError:
        return False
    return kernel.name in _DEVICE_BOUNDS_CHUNK_PREFILL


# State checkpoints land on scan chunk boundaries: every interval is a
# multiple of this many tokens.
GDN_STATE_CHECKPOINT_GRANULARITY = 64


def gdn_chunk_prefill_stores_state_checkpoints(
    dtype: torch.dtype,
    *,
    head_dim: int,
    value_head_dim: int,
    num_q_heads: int,
    num_v_heads: int,
    qk_l2norm: bool,
    interval_granularity: int,
) -> bool:
    """Whether a ``gdn_chunk_prefill`` kernel stores checkpoints at these intervals.

    That is, whether the kernel for this geometry takes ``state_checkpoints``
    (stores states from inside its scan) at intervals that are multiples of
    ``interval_granularity``.

    Args:
        dtype: Q/K/V dtype.
        head_dim: Query/key head dimension.
        value_head_dim: Value head dimension.
        num_q_heads: Query (and key) head count.
        num_v_heads: Value head count.
        qk_l2norm: Whether the scan L2-normalizes Q/K.
        interval_granularity: Every interval the caller asks for is a multiple
            of it; the scan stores only at multiples of
            ``GDN_STATE_CHECKPOINT_GRANULARITY``.

    Returns:
        ``True`` when kernel selection finds one and the intervals fit it.
    """
    if (
        interval_granularity <= 0
        or interval_granularity % GDN_STATE_CHECKPOINT_GRANULARITY
    ):
        return False
    probe = torch.empty(0, dtype=dtype, device="meta")
    try:
        select_kernel(
            "attention",
            "gdn_chunk_prefill",
            _attention_format_signature(q=probe, k=probe, v=probe),
            traits=_chunk_prefill_traits(
                head_dim,
                value_head_dim,
                num_q_heads,
                num_v_heads,
                False,
                qk_l2norm,
                True,
            ),
        )
    except NoKernelFoundError:
        return False
    return True


def gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
    state_checkpoints: GdnStateCheckpoints | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> GdnChunkPrefillResult:
    """Run Gated Delta Net chunked prefill through kernel selection.

    Args:
        q: Query tensor shaped ``[1, total_tokens, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[1, total_tokens, num_k_heads, head_dim]``.
        v: Value tensor shaped ``[1, total_tokens, num_v_heads, head_v_dim]``.
        g: Log-space forget gate shaped ``[1, total_tokens, num_v_heads]``.
        beta: Beta gate shaped ``[1, total_tokens, num_v_heads]``.
        scale: Attention scale. ``None`` lets the implementation use its default.
        initial_state: Recurrent state, K-last: ``[batch, num_v_heads,
            head_v_dim, head_dim]``. This matches flashinfer's native GDN
            decode/MTP layout (and the runtime's SSM state pool); backends
            whose own math is FLA-native (e.g. Triton) transpose internally.
        cu_seqlens: Cumulative sequence lengths for variable-length prefill.
        qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        output_final_state: Whether to return the final recurrent state.
        output_h: Whether to return intermediate recurrent checkpoints in the
            selected backend's native layout.
        state_checkpoints: Optional caller-owned buffer the scan stores
            states into at a fixed interval (see ``GdnStateCheckpoints``);
            exclusive with ``output_h``.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        ``GdnChunkPrefillResult`` with output, final state (K-last, same
        layout as ``initial_state``), and optional backend-native recurrent
        checkpoints (also K-last).
    """
    head_dim = q.shape[-1]
    value_head_dim = v.shape[-1]
    num_q_heads = q.shape[-2]
    num_v_heads = v.shape[-2]
    if output_h and state_checkpoints is not None:
        raise ValueError(
            "gdn_chunk_prefill: output_h and state_checkpoints are exclusive"
        )
    traits = _chunk_prefill_traits(
        head_dim,
        value_head_dim,
        num_q_heads,
        num_v_heads,
        output_h,
        qk_l2norm,
        state_checkpoints is not None,
    )
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "gdn_chunk_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cu_seqlens.shape[0] - 1,
        "total_tokens": q.shape[1] if q.dim() == 4 else q.shape[0],
        "num_q_heads": num_q_heads,
        "num_v_heads": num_v_heads,
        "head_dim": head_dim,
        "head_v_dim": value_head_dim,
    }
    ShapeCapture.get().record(
        "attention",
        "gdn_chunk_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "gdn_chunk_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=qk_l2norm,
            output_final_state=output_final_state,
            output_h=output_h,
            state_checkpoints=state_checkpoints,
        )


def _qkv_rows_strided(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether q/k/v rows lie farther apart than one packed q|k|v row.

    Contiguous tensors, and the sections of one dense ``[..., q | k | v]``
    row split apart (as the dense decode path hands them over), are not
    strided; views into a wider projection row are.
    """
    width = sum(t.shape[-2] * t.shape[-1] for t in (q, k, v))
    for t in (q, k, v):
        if t.is_contiguous():
            continue
        if t.stride(-1) != 1 or t.stride(-2) != t.shape[-1]:
            return True
        row = width
        for dim in reversed(range(t.ndim - 2)):
            if t.shape[dim] != 1 and t.stride(dim) != row:
                return True
            row *= t.shape[dim]
    return False


def gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run one single-token (T=1) GDN decode step through kernel selection.

    Args:
        q: Query tensor shaped ``[B, 1, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[B, 1, num_q_heads, head_dim]``.
        v: Value tensor shaped ``[B, 1, num_v_heads, head_v_dim]``.
            q/k/v may be sections of one packed ``q | k | v`` row. Rows
            spaced wider (views into a projection row; heads stay
            contiguous) only reach kernels declaring the ``strided_qkv``
            trait (see ``gdn_decode_supports``).
        A_log: Floating-point log decay parameter shaped ``[num_v_heads]``.
            Backends that require FP32 normalize it internally.
        a: Input-dependent decay shaped ``[B, 1, num_v_heads]``.
        dt_bias: Floating-point decay bias shaped ``[num_v_heads]``. Backends
            that require FP32 normalize it internally.
        b: Update-gate (beta) input shaped ``[B, 1, num_v_heads]``.
        initial_state: SSM state pool, K-last ``[pool_size, num_v_heads,
            head_v_dim, head_dim]`` (matches the runtime's SSM state pool).
        initial_state_indices: Per-batch read row, shaped ``[B]``. ``-1``
            marks CUDA-graph padding; handled internally, no caller clamp
            needed.
        scale: Attention scale. ``None`` lets the implementation use its default.
        output_state_indices: Per-batch write row, shaped ``[B]``. ``None``
            writes back to ``initial_state_indices`` (the common, non-flat
            pool case); pass distinct rows for flat dual-index state paging.
        use_qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Decode output shaped ``[B, 1, num_v_heads, head_v_dim]`` (q.dtype).
    """
    head_dim = q.shape[-1]
    signature = _attention_format_signature(q=q, k=k, v=v)
    traits = {"head_dim": head_dim}
    if _qkv_rows_strided(q, k, v):
        traits["strided_qkv"] = True
    kernel = select_kernel(
        "attention",
        "gdn_decode_step",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_decode_step",
        q.dtype,
        kernel_name=kernel.name,
        batch_size=q.shape[0],
        num_v_heads=v.shape[-2],
        head_dim=head_dim,
        head_v_dim=v.shape[-1],
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            output_state_indices=output_state_indices,
            use_qk_l2norm=use_qk_l2norm,
        )


# ReplaySSM draft trees of these sizes verify in the chunked form, which keeps no node states.
GDN_TREE_VERIFY_CHUNKED_MIN_NODES = 8
GDN_TREE_VERIFY_CHUNKED_MAX_NODES = 16


def gdn_tree_verify_needs_node_states(num_nodes: int) -> bool:
    """Whether a ReplaySSM draft-tree ``gdn_decode_mtp`` of ``num_nodes`` nodes takes
    an ``intermediate_states_buffer``: trees outside
    ``GDN_TREE_VERIFY_CHUNKED_MIN_NODES`` to ``GDN_TREE_VERIFY_CHUNKED_MAX_NODES``
    nodes verify step by step and reload their branch points' states from it."""
    return not (
        GDN_TREE_VERIFY_CHUNKED_MIN_NODES
        <= num_nodes
        <= GDN_TREE_VERIFY_CHUNKED_MAX_NODES
    )


class GdnChainVerifyForm(str, Enum):
    """How a ReplaySSM chain verify runs (``gdn_chain_verify_form``)."""

    # Step by step through ``gdn_decode_mtp``.
    STEP = "step"
    # Chunked as a one-path tree through ``gdn_decode_mtp``.
    TREE = "tree"
    # Chunked through ``gdn_chain_verify`` (16-bit state, tensor-core products).
    CHAIN = "chain"


# A ReplaySSM chain verifies faster in the tree form from these sizes on, measured on Blackwell.
GDN_CHAIN_VERIFY_CHUNKED_MIN_TOKENS = 13
GDN_CHAIN_VERIFY_CHUNKED_MIN_STATES = 24

# State dtypes whose 2- to 16-token chains take the chain form (measured on Blackwell).
GDN_CHAIN_VERIFY_CHAIN_STATE_DTYPES = frozenset({torch.bfloat16})


def gdn_chain_verify_supported(
    head_k_dim: int, head_v_dim: int, state_dtype: torch.dtype
) -> bool:
    """Whether ``gdn_chain_verify`` takes 16-bit state and equal power-of-two dims from 64."""
    return (
        state_dtype in (torch.bfloat16, torch.float16)
        and head_k_dim == head_v_dim
        and head_k_dim >= 64
        and head_k_dim & (head_k_dim - 1) == 0
    )


def gdn_chain_verify_form(
    num_tokens: int, num_states: int, state_dtype: torch.dtype, *, chain_form: bool
) -> GdnChainVerifyForm:
    """The fastest form of a chain of ``num_tokens`` over ``num_states`` states."""
    if (
        not current_platform().is_blackwell
        or num_tokens > GDN_TREE_VERIFY_CHUNKED_MAX_NODES
    ):
        return GdnChainVerifyForm.STEP
    if (
        chain_form
        and num_tokens >= 2
        and state_dtype in GDN_CHAIN_VERIFY_CHAIN_STATE_DTYPES
    ):
        return GdnChainVerifyForm.CHAIN
    if (
        num_tokens >= GDN_CHAIN_VERIFY_CHUNKED_MIN_TOKENS
        and num_states >= GDN_CHAIN_VERIFY_CHUNKED_MIN_STATES
    ):
        return GdnChainVerifyForm.TREE
    return GdnChainVerifyForm.STEP


def gdn_decode_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    disable_state_update: bool = True,
    use_qk_l2norm: bool = True,
    intermediate_states_buffer: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    tree_ancestors: torch.Tensor | None,
    replay_payload: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run one multi-token (T>1) GDN MTP verify step through kernel selection.

    Args:
        q: Query tensor shaped ``[B, T, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[B, T, num_q_heads, head_dim]``.
        v: Value tensor shaped ``[B, T, num_v_heads, head_v_dim]``.
        A_log: Floating-point log decay parameter shaped ``[num_v_heads]``.
            Backends that require FP32 normalize it internally.
        a: Input-dependent decay shaped ``[B, T, num_v_heads]``.
        dt_bias: Floating-point decay bias shaped ``[num_v_heads]``. Backends
            that require FP32 normalize it internally.
        b: Update-gate (beta) input shaped ``[B, T, num_v_heads]``.
        initial_state: SSM state pool, K-last ``[pool_size, num_v_heads,
            head_v_dim, head_dim]`` (matches the runtime's SSM state pool).
        initial_state_indices: Per-batch read row, shaped ``[B]``. When
            ``output_state_indices`` is not provided and
            ``disable_state_update=False``, the final state is written back to
            that same row. Padding handling is solution and state-dtype
            specific: the portable Triton and FlashInfer FP32 paths suppress
            state reads and writes for negative rows, while FlashInfer's BF16
            fast path redirects them to row 0 and requires the caller to
            reserve that row.
        scale: Attention scale. ``None`` lets the implementation use its default.
        disable_state_update: When True (default), never write back to
            ``initial_state_indices``.
        use_qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        intermediate_states_buffer: Optional batch-scoped ``[B, T,
            num_v_heads, head_v_dim, head_dim]`` (K-last, same dtype as
            ``initial_state``) buffer that receives every step's post-update
            state at ``buffer[i_n, step]`` (for a ReplaySSM tree, only the
            branch points' states).
        output_state_indices: Optional per-token state-pool destinations shaped
            ``[B, T]`` with dtype ``torch.int32``. When provided, each
            post-update state ``h_{t+1}`` is written directly to
            ``initial_state[output_state_indices[i, t]]``. Negative entries
            are safe only when the selected solution skips the corresponding
            negative initial-state row; otherwise entries must be
            non-negative. This is mutually exclusive with
            ``intermediate_states_buffer`` and requires
            ``disable_state_update=False``.
        output_state: Optional K-last pool shaped like ``initial_state`` (any
            row count) that ``output_state_indices`` addresses instead of
            ``initial_state``, so a verify can read the committed state in
            place and stage its per-token states elsewhere. Requires
            ``output_state_indices``; only kernels declaring the
            ``separate_output_state`` trait accept it (see
            ``gdn_decode_supports``).
        tree_ancestors: Optional contiguous int64 ``[B, T]`` draft-tree
            ancestor-or-self bitmask, ``T <= 64``: bit ``s`` of row ``t`` is
            set when step ``s`` is ``t`` or one of its ancestors, and ancestors
            precede their descendants. Step ``t`` continues from its parent's
            state (the initial state for a root) instead of step ``t - 1``.
            With ``output_state_indices`` every node's state goes to the pool;
            without it (ReplaySSM verify) the pool is left untouched, which
            needs ``disable_state_update``, and an ``intermediate_states_buffer``
            is given exactly when ``gdn_tree_verify_needs_node_states(T)``.
            Runs the Triton solution; ``None`` is a chain.
        replay_payload: Optional contiguous ``[B * T, num_q_heads * head_dim +
            num_v_heads * head_v_dim + 2 * num_v_heads]`` ReplaySSM payload, in
            the inputs' dtype, that a chunked ReplaySSM tree verify fills with
            each token's ``[k | v | a | b]`` as read.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Decode output shaped ``[B, T, num_v_heads, head_v_dim]`` (q.dtype).
        Outputs for negative initial-state indices are undefined and must be
        ignored, including on the FlashInfer FP32 path.
    """
    if output_state_indices is not None:
        if output_state_indices.shape != q.shape[:2]:
            raise ValueError(
                "output_state_indices must have shape "
                f"{tuple(q.shape[:2])}, got {tuple(output_state_indices.shape)}"
            )
        if output_state_indices.dtype != torch.int32:
            raise ValueError(
                "output_state_indices must have dtype torch.int32, got "
                f"{output_state_indices.dtype}"
            )
        if intermediate_states_buffer is not None:
            raise ValueError(
                "output_state_indices and intermediate_states_buffer are "
                "mutually exclusive"
            )
        if disable_state_update:
            raise ValueError("output_state_indices requires disable_state_update=False")
    if output_state is not None:
        if output_state_indices is None:
            raise ValueError("output_state requires output_state_indices")
        if (
            output_state.shape[1:] != initial_state.shape[1:]
            or output_state.dtype != initial_state.dtype
        ):
            raise ValueError(
                "output_state must match initial_state's row shape and dtype, got "
                f"{output_state.dtype} {tuple(output_state.shape)} vs "
                f"{initial_state.dtype} {tuple(initial_state.shape)}"
            )

    if tree_ancestors is not None:
        if (
            tree_ancestors.shape != q.shape[:2]
            or tree_ancestors.dtype != torch.int64
            or not tree_ancestors.is_contiguous()
            or q.shape[1] > 64
        ):
            raise ValueError(
                f"tree_ancestors must be contiguous int64 {tuple(q.shape[:2])} with "
                f"T <= 64, got {tree_ancestors.dtype} {tuple(tree_ancestors.shape)} "
                f"strides {tree_ancestors.stride()}"
            )
        if output_state_indices is None:
            if not disable_state_update:
                raise ValueError(
                    "a draft tree without output_state_indices leaves the pool "
                    "untouched: disable_state_update must be True"
                )
            if gdn_tree_verify_needs_node_states(q.shape[1]) != (
                intermediate_states_buffer is not None
            ):
                raise ValueError(
                    f"a ReplaySSM tree of {q.shape[1]} nodes takes an "
                    "intermediate_states_buffer exactly when "
                    "gdn_tree_verify_needs_node_states"
                )
        if solution not in (None, "triton"):
            raise ValueError(
                f"draft-tree GDN verify runs the Triton solution, got {solution}"
            )
        solution = "triton"

    head_dim = q.shape[-1]
    signature = _attention_format_signature(q=q, k=k, v=v)
    traits = {"head_dim": head_dim}
    if output_state is not None:
        traits["separate_output_state"] = True
    if _qkv_rows_strided(q, k, v):
        traits["strided_qkv"] = True
    kernel = select_kernel(
        "attention",
        "gdn_decode_mtp",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_decode_mtp",
        q.dtype,
        kernel_name=kernel.name,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        num_v_heads=v.shape[-2],
        head_dim=head_dim,
        head_v_dim=v.shape[-1],
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            disable_state_update=disable_state_update,
            use_qk_l2norm=use_qk_l2norm,
            intermediate_states_buffer=intermediate_states_buffer,
            output_state_indices=output_state_indices,
            output_state=output_state,
            tree_ancestors=tree_ancestors,
            replay_payload=replay_payload,
        )


def gdn_chain_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Read-only chain verify of whole windows from the committed states, in the chunked form.

    Args:
        q: Query ``[B, T, num_q_heads, head_dim]``, ``T <= 16``, request-major rows.
        k: Key tensor shaped like ``q``.
        v: Value tensor shaped ``[B, T, num_v_heads, head_v_dim]``, dense heads.
        A_log: Log decay parameter shaped ``[num_v_heads]``.
        a: Input-dependent decay shaped ``[B, T, num_v_heads]``, unit head stride.
        dt_bias: Decay bias shaped ``[num_v_heads]``.
        b: Update-gate (beta) input shaped like ``a``.
        initial_state: K-last 16-bit pool ``[pages, num_v_heads, head_v_dim, head_dim]``.
        initial_state_indices: Per-request page ``[B]``; negative marks padding (zero outputs).
        scale: Attention scale; ``None`` uses ``head_dim ** -0.5``.
        use_qk_l2norm: Whether to L2-normalize q and k in the kernel.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Verify output shaped ``[B, T, num_v_heads, head_v_dim]`` (q.dtype).
    """
    head_dim = q.shape[-1]
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "gdn_chain_verify",
        signature,
        traits={"head_dim": head_dim},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_chain_verify",
        q.dtype,
        kernel_name=kernel.name,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        num_v_heads=v.shape[-2],
        head_dim=head_dim,
        head_v_dim=v.shape[-1],
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
        )


def validate_replay_commit_setup(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    geometry: tuple[int, int, int, int],
    state_dtype: torch.dtype,
) -> None:
    """Validate fixed GDN/Mamba2 replay storage when the workspace is created.

    Call again after replacing the workspace. Metadata allocation supplies
    contiguous uint64 address, int64 stride, and int32 index tables on the
    payload device; execution checks their current row counts.

    Args:
        payload: Contiguous packed K/V/a/b storage shaped [layers, rows, width].
        parameters: FP32 A_log/dt_bias table shaped [layers, 2, value_heads].
        geometry: (key_heads, value_heads, key_dim, value_dim).
        state_dtype: Element dtype shared by every recurrent-state pool.
    """
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = geometry
    if num_v_heads <= 0 or num_k_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    if head_k_dim <= 0 or head_v_dim <= 0:
        raise ValueError("replay head dimensions must be positive")
    if state_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported replay state dtype: {state_dtype}")

    payload_width = (
        num_k_heads * head_k_dim + num_v_heads * head_v_dim + 2 * num_v_heads
    )
    if (
        payload.ndim != 3
        or payload.shape[-1] != payload_width
        or not payload.is_contiguous()
    ):
        raise ValueError(
            f"replay payload must be contiguous [L, rows, {payload_width}]"
        )
    if parameters.shape != (payload.shape[0], 2, num_v_heads):
        raise ValueError(
            "replay parameters must have shape "
            f"{(payload.shape[0], 2, num_v_heads)}, got {tuple(parameters.shape)}"
        )
    if (
        parameters.dtype != torch.float32
        or parameters.device != payload.device
        or not parameters.is_contiguous()
    ):
        raise ValueError(
            "replay parameters must be contiguous FP32 on the payload device"
        )


def gdn_decode_supports(
    op: str,
    dtype: torch.dtype,
    *,
    head_dim: int,
    trait: str,
    solution: str | None,
) -> bool:
    """Whether the kernel a GDN decode op selects by default also accepts a
    feature, so a caller can use it without changing the selected kernel.

    Args:
        op: ``"gdn_decode_step"`` or ``"gdn_decode_mtp"``.
        dtype: q/k/v dtype.
        head_dim: Key head dimension.
        trait: ``"separate_output_state"`` (an ``output_state`` pool) or
            ``"strided_qkv"`` (q/k/v rows spaced wider than one packed
            ``q | k | v`` row, with contiguous heads; such kernels read a/b by
            stride too).
        solution: The solution the caller forces, or ``None``.

    Returns:
        ``True`` when the default selection and the selection requiring
        ``trait`` pick the same kernel.
    """
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    try:
        default = select_kernel(
            "attention",
            op,
            signature,
            traits={"head_dim": head_dim},
            solution=solution,
        )
        featured = select_kernel(
            "attention",
            op,
            signature,
            traits={"head_dim": head_dim, trait: True},
            solution=solution,
        )
    except NoKernelFoundError:
        return False
    return default.name == featured.name


def validate_replay_commit_args(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    geometry: tuple[int, int, int, int],
    state_dtype: torch.dtype,
) -> bool:
    """Validate the current replay batch against a prepared workspace.

    ``validate_replay_commit_setup`` owns fixed payload/parameter geometry,
    dtype and layout. Keep this signature shared by the GDN and Mamba2 entries;
    metadata allocation supplies contiguous tables on the payload device.

    Returns:
        ``False`` when there is no layer or request to replay.
    """
    num_layers = payload.shape[0]
    batch_size = accepted_length.numel()
    if num_layers == 0 or batch_size == 0:
        return False
    if draft_token_num <= 0:
        raise ValueError("draft_token_num must be positive")
    if payload.shape[1] < batch_size * draft_token_num:
        raise ValueError("replay payload has insufficient token capacity")
    if state_addresses.shape != (num_layers,) or state_row_strides.shape != (
        num_layers,
    ):
        raise ValueError(
            "replay address and stride tables must have one entry per layer"
        )
    expected_indices = (num_layers, batch_size)
    if (
        read_indices.shape != expected_indices
        or write_indices.shape != expected_indices
    ):
        raise ValueError(
            f"replay read/write indices must have shape {expected_indices}"
        )
    if accepted_length.ndim != 1:
        raise ValueError("accepted_length must be one-dimensional")
    return True


def gdn_replay_commit(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    geometry: tuple[int, int, int, int],
    state_dtype: torch.dtype,
    override: str | None = None,
    solution: str | None = None,
) -> None:
    """Replay every GDN layer's accepted prefix in one kernel launch.

    K/V/a/b share one layer-major allocation. Recurrent slabs may remain
    physically disjoint: ``state_addresses`` and ``state_row_strides`` expose
    them as a layer-indexed table to the kernel. Each program decodes its
    layer, request, and value-head coordinates and writes only the final
    accepted state.

    Args:
        payload: Contiguous packed K/V/a/b storage shaped
            ``[L, token_capacity, H*K + HV*V + 2*HV]``. The first ``B*T``
            rows of each layer hold the current request-major verify window.
        parameters: FP32 A_log/dt_bias table shaped ``[L, 2, HV]``.
        state_addresses: uint64 base-address table shaped ``[L]`` for the
            K-last recurrent-state pools.
        state_row_strides: int64 row strides in elements, shaped ``[L]``.
        read_indices: Committed-state pages shaped ``[L, B]``.
        write_indices: Accepted-state destination pages shaped ``[L, B]``.
        accepted_length: Accepted verified-token count per request, shaped ``[B]``.
        draft_token_num: Number of verify positions per request (``T``).
        geometry: ``(num_k_heads, num_v_heads, head_k_dim, head_v_dim)``.
        state_dtype: Element dtype shared by all recurrent-state pools.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
    """
    if not validate_replay_commit_args(
        payload,
        parameters,
        state_addresses=state_addresses,
        state_row_strides=state_row_strides,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        draft_token_num=draft_token_num,
        geometry=geometry,
        state_dtype=state_dtype,
    ):
        return
    num_layers = payload.shape[0]
    batch_size = accepted_length.numel()
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = geometry

    signature = _attention_format_signature(q=payload, k=payload, v=payload)
    kernel = select_kernel(
        "attention",
        "gdn_replay_commit",
        signature,
        traits={"flat_state": True},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_replay_commit",
        payload.dtype,
        kernel_name=kernel.name,
        batch_size=batch_size,
        seq_len=draft_token_num,
        num_layers=num_layers,
        num_v_heads=num_v_heads,
        head_dim=head_k_dim,
        head_v_dim=head_v_dim,
    ):
        kernel(
            payload=payload,
            parameters=parameters,
            state_addresses=state_addresses,
            state_row_strides=state_row_strides,
            read_indices=read_indices,
            write_indices=write_indices,
            accepted_length=accepted_length,
            draft_token_num=draft_token_num,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            state_dtype=state_dtype,
        )


def gdn_replay_commit_supported(
    dtype: torch.dtype = torch.bfloat16,
    *,
    solution: str | None = None,
) -> bool:
    """Whether ReplaySSM can replace per-draft GDN recurrent-state scratch.

    Args:
        dtype: Activation dtype used by the target verify pass.
        solution: Optional registered solution restriction.

    Returns:
        ``True`` when a compatible GDN replay kernel is registered for the
        current platform.
    """
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    try:
        select_kernel(
            "attention",
            "gdn_replay_commit",
            signature,
            traits={"flat_state": True},
            solution=solution,
        )
    except NoKernelFoundError:
        return False
    return True


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.gdn.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.gdn.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.gdn.triton  # noqa: E402,F401

# isort: on

__all__ = [
    "GDN_TREE_VERIFY_CHUNKED_MIN_NODES",
    "GDN_TREE_VERIFY_CHUNKED_MAX_NODES",
    "GdnCheckpointLayout",
    "GdnChunkPrefillResult",
    "GDN_STATE_CHECKPOINT_GRANULARITY",
    "GdnChainVerifyForm",
    "GdnStateCheckpoints",
    "gdn_chain_verify",
    "gdn_chain_verify_form",
    "gdn_chain_verify_supported",
    "gdn_chunk_prefill",
    "gdn_chunk_prefill_capturable",
    "gdn_chunk_prefill_stores_state_checkpoints",
    "gdn_decode_step",
    "gdn_decode_mtp",
    "gdn_decode_supports",
    "gdn_replay_commit",
    "gdn_replay_commit_supported",
    "gdn_tree_verify_needs_node_states",
    "validate_replay_commit_args",
    "validate_replay_commit_setup",
]
