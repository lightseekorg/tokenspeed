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

"""Communication ops for distributed communication.

All ops require explicit group (tuple of ranks) and rank parameters.
Groups are looked up from pg_manager internally via comm_backend.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import torch
import torch.distributed
from tokenspeed_kernel.ops.communication import (
    allgather_dual_rmsnorm,
    allreduce_residual_rmsnorm,
)
from tokenspeed_kernel.ops.communication import (
    prepare_allreduce_fusion as kernel_prepare_allreduce_fusion,
)
from tokenspeed_kernel.ops.communication import (
    reducescatter_residual_rmsnorm,
)

from tokenspeed.runtime.distributed.comm_backend import (
    CommBackend,
    CommWork,
    Group,
    get_global_backend,
)
from tokenspeed.runtime.distributed.comm_backend.projection import (
    ProjectionSpec,
    ProjectionWorkspace,
)
from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (  # noqa: F401
    MAX_ONESHOT_BYTES as COMM_ONESHOT_MAX_BYTES,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.workspace import WorkspacePool


def _get_process_group(group: Group):
    return pg_manager.get_device_process_group(group)


# ---------------------------------------------------------------------------
# Fusion parameters
# ---------------------------------------------------------------------------


class FusionOp(IntEnum):
    """What post-communication fusion to apply."""

    NONE = 0
    # all_reduce + residual_add + RMSNorm
    RESIDUAL_RMS_NORM = 1
    # reduce_scatter + residual_add + RMSNorm
    RS_RESIDUAL_RMS_NORM = 2
    # all_gather + dual RMSNorm (for MLA)
    AG_DUAL_RMS_NORM = 3


@dataclass
class FusionParams:
    """Optional fusion context passed to fused comm_ops functions.

    Not all fields are used by every ``FusionOp``. Only the relevant
    subset is accessed.
    """

    fusion_op: FusionOp = FusionOp.NONE

    # --- For RESIDUAL_RMS_NORM / RS_RESIDUAL_RMS_NORM ---
    residual: torch.Tensor | None = None
    norm_weight: torch.Tensor | None = None
    eps: float = 1e-6

    # --- For AG_DUAL_RMS_NORM ---
    norm_weight_2: torch.Tensor | None = None
    eps_2: float = 1e-6

    # --- For reduce-scatter fusion ---
    add_in: torch.Tensor | None = None
    residual_reduce_scattered: bool = False
    has_partial_norm_out: bool = False

    # --- Shared by RESIDUAL_RMS_NORM / RS_RESIDUAL_RMS_NORM / AG_DUAL_RMS_NORM ---
    max_token_num: int = 0

    # --- For FP8 block quantization ---
    block_quant_fp8: bool = False

    # --- General ---
    total_num_tokens: int = 0
    trigger_completion_at_end: bool = False
    fp32_acc: bool = False
    max_sm_to_use: int | None = None


# ---------------------------------------------------------------------------
# Basic primitives
# ---------------------------------------------------------------------------


def prepare_projection_collectives(
    spec: ProjectionSpec,
    backend: CommBackend | None,
    scratch_pool: "WorkspacePool | None" = None,
) -> ProjectionWorkspace:
    """Collectively allocate bounded projection scratch before graph capture.

    The backend owns optimized selection, padding, layout conversion and
    fallbacks. Callers may share the result across sequential same-spec layers,
    but never across concurrently executing streams or models. Execute with
    the same backend used to prepare the workspace; None selects the global
    backend, as for ordinary collectives. An optional frozen, model-private
    scratch_pool shares generic buffers across serialized projections; borrowed
    intermediates expire when another projection starts using that pool.
    """
    pg_manager.init_process_group(spec.group, backend=None)
    if backend is None:
        backend = get_global_backend()
    return backend.prepare_projection(spec, scratch_pool)


def try_fused_dp_column_linear(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor | None,
    owner_rows: tuple[int, ...],
    out: torch.Tensor,
    workspace: ProjectionWorkspace,
    backend: CommBackend | None,
) -> bool:
    """Attempt a fused DP column-parallel linear operation on locally owned rows.

    Fuses input all-gather, quantization, GEMM, and inverse all-to-all so each
    owner receives all output channels for its own input rows.

    Args:
        tensor: Full input channels for this owner's physical token rows.
        weight: Local output-channel shard of the linear weight.
        weight_scales: Block-FP8 weight scales, or None for unsupported formats.
        owner_rows: Physical counts in subgroup order, including graph padding.
        out: Caller-owned destination for this owner's complete output channels.
        workspace: Collectively prepared scratch for this linear operation.
        backend: Backend used to prepare the workspace; None selects the global
            backend.

    Returns:
        True when the fused operation wrote out. False leaves out untouched so
        the caller can run the ordinary all-gather, GEMM, and all-to-all path.
        Invalid inputs and execution failures remain visible.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.try_fused_dp_column_linear(
        tensor, weight, weight_scales, owner_rows, out, workspace
    )


def try_fused_dp_row_linear(
    tensor: torch.Tensor,
    input_scales: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scales: torch.Tensor | None,
    owner_rows: tuple[int, ...],
    out: torch.Tensor,
    workspace: ProjectionWorkspace,
    backend: CommBackend | None,
) -> bool:
    """Attempt a fused DP row-parallel linear operation on locally owned rows.

    Fuses input all-to-all, quantization, GEMM, and reduce-scatter so each
    owner receives all output channels for its own input rows.

    Args:
        tensor: Full input channels for this owner's physical token rows.
        input_scales: FP32 per-128 scales for E4M3 input, or None for BF16.
        weight: Local input-channel shard of the linear weight.
        weight_scales: Block-FP8 weight scales, or None for unsupported formats.
        owner_rows: Physical counts in subgroup order, including graph padding.
        out: Caller-owned destination for this owner's complete output channels.
        workspace: Collectively prepared scratch for this linear operation.
        backend: Backend used to prepare the workspace; None selects the global
            backend.

    Returns:
        True when the fused operation wrote out. False leaves out untouched so
        the caller can run the ordinary all-to-all, GEMM, and reduce-scatter path.
        Invalid inputs and execution failures remain visible.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.try_fused_dp_row_linear(
        tensor, input_scales, weight, weight_scales, owner_rows, out, workspace
    )


def projection_all_gather(
    tensor: torch.Tensor,
    rows: int,
    quantize: bool,
    workspace: ProjectionWorkspace,
    backend: CommBackend | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gather padded owner rows into borrowed activations and optional FP8 scales.

    rows is the physical per-owner extent. quantize permits fused 1x128 FP8
    quantization with MN-major scales; unsupported paths return ordinary
    activations and None so the Linear can use its usual GEMM.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.projection_all_gather(tensor, rows, quantize, workspace)


def projection_all_to_all(
    tensor: torch.Tensor,
    rows: int,
    inverse: bool,
    quantize: bool,
    out: torch.Tensor | None,
    workspace: ProjectionWorkspace,
    backend: CommBackend | None,
    *,
    input_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Exchange token/channel axes, optionally fusing forward FP8 quantization.

    Forward maps [local_rows,K] to [TP*rows,K/TP] with zero owner padding.
    It requires out=None and returns borrowed output, optionally quantized to
    FP8 with 1x128 MN-major scales. Inverse maps [TP*rows,N/TP] to [rows,N],
    requires quantize=False and writes into caller-owned out; callers may
    retain it across later communication calls. For prequantized forward input,
    input_scales supplies FP32 per-128 scales; both are exchanged unchanged
    with inverse=False and quantize=False.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.projection_all_to_all(
        tensor, rows, inverse, quantize, out, workspace, input_scales=input_scales
    )


def acquire_projection_output(
    rows: int, workspace: ProjectionWorkspace, backend: CommBackend | None
) -> torch.Tensor:
    """Borrow a [TP*rows,N] GEMM destination for the following ReduceScatter."""
    if backend is None:
        backend = get_global_backend()
    return backend.acquire_projection_output(rows, workspace)


def projection_reduce_scatter(
    tensor: torch.Tensor,
    rows: int,
    workspace: ProjectionWorkspace,
    backend: CommBackend | None,
) -> torch.Tensor:
    """Sum [TP*rows,N] partials into owned [rows,N] local outputs."""
    if backend is None:
        backend = get_global_backend()
    return backend.projection_reduce_scatter(tensor, rows, workspace)


def all_reduce(
    tensor: torch.Tensor | tuple[torch.Tensor, ...],
    group: Group,
    backend: CommBackend | None = None,
    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """All-reduce one tensor or independent tensors across a group."""
    if backend is None:
        backend = get_global_backend()
    return backend.all_reduce(tensor, group, op=op)


def supports_all_reduce_mhc_norm(
    x: torch.Tensor, norm_weight: torch.Tensor, group: Group
) -> bool:
    """Select fusion before scheduling the producer's ordinary all-reduce.

    x describes the producer's anticipated output shape, dtype, device and
    layout, including CED narrowing and graph padding. The producer must honor
    that spec; HC residual/coefficient formats follow all_reduce_mhc_norm's
    contract. Backend configuration and the prepared workspace remain fixed
    between this query and execution. Only the shape, dtype, device and
    contiguity of norm_weight are inspected. It may be a producer norm with
    equivalent storage; execution uses the consumer's own weight and epsilon.
    This query does not read tensor values, retain tensors or launch GPU work.
    """
    return get_global_backend().supports_all_reduce_mhc_norm(x, norm_weight, group)


def all_reduce_mhc_norm(
    x, residual, post, comb, pre, weight, eps, group: Group
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute fusion selected by supports_all_reduce_mhc_norm.

    x is BF16 [T,H], residual BF16 [T,4,H], post/pre FP32 [T,4], comb
    FP32 [T,4,4], and weight BF16 [H]. All tensors are contiguous and on the
    same CUDA device; eps is positive and group is the admitted prepared group.
    Returns updated residual and normalized input. Contract violations and
    execution errors propagate; this operation never requests a fallback.
    """
    return get_global_backend().all_reduce_mhc_norm(
        x, residual, post, comb, pre, weight, eps, group
    )


def all_reduce_mhc_post(
    tensor: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> torch.Tensor | None:
    """Fused all-reduce + hc=4 mHC post, or None if the backend has none.

    See ``CommBackend.all_reduce_mhc_post``. On None nothing has run, and the
    caller reduces ``tensor`` and applies the post itself.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.all_reduce_mhc_post(tensor, residual, post, comb, group)


def prepare_all_reduce_lane(
    group: Group,
    hidden_dim: int,
    backend: CommBackend | None = None,
) -> bool:
    """Prepare a backend-owned one-shot lane for a wider fused reduction."""

    if backend is None:
        backend = get_global_backend()
    # No try/except: "backend can't do it" is already the base-class default
    # (returns False), and this call is COLLECTIVE — swallowing a real error
    # on one rank while peers succeed would leave the group disagreeing on
    # the lane width. Let real failures propagate loudly.
    return backend.prepare_all_reduce_lane(group, hidden_dim)


def prepare_all_reduce_buffers(
    group: Group,
    *,
    staged_max_numel: int,
    producer_direct_max_numel: int,
    dtype: torch.dtype,
    backend: CommBackend | None,
) -> bool:
    """Ask the active backend to allocate all-reduce buffers before cache planning.

    Args:
        group: Global ranks participating in the reductions.
        staged_max_numel: Maximum ordinary all-reduce payload in elements.
        producer_direct_max_numel: Maximum producer-direct payload in elements.
        dtype: Element type shared by the prepared paths.
        backend: Backend to prepare, or ``None`` to use the global backend.

    Returns:
        Whether the active backend prepared the requested buffers.
    """

    if backend is None:
        backend = get_global_backend()
    return backend.prepare_all_reduce_buffers(
        group,
        staged_max_numel=staged_max_numel,
        producer_direct_max_numel=producer_direct_max_numel,
        dtype=dtype,
    )


def prepare_all_reduce_fusion(
    group: Group,
    hidden_dim: int,
    max_token_num: int,
) -> bool:
    """Prepare fused all-reduce kernels before graph capture."""

    try:
        process_group = _get_process_group(group)
        return kernel_prepare_allreduce_fusion(
            rank=process_group.rank(),
            group=process_group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
        )
    except Exception:
        return False


def can_acquire_all_reduce_outputs(
    shapes: tuple[tuple[int, ...], ...],
    like: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
) -> bool:
    """Whether ``acquire_all_reduce_outputs`` returns producer-direct memory.

    ``acquire_all_reduce_outputs`` always returns writable buffers, falling
    back to ordinary allocations the collective has to stage. Ask here when the
    buffers are only worth taking if the reduction consumes them in place.

    This is COLLECTIVE in the same sense the acquire is: every rank of
    ``group`` must call it with identical arguments, or the group will disagree
    on which collective the tail runs.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.can_acquire_all_reduce_outputs(shapes, like, group, op=op)


def acquire_all_reduce_outputs(
    shapes: tuple[tuple[int, ...], ...],
    like: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
) -> tuple[torch.Tensor, ...]:
    """Acquire writable producer outputs for a later all-reduce.

    This function does not launch a collective. Fill the returned outputs,
    then pass them to ``all_reduce``.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.acquire_all_reduce_outputs(shapes, like, group, op=op)


def all_gather(
    tensor: torch.Tensor,
    group: Group,
    dim: int = -1,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """All-gather the tensor across the given communication group."""
    if backend is None:
        backend = get_global_backend()
    return backend.all_gather(tensor, group, dim)


def all_gather_single(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> None:
    """All-gather input into a pre-allocated output buffer."""
    if backend is None:
        backend = get_global_backend()
    backend.all_gather_single(output, input, group)


def all_gather_single_async(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> CommWork:
    """Start :func:`all_gather_single` and return its work; ``wait()`` on it
    orders the current stream after the collective. To overlap the shared
    collectives over the same ranks, pass a backend on named process groups
    (``CommBackend.for_named_groups``): collectives sharing a communicator
    run in issue order."""
    if backend is None:
        backend = get_global_backend()
    return backend.all_gather_single_async(output, input, group)


def reduce_scatter(
    tensor: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """Reduce-scatter the tensor across the given communication group."""
    if backend is None:
        backend = get_global_backend()
    return backend.reduce_scatter(tensor, group)


def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
    backend: CommBackend | None = None,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
) -> None:
    """All-to-all along dim 0 into a pre-allocated output buffer.

    Without split sizes the exchange is even; see
    ``CommBackend.all_to_all_single`` for the uneven form.
    """
    if backend is None:
        backend = get_global_backend()
    backend.all_to_all_single(
        output,
        input,
        group,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )


def _check_split_sizes(split_sizes: list[int], group: Group, rows: int) -> None:
    if len(split_sizes) != len(group):
        raise ValueError(
            f"expected one row count per rank of a {len(group)}-rank group, got "
            f"{len(split_sizes)}"
        )
    if any(count < 0 for count in split_sizes):
        raise ValueError(f"row counts must be non-negative, got {split_sizes}")
    if sum(split_sizes) != rows:
        raise ValueError(
            f"row counts {split_sizes} sum to {sum(split_sizes)}, but the tensor "
            f"has {rows} rows"
        )


def all_to_all_transpose(
    x: torch.Tensor,
    group: Group,
    input_split_sizes: list[int],
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """Exchange token shards for feature shards across ``group``.

    ``x`` is ``[T_full, F_local]``: this rank's feature shard of every rank's
    tokens, rows rank-major (``input_split_sizes[i]`` rows belong to the
    group's ``i``-th rank). The result is ``[T_own, W * F_local]``: this
    rank's own tokens with the feature shards of all ``W`` ranks concatenated
    in rank order. Pure data movement, so the bytes do not depend on the row
    counts. A rank may own zero rows.

    This is the tail of a column-parallel GEMM on hidden (TP batch
    invariance), the logits transpose of a vocab-sharded LM head under
    attention DP, and the heads-to-tokens leg of head-sharded attention.
    """
    if x.dim() != 2:
        raise ValueError(f"all_to_all_transpose takes a 2-D tensor, got {x.dim()}-D")
    world_size = len(group)
    _check_split_sizes(input_split_sizes, group, x.shape[0])
    if world_size == 1:
        return x
    rows = input_split_sizes[group.index(torch.distributed.get_rank())]
    width = x.shape[1]
    if x.shape[0] == 0:
        # Nothing to move for the whole group (every rank reads the same
        # counts, so every rank skips the collective together).
        return x.new_empty(0, world_size * width)
    received = x.new_empty(world_size * rows, width)
    all_to_all_single(
        received,
        x.contiguous(),
        group,
        backend=backend,
        output_split_sizes=[rows] * world_size,
        input_split_sizes=input_split_sizes,
    )
    # Received chunk i is rank i's feature shard of my rows.
    return (
        received.view(world_size, rows, width)
        .transpose(0, 1)
        .reshape(rows, world_size * width)
    )


def all_to_all_head_scatter(
    x: torch.Tensor,
    group: Group,
    output_split_sizes: list[int],
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """Inverse of ``all_to_all_transpose`` for per-head activations.

    ``x`` is ``[T_own, W * H_local, D]``: this rank's own tokens with every
    rank's head block in rank order. The result is ``[T_full, H_local, D]``:
    this rank's head block of every rank's tokens, rows rank-major
    (``output_split_sizes[i]`` rows from the group's ``i``-th rank). Pure data
    movement; a rank may own zero rows.
    """
    if x.dim() != 3:
        raise ValueError(
            f"all_to_all_head_scatter takes a [tokens, heads, dim] tensor, got "
            f"{x.dim()}-D"
        )
    world_size = len(group)
    if x.shape[1] % world_size:
        raise ValueError(
            f"{x.shape[1]} heads do not split over {world_size} ranks evenly"
        )
    rows_full = sum(output_split_sizes)
    _check_split_sizes(output_split_sizes, group, rows_full)
    if world_size == 1:
        return x
    rows_own, heads, dim = x.shape
    heads_local = heads // world_size
    if output_split_sizes[group.index(torch.distributed.get_rank())] != rows_own:
        raise ValueError(
            f"this rank owns {rows_own} rows but the row counts "
            f"{output_split_sizes} give it "
            f"{output_split_sizes[group.index(torch.distributed.get_rank())]}"
        )
    if rows_full == 0:
        # Nothing to move for the whole group; every rank skips together.
        return x.new_empty(0, heads_local, dim)
    # Head block i (destined for rank i) becomes send chunk i.
    sent = (
        x.view(rows_own, world_size, heads_local * dim)
        .transpose(0, 1)
        .reshape(world_size * rows_own, heads_local * dim)
    )
    received = x.new_empty(rows_full, heads_local * dim)
    all_to_all_single(
        received,
        sent,
        group,
        backend=backend,
        output_split_sizes=output_split_sizes,
        input_split_sizes=[rows_own] * world_size,
    )
    return received.view(rows_full, heads_local, dim)


# ---------------------------------------------------------------------------
# Fused ops (comm + residual + norm)
# ---------------------------------------------------------------------------


def fused_all_reduce(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """All-reduce with optional fused residual + RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.all_reduce(tensor, group)

    if fusion_params.fusion_op == FusionOp.RESIDUAL_RMS_NORM:
        return allreduce_residual_rmsnorm(
            input_tensor=tensor,
            residual=fusion_params.residual,
            weight=fusion_params.norm_weight,
            rank=rank,
            group=_get_process_group(group),
            eps=fusion_params.eps,
            fp32_acc=fusion_params.fp32_acc,
            block_quant_fp8=fusion_params.block_quant_fp8,
            residual_reduce_scattered=fusion_params.residual_reduce_scattered,
            has_partial_norm_out=fusion_params.has_partial_norm_out,
            trigger_completion_at_end=fusion_params.trigger_completion_at_end,
            max_sm_to_use=fusion_params.max_sm_to_use,
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_all_reduce"
    )


def fused_reduce_scatter(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reduce-scatter with optional fused residual + RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.reduce_scatter(tensor, group)

    if fusion_params.fusion_op == FusionOp.RS_RESIDUAL_RMS_NORM:
        return reducescatter_residual_rmsnorm(
            input_tensor=tensor,
            weight=fusion_params.norm_weight,
            residual=fusion_params.residual,
            eps=fusion_params.eps,
            rank=rank,
            group=_get_process_group(group),
            add_in=fusion_params.add_in,
            fp32_acc=fusion_params.fp32_acc,
            block_quant_fp8=fusion_params.block_quant_fp8,
            # Shape-derived growth; post-capture grows are refused -- arm before capture.
            max_token_num=fusion_params.max_token_num or tensor.shape[0],
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_reduce_scatter"
    )


def fused_all_gather(
    tensor: torch.Tensor,
    rank: int,
    group: Group,
    dim: int = -1,
    backend: CommBackend | None = None,
    fusion_params: FusionParams | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """All-gather with optional fused dual-RMSNorm."""
    if backend is None:
        backend = get_global_backend()

    if fusion_params is None or fusion_params.fusion_op == FusionOp.NONE:
        return backend.all_gather(tensor, group, dim)

    if fusion_params.fusion_op == FusionOp.AG_DUAL_RMS_NORM:
        return allgather_dual_rmsnorm(
            qkv=tensor,
            weight_q_a=fusion_params.norm_weight,
            eps_q=fusion_params.eps,
            weight_kv_a=fusion_params.norm_weight_2,
            eps_kv=fusion_params.eps_2,
            rank=rank,
            group=_get_process_group(group),
            total_num_tokens=fusion_params.total_num_tokens,
            # Shape-derived growth; post-capture grows are refused -- arm before capture.
            max_token_num=fusion_params.max_token_num
            or max(tensor.shape[0], fusion_params.total_num_tokens),
            fp32_acc=fusion_params.fp32_acc,
            block_quant_fp8=fusion_params.block_quant_fp8,
        )

    raise ValueError(
        f"Unsupported fusion_op {fusion_params.fusion_op} for fused_all_gather"
    )


# ---------------------------------------------------------------------------
# Token-aware ops (uneven token distribution via TritonRSAG)
# ---------------------------------------------------------------------------

# The wire alignment of a byte-preserving row gather: the low-latency
# all-gather moves 16-byte vectors, so every rank's payload must be a
# multiple of it whatever its row count.
_ROW_GATHER_ALIGN_BYTES = 16


def token_all_gather(
    tensor: torch.Tensor,
    group: Group,
    scattered_num_tokens: list[int],
    backend=None,
) -> torch.Tensor:
    """All-gather with token-aware distribution (TritonRSAG).

    Args:
        scattered_num_tokens: Number of tokens on each rank in the group,
            e.g. [50, 50, 51, 49] for 4 ranks with 200 total tokens.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.token_all_gather(tensor, group, scattered_num_tokens)


def token_all_gather_rows(
    rows: torch.Tensor,
    group: Group,
    scattered_num_tokens: list[int],
    backend=None,
) -> torch.Tensor:
    """Token-aware all-gather of 2-D rows of any dtype, byte-preserving.

    :func:`token_all_gather` moves bf16 rows (its low-latency solution
    asserts the dtype); pure data movement -- gathered cache rows, packed
    index-K bytes, fp32 scales, token ids -- has no dtype of its own, so
    non-bf16 rows travel as bf16 pairs of their bytes and come back viewed
    as the input dtype. The row byte width must be even. A row is padded
    to a multiple of ``_ROW_GATHER_ALIGN_BYTES`` on the wire: the
    low-latency solution moves 16-byte vectors and the row counts are the
    caller's (a query shard, a page owner's rows), so a narrow row -- one
    int64 token id is 8 bytes -- would only align for even counts.

    Args:
        rows: ``[local_rows, width]`` this rank's rows.
        group: The gather group.
        scattered_num_tokens: Rows every rank of the group contributes.

    Returns:
        ``[sum(scattered_num_tokens), width]`` rows in rank order, ``rows``'
        dtype.
    """
    if rows.dim() != 2:
        raise ValueError(f"token_all_gather_rows takes 2-D rows, got {rows.dim()}-D")
    row_bytes = rows.shape[1] * rows.element_size()
    pad_bytes = -row_bytes % _ROW_GATHER_ALIGN_BYTES
    if rows.dtype == torch.bfloat16 and pad_bytes == 0:
        return token_all_gather(rows.contiguous(), group, scattered_num_tokens, backend)
    if row_bytes % 2:
        raise ValueError(
            f"rows of {row_bytes} bytes cannot travel as bf16 pairs; pad the row "
            "to an even byte width"
        )
    payload = rows.contiguous().view(torch.uint8).view(torch.bfloat16)
    if pad_bytes:
        payload = torch.nn.functional.pad(payload, (0, pad_bytes // 2))
    gathered = token_all_gather(payload, group, scattered_num_tokens, backend)
    if pad_bytes:
        gathered = gathered[:, : row_bytes // 2].contiguous()
    return gathered.view(torch.uint8).view(rows.dtype)


def token_reduce_scatter(
    tensor: torch.Tensor,
    group: Group,
    scattered_num_tokens: list[int],
    backend=None,
) -> torch.Tensor:
    """Reduce-scatter with token-aware distribution (TritonRSAG).

    Args:
        scattered_num_tokens: Number of tokens on each rank in the group,
            e.g. [50, 50, 51, 49] for 4 ranks with 200 total tokens.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.token_reduce_scatter(tensor, group, scattered_num_tokens)


def pp_send(
    tensor: torch.Tensor,
    dst_group_index: int,
    group: Group,
    backend: CommBackend | None = None,
) -> None:
    """Send a tensor to another pipeline stage (P2P over the PP group).

    Args:
        tensor: Contiguous device tensor to send.
        dst_group_index: Destination position within ``group`` (the PP rank of
            the receiving stage), not a global rank.
        group: The PP group (one rank per stage, same intra-stage position).
        backend: Communication backend; defaults to the global backend.
    """
    if backend is None:
        backend = get_global_backend()
    backend.send(tensor.contiguous(), dst_group_index, group)


def pp_recv(
    size: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    src_group_index: int,
    group: Group,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """Receive a tensor from another pipeline stage (P2P over the PP group).

    The shape/dtype are supplied by the caller: every PP rank runs the same
    deterministic scheduler, so the receiver derives the payload geometry from
    its own forward op without a metadata exchange.

    Args:
        size: Shape of the incoming tensor.
        dtype: Element type of the incoming tensor.
        device: Device to allocate the receive buffer on.
        src_group_index: Source position within ``group`` (the PP rank of the
            sending stage), not a global rank.
        group: The PP group (one rank per stage, same intra-stage position).
        backend: Communication backend; defaults to the global backend.

    Returns:
        The received tensor.
    """
    if backend is None:
        backend = get_global_backend()
    return backend.recv(torch.Size(size), dtype, device, src_group_index, group)
