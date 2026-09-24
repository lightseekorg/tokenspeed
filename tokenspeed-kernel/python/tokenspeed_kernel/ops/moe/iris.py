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

"""Token-sharded Kimi K3 MoE tail using an already prepared Iris workspace."""

import math

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform


def iris_kimi3_moe_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    prefix: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    eps: float | None,
    group: dist.ProcessGroup,
) -> torch.Tensor | None:
    """Reduce, normalize, project and gather a producer-direct K3 MoE tail.

    Each of eight ranks reduces its token rows, projects those rows using the
    replicated weight, then pushes the final residual into the prepared result.
    The result is borrowed and the producer inputs are preserved.

    Args:
        routed_partial: Contiguous BF16 routed partial, shaped ``[M, 3584]``.
        shared_partial: Contiguous BF16 shared partial, shaped ``[M, 7168]``,
            immediately following ``routed_partial`` in the prepared Iris input.
        prefix: Replicated contiguous BF16 residual, shaped ``[M, 7168]``.
            May exactly alias the borrowed result, which is then updated in
            place. Shifted overlaps and aliases with producer scratch are
            unsupported. A disjoint prefix is preserved.
        projection_weight: Replicated contiguous BF16 weight, ``[7168, 3584]``.
        norm_weight: Replicated contiguous BF16 RMSNorm weight, ``[3584]``,
            or None to omit normalization.
        eps: Positive finite RMSNorm epsilon, or None with no normalization.
        group: The same eight-rank device process group that owns the prepared
            producer-direct Iris buffers. Every rank must call on the stream
            that joins both producers, with rank-uniform shapes and eligibility.

    Returns:
        Borrowed ``[M, 7168]`` BF16 output, valid until the next tail on this
        group; ordinary producers and collectives do not reuse it. All consumers
        must finish on the calling stream (or join it) before that next call,
        which may consume the result as its prefix. Retained results must be
        cloned. Weights must not alias the result.
        Returns None without allocating or launching if the inputs are
        unsupported or not owned by this group.
        M must be positive and divisible by eight. The runtime owns the measured
        token-count window; this operation only checks its execution contract.

    No symmetric allocation or process-group initialization happens here.
    The operation reuses the existing reduction scratch and epoch flags. Like
    the ordinary Iris reduction, invocations sharing this state are serialized
    on one stream, including graph capture and replay.
    """
    if not current_platform().is_cdna4 or routed_partial.ndim != 2:
        return None
    rows, latent = routed_partial.shape
    if (
        rows <= 0
        or rows % 8 != 0
        or latent != 3584
        or shared_partial.shape != (rows, 7168)
        or prefix.shape != (rows, 7168)
        or projection_weight.shape != (7168, 3584)
        or group.size() != 8
    ):
        return None
    tensors = (routed_partial, shared_partial, prefix, projection_weight)
    if norm_weight is not None:
        if (
            norm_weight.shape != (3584,)
            or eps is None
            or not math.isfinite(eps)
            or eps <= 0
        ):
            return None
        tensors += (norm_weight,)
    elif eps is not None:
        return None
    if any(
        not tensor.is_cuda
        or tensor.device != routed_partial.device
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        for tensor in tensors
    ):
        return None

    # Iris is optional; import it only after the architecture/shape contract
    # holds. Look up the owner, never allocate a competing communication state.
    from tokenspeed_kernel.ops.communication.iris import IRIS_AR_STATES

    state = next(
        (
            candidate
            for candidate in IRIS_AR_STATES.values()
            if candidate.group is group
            and candidate.owns_outputs((routed_partial, shared_partial))
        ),
        None,
    )
    if state is None:
        return None
    local_rows = rows // 8
    routed_elements = local_rows * 3584
    shared_elements = local_rows * 7168
    scratch = state._producer_direct_scratch_buf
    flags = state._producer_direct_ready_flags
    programs = 24
    gather_programs = 128
    result_buffer = state._moe_tail_output_buf
    gather_flags = state._moe_tail_ready_flags
    if (
        scratch is None
        or scratch.numel() < routed_elements + shared_elements
        or flags is None
        or flags.shape[0] < programs
        or gather_flags is None
        or gather_flags.shape[0] < gather_programs
        or result_buffer is None
        or result_buffer.shape[0] < rows
    ):
        return None
    # Reject unsafe overlaps before launching either collective.
    for tensor in tensors[2:]:
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        for buffer in (state._input_buf, scratch, result_buffer):
            buffer_start = buffer.data_ptr()
            buffer_end = buffer_start + buffer.numel() * buffer.element_size()
            if start < buffer_end and buffer_start < end:
                # Exact prefix aliasing is safe: each rank reads then writes
                # only its own rows. Shifted aliases can cross unread tiles.
                if not (
                    buffer is result_buffer
                    and tensor is prefix
                    and start == buffer_start
                ):
                    return None

    from tokenspeed_kernel.ops.communication._iris.prefill import (
        iris_moe_add_push_gather_gluon_kernel,
        iris_moe_reduce_scatter_gluon_kernel,
    )
    from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_latent_projection
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

    routed = scratch[:routed_elements].view(local_rows, 3584)
    shared = scratch[routed_elements : routed_elements + shared_elements].view(
        local_rows, 7168
    )
    projected = torch.empty(
        (local_rows, 7168), device=prefix.device, dtype=prefix.dtype
    )
    output = result_buffer[:rows]
    iris_moe_reduce_scatter_gluon_kernel[(programs,)](
        state._input_buf,
        scratch,
        flags,
        *state._heap_base_addresses,
        RANK=state.rank_in_group,
        ROWS=rows,
        FIRST_WIDTH=3584,
        SECOND_WIDTH=7168,
        BLOCK_ELEMENTS=2048,
        NUM_PROGRAMS=programs,
        NUM_WARPS=4,
        num_warps=4,
    )
    normalized = (
        rmsnorm(routed, norm_weight, eps, residual=None, out=None)
        if norm_weight is not None
        else routed
    )
    kimi3_latent_projection(
        normalized, projection_weight, out=projected, solution="auto"
    )
    iris_moe_add_push_gather_gluon_kernel[(gather_programs,)](
        projected,
        shared,
        prefix,
        output,
        gather_flags,
        *state._heap_base_addresses,
        RANK=state.rank_in_group,
        PARTITION_ELEMENTS=shared_elements,
        BLOCK_ELEMENTS=2048,
        NUM_PROGRAMS=gather_programs,
        NUM_WARPS=4,
        num_warps=4,
    )
    return output
