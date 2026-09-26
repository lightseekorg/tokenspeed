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

"""Token-sharded attention using the prepared Iris producer and MoE result."""

import math

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform


def _overlaps(tensor: torch.Tensor, buffer: torch.Tensor) -> bool:
    if tensor.numel() == 0:
        return False
    # History can have padding between blocks or tokens. Include its full span.
    span = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    begin = tensor.data_ptr()
    end = begin + span * tensor.element_size()
    return (
        begin < buffer.data_ptr() + buffer.numel() * buffer.element_size()
        and buffer.data_ptr() < end
    )


def iris_attention_prefill_mix(
    partial: torch.Tensor,
    residual: torch.Tensor | None,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float,
    out_norm_weight: torch.Tensor,
    out_norm_eps: float,
    num_valid_blocks: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Reduce attention token shards and gather their normalized AttnRes mix.

    Args:
        partial: Prepared contiguous CUDA BF16 projection, ``[M, 7168]``.
            M is positive and divisible by eight. The producer is preserved.
        residual: Replicated contiguous BF16 residual, ``[M, 7168]``, or None
            on a block-write layer. A disjoint residual is preserved. An exact
            alias of the prepared MoE result is consumed before that result is
            overwritten; shifted overlaps are unsupported.
        block_residual: Replicated BF16 history, ``[K, M, 7168]``, with a
            contiguous hidden dimension. Padding between tokens/blocks is allowed.
        res_weight: Replicated contiguous BF16 scorer weight, ``[7168]``.
        rms_weight: Replicated contiguous BF16 score RMSNorm weight, ``[7168]``.
        eps: Positive finite score RMSNorm epsilon.
        out_norm_weight: Replicated contiguous BF16 output RMSNorm weight,
            ``[7168]``.
        out_norm_eps: Positive finite output RMSNorm epsilon.
        num_valid_blocks: Number of leading history snapshots to mix, from 0
            through 11. History and weights must not overlap collective storage.
        group: The exact eight-rank group owning the prepared producer. Every
            rank supplies the same shapes, eligibility and operation order.

    Returns:
        An owned residual shard ``[M/8, 7168]`` and a borrowed replicated
        activation ``[M, 7168]``; or None before allocation or publication when
        unsupported. The activation reuses the MoE result, so its consumers must
        finish on the calling stream (or join it) before the next attention mix
        or MoE tail on this group. Clone retained activations. The residual stays
        explicitly sharded until the MoE tail consumes it or fallback gathers it.

    No process groups or symmetric buffers are created. Calls sharing this
    state are serialized on one stream, including capture and replay. Runtime
    policy selects the profitable token window independently of this contract.
    """
    if (
        not current_platform().is_cdna4
        or partial.ndim != 2
        or partial.shape[0] <= 0
        or partial.shape[0] % 8 != 0
        or partial.shape[1] != 7168
        or not partial.is_cuda
        or partial.dtype != torch.bfloat16
        or not partial.is_contiguous()
        or group.size() != 8
        or block_residual.ndim != 3
        or block_residual.shape[1:] != partial.shape
        or block_residual.dtype != partial.dtype
        or block_residual.device != partial.device
        or block_residual.stride(-1) != 1
        or (partial.shape[0] - 1) * block_residual.stride(1) + 7168 >= 1 << 30
        or not isinstance(num_valid_blocks, int)
        or not 0 <= num_valid_blocks <= min(11, block_residual.shape[0])
        or not math.isfinite(eps)
        or eps <= 0
        or not math.isfinite(out_norm_eps)
        or out_norm_eps <= 0
    ):
        return None
    if residual is not None and (
        residual.shape != partial.shape
        or residual.dtype != partial.dtype
        or residual.device != partial.device
        or not residual.is_contiguous()
    ):
        return None
    weights = (res_weight, rms_weight, out_norm_weight)
    if any(
        weight.shape != (7168,)
        or weight.dtype != partial.dtype
        or weight.device != partial.device
        or not weight.is_contiguous()
        for weight in weights
    ):
        return None

    from tokenspeed_kernel.ops.communication.iris import IRIS_AR_STATES

    state = next(
        (
            s
            for s in IRIS_AR_STATES.values()
            if s.group is group and s.owns_outputs((partial,))
        ),
        None,
    )
    if state is None:
        return None
    output_buffer = state._moe_tail_output_buf
    flags = state._producer_direct_ready_flags
    gather_flags = state._moe_tail_ready_flags
    if (
        output_buffer is None
        or output_buffer.shape[0] < partial.shape[0]
        or flags is None
        or flags.shape[0] < 24
        or gather_flags is None
        or gather_flags.shape[0] < 128
    ):
        return None
    protected = (state._input_buf, state._producer_direct_scratch_buf, output_buffer)
    for tensor in (block_residual, *weights):
        if any(
            buffer is not None and _overlaps(tensor, buffer) for buffer in protected
        ):
            return None
    if residual is not None:
        for buffer in protected:
            if (
                buffer is not None
                and _overlaps(residual, buffer)
                and not (
                    buffer is output_buffer and residual.data_ptr() == buffer.data_ptr()
                )
            ):
                return None

    from tokenspeed_kernel.ops.residual import attn_res_fwd, attn_res_fwd_available

    rows = partial.shape[0] // 8
    partition = rows * 7168
    first_row = state.rank_in_group * rows
    history = block_residual[:, first_row : first_row + rows]
    if not attn_res_fwd_available(
        partial[:rows],
        history,
        res_weight,
        rms_weight,
        eps,
        out_norm_weight=out_norm_weight,
        out_norm_eps=out_norm_eps,
        delta=None,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=-1,
    ):
        return None

    from tokenspeed_kernel.ops.communication._iris.attention import (
        iris_attention_mix_push_gluon_kernel,
        iris_attention_push_gather_gluon_kernel,
        iris_attention_reduce_scatter_gluon_kernel,
    )

    prefix = torch.empty_like(partial[:rows])
    output = output_buffer[: partial.shape[0]]
    programs = min(24, (partition + 2047) // 2048)
    iris_attention_reduce_scatter_gluon_kernel[(programs,)](
        partial,
        residual,
        prefix,
        flags,
        *state._heap_base_addresses,
        RANK=state.rank_in_group,
        PARTITION_ELEMENTS=partition,
        BLOCK_ELEMENTS=2048,
        NUM_PROGRAMS=programs,
        NUM_WARPS=4,
        HAS_RESIDUAL=residual is not None,
        num_warps=4,
    )
    # The ordinary mixer is faster in the middle token range. At larger sizes,
    # longer histories also need enough rows to amortize live peer pointers.
    fuse_mix = (partial.shape[0] < 1024 or partial.shape[0] >= 4096) and (
        num_valid_blocks <= 6 or (partial.shape[0] >= 7680 and num_valid_blocks <= 8)
    )
    if fuse_mix:
        gather_programs = min(128, rows)
        num_subgroups = 8 if num_valid_blocks <= 7 else 4
        iris_attention_mix_push_gluon_kernel[(gather_programs,)](
            prefix,
            output,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight,
            gather_flags,
            *state._heap_base_addresses,
            RANK=state.rank_in_group,
            LOCAL_ROWS=rows,
            STRIDE_BLOCK_T=block_residual.stride(1),
            STRIDE_BLOCK_N=block_residual.stride(0),
            NUM_VALID_BLOCKS=num_valid_blocks,
            SCORE_EPS=eps,
            OUTPUT_EPS=out_norm_eps,
            NUM_PROGRAMS=gather_programs,
            NUM_WARPS=num_subgroups,
            num_warps=num_subgroups,
        )
    else:
        # Longer histories favor the existing mixer without persistent peer
        # pointers occupying registers throughout the candidate reductions.
        mixed = attn_res_fwd(
            prefix,
            history,
            res_weight,
            rms_weight,
            eps,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            delta=None,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=-1,
        )
        gather_programs = min(32, (partition + 2047) // 2048)
        iris_attention_push_gather_gluon_kernel[(gather_programs,)](
            mixed,
            output,
            gather_flags,
            *state._heap_base_addresses,
            RANK=state.rank_in_group,
            PARTITION_ELEMENTS=partition,
            BLOCK_ELEMENTS=2048,
            NUM_PROGRAMS=gather_programs,
            NUM_WARPS=4,
            num_warps=4,
        )
    return prefix, output
