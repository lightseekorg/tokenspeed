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

"""K3 attn all-reduce with an AttnRes epilogue using precomputed history partials."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import gl, gluon, triton
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_drain_subgroup_vmem,
    _iris_heap_base,
    _iris_sync_rank_token,
)
from tokenspeed_kernel.platform import current_platform


@dataclass(frozen=True)
class KimiK3AttnResKernelConfig:
    """Kimi-K3 fused attention-TP all-reduce and AttnRes contract.

    Attributes:
        world_size: Required attention tensor-parallel group size.
        hidden_size: Kimi-K3 attention output width.
        num_subgroups: Number of subgroups in each kernel workgroup.
        elements_per_thread: Number of hidden elements processed per thread.
    """

    world_size: int
    hidden_size: int
    num_subgroups: int
    elements_per_thread: int

    def __post_init__(self) -> None:
        if (
            self.world_size <= 1
            or self.hidden_size <= 0
            or self.num_subgroups <= 0
            or self.elements_per_thread <= 0
        ):
            raise ValueError("invalid Kimi-K3 AttnRes Iris kernel configuration")


ATTNRES_MAX_ROWS = 16
ATTNRES_KERNEL_CONFIG = KimiK3AttnResKernelConfig(
    world_size=8,
    hidden_size=7168,
    num_subgroups=16,
    elements_per_thread=8,
)


def attnres_supported(
    partial: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    world_size: int,
    device: torch.device,
) -> bool:
    """Check the fused tensor contract without allocating a workspace."""
    m, s, acc = scratch
    rows = partial.shape[0] if partial.ndim == 2 else 0
    hidden = ATTNRES_KERNEL_CONFIG.hidden_size
    return (
        current_platform().is_cdna4
        and world_size == ATTNRES_KERNEL_CONFIG.world_size
        and partial.is_cuda
        and 0 < rows <= ATTNRES_MAX_ROWS
        and partial.shape == residual.shape == acc.shape == (rows, hidden)
        and score_weight.shape == output_weight.shape == (hidden,)
        and m.shape == s.shape == (rows,)
        and partial.dtype
        == residual.dtype
        == score_weight.dtype
        == output_weight.dtype
        == torch.bfloat16
        and m.dtype == s.dtype == acc.dtype == torch.float32
        and all(
            tensor.device == device and tensor.is_contiguous()
            for tensor in (partial, residual, score_weight, output_weight, m, s, acc)
        )
    )


class IrisAttnResWorkspace:
    """Own the push inbox, per-row generations, and AttnRes launch contract."""

    @staticmethod
    def heap_bytes(
        max_numel: int, max_rows: int, world_size: int, dtype: torch.dtype
    ) -> int:
        return (
            2
            * world_size
            * (max_numel * dtype.itemsize + max_rows * torch.int32.itemsize)
        )

    def __init__(
        self,
        ctx,
        *,
        group: dist.ProcessGroup,
        rank: int,
        device: torch.device,
        heap_bases: tuple[int, ...],
        max_numel: int,
        max_rows: int,
        dtype: torch.dtype,
    ) -> None:
        self.world_size = group.size()
        self.rank = rank
        self.device = device
        self.heap_bases = heap_bases
        self.max_numel = max_numel
        self.max_rows = max_rows
        self.inbox = ctx.zeros((2, self.world_size, max_numel), dtype=dtype)
        self.epochs = torch.zeros((max_rows,), dtype=torch.int32, device=device)
        self.ready_flags = ctx.zeros((2, max_rows, self.world_size), dtype=torch.int32)
        offset = self.inbox.data_ptr() - heap_bases[rank]
        self.peer_inboxes = tuple(base + offset for base in heap_bases)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def run(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce a Kimi-K3 attention partial and finish its AttnRes mix."""
        assert attnres_supported(
            partial,
            residual,
            score_weight,
            output_weight,
            scratch,
            world_size=self.world_size,
            device=self.device,
        )
        assert partial.shape[0] <= self.max_rows
        assert partial.numel() <= self.max_numel
        assert eps > 0.0
        kernel_config = ATTNRES_KERNEL_CONFIG
        num_tokens = partial.shape[0]
        m, s_, acc = scratch

        hidden = torch.empty_like(partial)
        residual_out = torch.empty_like(residual)
        iris_k3attn_push_oneshot[(num_tokens,)](
            partial,
            residual,
            self.inbox,
            score_weight,
            output_weight,
            m,
            s_,
            acc,
            hidden,
            residual_out,
            self.epochs,
            self.ready_flags,
            *self.peer_inboxes,
            *self.heap_bases,
            RANK=self.rank,
            WORLD_SIZE=self.world_size,
            HIDDEN=kernel_config.hidden_size,
            BLOCK=triton.next_power_of_2(kernel_config.hidden_size),
            MAX_ELEMENTS=self.max_numel,
            READY_SLOT_STRIDE=self.max_rows * self.world_size,
            EPS=eps,
            ELEMENTS_PER_THREAD=kernel_config.elements_per_thread,
            NUM_WARPS=kernel_config.num_subgroups,
            SUBGROUP_SIZE=64,
            num_warps=kernel_config.num_subgroups,
        )
        return hidden, residual_out


@gluon.jit
def _iris_attnres_epilogue(
    reduced,
    residual_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    row_offsets,
    weight_offsets,
    mask,
    row,
    HIDDEN: gl.constexpr,
    EPS: gl.constexpr,
):
    """Add the residual, merge history softmax partials, and RMS-normalize."""
    reduced = reduced.to(gl.bfloat16).to(gl.float32)
    residual = gl.amd.cdna4.buffer_load(
        residual_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    prefix = (reduced + residual).to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        prefix.to(residual_out_ptr.dtype.element_ty),
        residual_out_ptr,
        row_offsets,
        mask=mask,
    )
    score_weight = gl.amd.cdna4.buffer_load(
        score_weight_ptr,
        weight_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    square_sum = gl.sum(gl.where(mask, prefix * prefix, 0.0), axis=0)
    dot = gl.sum(gl.where(mask, prefix * score_weight, 0.0), axis=0)
    prefix_logit = dot * gl.rsqrt(square_sum / HIDDEN + EPS)
    block_m = gl.load(scratch_m_ptr + row)
    block_s = gl.load(scratch_s_ptr + row)
    maximum = gl.maximum(block_m, prefix_logit)
    block_correction = gl.exp(block_m - maximum)
    prefix_weight = gl.exp(prefix_logit - maximum)
    inverse_sum = 1.0 / (block_s * block_correction + prefix_weight)
    block_acc = gl.amd.cdna4.buffer_load(
        scratch_acc_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    mixed = (
        ((block_acc * block_correction + prefix_weight * prefix) * inverse_sum)
        .to(gl.bfloat16)
        .to(gl.float32)
    )
    output_square_sum = gl.sum(gl.where(mask, mixed * mixed, 0.0), axis=0)
    inverse_rms = gl.rsqrt(output_square_sum / HIDDEN + EPS)
    output_weight = gl.amd.cdna4.buffer_load(
        output_weight_ptr,
        weight_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        (mixed * inverse_rms * output_weight).to(hidden_ptr.dtype.element_ty),
        hidden_ptr,
        row_offsets,
        mask=mask,
    )


@gluon.jit
def iris_k3attn_push_oneshot(
    partial_ptr,
    residual_ptr,
    inbox_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    generations,
    ready_flags,
    inbox_0: gl.pointer_type(gl.bfloat16),
    inbox_1: gl.pointer_type(gl.bfloat16),
    inbox_2: gl.pointer_type(gl.bfloat16),
    inbox_3: gl.pointer_type(gl.bfloat16),
    inbox_4: gl.pointer_type(gl.bfloat16),
    inbox_5: gl.pointer_type(gl.bfloat16),
    inbox_6: gl.pointer_type(gl.bfloat16),
    inbox_7: gl.pointer_type(gl.bfloat16),
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    READY_SLOT_STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """Push K3 attn rows into two alternating inboxes and reduce in rank order.

    Add the residual to the BF16 reduction, merge historical AttnRes partials,
    and RMS-normalize the BF16 mixture. Per-row generations guard slot reuse.
    """
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    element = gl.arange(0, BLOCK, layout=layout)
    mask = element < HIDDEN
    row_offsets = (row * HIDDEN + element).to(gl.int32)
    weight_offsets = element.to(gl.int32)
    local = gl.amd.cdna4.buffer_load(
        partial_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    )
    generation = gl.load(generations + row).to(gl.int32) + 1
    slot = generation & 1
    inbox_slot_offset = slot * WORLD_SIZE * MAX_ELEMENTS
    sync_ready_flags = ready_flags + slot * READY_SLOT_STRIDE
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    for peer_delta in gl.static_range(1, WORLD_SIZE):
        destination = (RANK + peer_delta) % WORLD_SIZE
        destination_inbox = _iris_heap_base(
            destination,
            inbox_0,
            inbox_1,
            inbox_2,
            inbox_3,
            inbox_4,
            inbox_5,
            inbox_6,
            inbox_7,
        )
        gl.amd.cdna4.buffer_store(
            local,
            destination_inbox + inbox_slot_offset + RANK * MAX_ELEMENTS,
            row_offsets,
            mask=mask,
            cache=".wt",
        )
    _iris_drain_subgroup_vmem()
    # Join subgroup-local drains before publishing the generation to peers.
    gl.barrier()

    _iris_sync_rank_token(
        sync_ready_flags,
        row,
        generation,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
    )
    local_inbox = inbox_sym_ptr + inbox_slot_offset

    if RANK == 0:
        reduced = local.to(gl.float32)
    else:
        reduced = gl.amd.cdna4.buffer_load(
            local_inbox,
            row_offsets,
            mask=mask,
            other=0.0,
            cache=".cg",
        ).to(gl.float32)
    for source in gl.static_range(1, WORLD_SIZE):
        if source == RANK:
            reduced += local.to(gl.float32)
        else:
            reduced += gl.amd.cdna4.buffer_load(
                local_inbox + source * MAX_ELEMENTS,
                row_offsets,
                mask=mask,
                other=0.0,
                cache=".cg",
            ).to(gl.float32)

    gl.store(generations + row, generation)
    _iris_attnres_epilogue(
        reduced,
        residual_ptr,
        score_weight_ptr,
        output_weight_ptr,
        scratch_m_ptr,
        scratch_s_ptr,
        scratch_acc_ptr,
        hidden_ptr,
        residual_out_ptr,
        row_offsets,
        weight_offsets,
        mask,
        row,
        HIDDEN,
        EPS,
    )
