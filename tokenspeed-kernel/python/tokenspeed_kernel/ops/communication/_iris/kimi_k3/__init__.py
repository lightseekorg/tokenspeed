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

"""Kimi-K3's prepared Iris fusions, shared by sequential decoder layers."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication.iris import IrisAllReduce


def allreduce_residual_attnres_max_tokens(world_size: int) -> int:
    """Return K3's measured fused AttnRes window for this group size."""
    return 16 if current_platform().is_cdna4 and world_size == 8 else 0


class K3IrisState:
    """Retain one group's all-reduce and its optional K3 fusion storage.

    Producers, attention mixing, and the MoE tail use these buffers in stream
    order. A sharded activation is a view of state storage overwritten by
    the next fusion; its local residual shard remains caller-owned.
    """

    def __init__(
        self,
        all_reduce: "IrisAllReduce",
        *,
        attnres_max_numel: int,
        attnres_max_rows: int,
        enable_lamport: bool,
        sharded_max_rows: int,
    ):
        from tokenspeed_kernel.ops.communication._iris.all_reduce import ProducerLamport
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.allreduce_sharded import (
            IrisAllReduceShardedState,
        )
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.attnres import (
            IrisAttnResState,
        )

        self.all_reduce = all_reduce
        self.enable_lamport = enable_lamport
        self.attnres = (
            IrisAttnResState(
                all_reduce.peers, attnres_max_numel, attnres_max_rows, all_reduce.dtype
            )
            if attnres_max_numel
            else None
        )
        # Only the K3 TP8 routed/shared pair uses this measured Lamport window.
        self.lamport_max_rows = (
            min(all_reduce.producer_direct_max_numel // (3584 + 7168), 6)
            if enable_lamport
            and ProducerLamport.supported(all_reduce.world_size, all_reduce.dtype)
            else 0
        )
        self.lamport = (
            ProducerLamport(
                all_reduce.producer, max_numel=self.lamport_max_rows * (3584 + 7168)
            )
            if self.lamport_max_rows
            else None
        )
        self.sharded = (
            IrisAllReduceShardedState(
                all_reduce.producer_pull, max_rows=sharded_max_rows
            )
            if sharded_max_rows
            else None
        )

    def reduce_moe(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Reduce owned producer outputs using K3's existing shape policy."""
        assert self.all_reduce.owns_outputs(tensors)
        shapes = tuple(tuple(tensor.shape) for tensor in tensors)
        if (
            self.lamport is not None
            and len(shapes) == 2
            and len(shapes[0]) == len(shapes[1]) == 2
            and 0 < shapes[0][0] == shapes[1][0] <= self.lamport_max_rows
            and (shapes[0][1], shapes[1][1]) in ((3584, 7168), (7168, 3584))
        ):
            output = torch.empty(
                sum(t.numel() for t in tensors),
                dtype=self.all_reduce.dtype,
                device=self.all_reduce.device,
            )
            self.lamport.run(output)
            return self.all_reduce.producer.views(output, shapes)
        return self.all_reduce.all_reduce_symmetric(tensors)

    def moe_tail(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix: torch.Tensor,
        projection_weight: torch.Tensor,
        *,
        prefix_is_sharded: bool,
        norm_weight: torch.Tensor | None,
        eps: float | None,
    ) -> torch.Tensor | None:
        """Reduce, normalize, and project owned row shards, then gather the output."""
        if self.sharded is None or not self.all_reduce.owns_outputs(
            (routed_partial, shared_partial)
        ):
            return None
        return self.sharded.moe_tail(
            routed_partial,
            shared_partial,
            prefix,
            projection_weight,
            prefix_is_sharded=prefix_is_sharded,
            norm_weight=norm_weight,
            eps=eps,
        )

    def attention_mix(
        self,
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
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Mix owned attention row shards and gather the normalized activation."""
        if self.sharded is None or not self.all_reduce.owns_outputs((partial,)):
            return None
        return self.sharded.attn_mix(
            partial,
            residual,
            block_residual,
            res_weight,
            rms_weight,
            eps=eps,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            num_valid_blocks=num_valid_blocks,
        )

    def attnres_supported(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        """Check the prepared fused AttnRes contract without allocating."""
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.attnres import (
            attnres_supported,
        )

        return (
            self.attnres is not None
            and partial.ndim == 2
            and partial.numel() <= self.attnres.max_numel
            and partial.shape[0] <= self.attnres.max_rows
            and attnres_supported(
                partial,
                residual,
                score_weight,
                output_weight,
                scratch,
                world_size=self.all_reduce.world_size,
                device=self.all_reduce.device,
            )
        )

    def attnres_combine(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized AttnRes output and the accumulated residual."""
        assert self.attnres is not None
        return self.attnres.run(
            partial, residual, score_weight, output_weight, scratch, eps
        )


def prepare_k3_communication(
    *,
    group: dist.ProcessGroup,
    rank_in_group: int,
    staged_max_numel: int,
    producer_direct_max_numel: int,
    attnres_max_numel: int,
    attnres_max_rows: int,
    enable_lamport: bool,
    sharded_max_rows: int,
    dtype: torch.dtype,
    device: torch.device,
    all_reduce: "IrisAllReduce | None",
) -> K3IrisState:
    """Prepare a group's combined K3 buffers before graph capture.

    Capacities are elements except the explicitly named row capacities. Zero
    omits that path. ``all_reduce`` shares an already prepared generic operation,
    or None creates it. The returned state owns all model-specific storage.
    """
    from tokenspeed_kernel.ops.communication._iris import (
        HEAP_HEADROOM_BYTES,
        MIN_HEAP_BYTES,
        IrisGroup,
        get_or_create_context,
    )
    from tokenspeed_kernel.ops.communication._iris.all_reduce import ProducerLamport
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.allreduce_sharded import (
        IrisAllReduceShardedState,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.attnres import (
        IrisAttnResState,
    )
    from tokenspeed_kernel.ops.communication.iris import IrisAllReduce

    world_size = group.size()
    if (
        min(
            staged_max_numel,
            producer_direct_max_numel,
            attnres_max_numel,
            attnres_max_rows,
            sharded_max_rows,
        )
        < 0
    ):
        raise ValueError("Iris communication capacities must be non-negative")
    if bool(attnres_max_numel) != bool(attnres_max_rows):
        raise ValueError(
            "AttnRes element and row capacities must both be zero or non-zero"
        )
    if all_reduce is not None:
        if (
            all_reduce.group is not group
            or all_reduce.dtype != dtype
            or all_reduce.rank_in_group != rank_in_group
            or all_reduce.device != device
            or all_reduce.staged_max_numel < staged_max_numel
            or all_reduce.producer_direct_max_numel < producer_direct_max_numel
        ):
            raise RuntimeError(
                "Prepare K3 communication before the group's first all-reduce"
            )
        producer_direct_max_numel = all_reduce.producer_direct_max_numel
    if sharded_max_rows:
        IrisAllReduceShardedState.validate_capacity(
            sharded_max_rows, world_size, dtype, producer_direct_max_numel
        )
    total_bytes = (
        IrisAllReduce.heap_bytes(
            staged_max_numel, producer_direct_max_numel, world_size, dtype
        )
        if all_reduce is None
        else 0
    )
    if attnres_max_numel:
        total_bytes += IrisAttnResState.heap_bytes(
            attnres_max_numel, attnres_max_rows, world_size, dtype
        )
    if enable_lamport and ProducerLamport.supported(world_size, dtype):
        lamport_numel = min(producer_direct_max_numel // (3584 + 7168), 6) * (
            3584 + 7168
        )
        if lamport_numel:
            total_bytes += ProducerLamport.heap_bytes(lamport_numel)
    sharded_bytes = (
        IrisAllReduceShardedState.heap_bytes(sharded_max_rows, world_size, dtype)
        if sharded_max_rows
        else 0
    )
    # Preserve the base heap's headroom for other collective operations.
    ctx = get_or_create_context(
        heap_size=max(MIN_HEAP_BYTES, total_bytes + HEAP_HEADROOM_BYTES)
        + sharded_bytes,
    )
    if all_reduce is None:
        all_reduce = IrisAllReduce(
            IrisGroup(ctx, group, rank_in_group, device),
            staged_max_numel,
            producer_direct_max_numel,
            dtype,
        )
    state = K3IrisState(
        all_reduce,
        attnres_max_numel=attnres_max_numel,
        attnres_max_rows=attnres_max_rows,
        enable_lamport=enable_lamport,
        sharded_max_rows=sharded_max_rows,
    )
    all_reduce.peers.finish_prepare()
    return state
