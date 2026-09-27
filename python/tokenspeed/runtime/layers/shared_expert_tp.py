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

"""Shared-expert intermediate-channel sharding with local token ownership."""

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.trtllm import (
    TrtllmAllGatherState,
    TrtllmReduceScatterState,
    trtllm_allgather,
    trtllm_reduce_scatter,
)

from tokenspeed.runtime.distributed.comm_ops import all_gather_single, reduce_scatter
from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)


def shared_expert_mapping(mapping, value):
    """Validate Kimi shared-expert TP independently of attention and routed EP."""
    try:
        size = int(value)
    except ValueError as exc:
        raise ValueError("Shared-expert TP size must be a positive integer") from exc
    if size == 1:
        return None
    if size < 1 or size >= mapping.world_size or mapping.world_size % size:
        raise ValueError(
            "Shared-expert TP must divide world size and be smaller than it"
        )
    if (
        mapping.attn.dp_size != mapping.world_size
        or mapping.attn.tp_size != 1
        or mapping.linear_attn.tp_size != 1
        or mapping.moe.ep_size != mapping.world_size
        or mapping.moe.tp_size != 1
        or mapping.pp_size != 1
    ):
        raise ValueError(
            "Shared-expert TP requires attention TP1/DPworld, MoE TP1/EPworld, PP1"
        )
    return DenseLayerMapping(
        rank=mapping.rank,
        world_size=mapping.world_size,
        tp_size=size,
        dp_size=mapping.world_size // size,
    )


def validate_shared_expert_settings(mapping, value):
    """Agree on raw settings world-wide before parsing or creating subgroups.

    Disabled and malformed settings must participate too: a rank-local return
    could otherwise leave enabled peers blocked in communicator construction.
    Returns the shared-expert mapping, or None when every rank disables TP.
    """
    if dist.is_initialized() and mapping.world_size > 1:
        pg_manager.init_process_group(mapping.world_group, backend="gloo")
        values = [None] * mapping.world_size
        dist.all_gather_object(
            values,
            value,
            group=pg_manager.get_process_group("gloo", mapping.world_group),
        )
        if len(set(values)) != 1:
            raise ValueError(f"Shared-expert TP settings differ across ranks: {values}")
    return shared_expert_mapping(mapping, value)


def initialize_shared_expert_group(parallel):
    """Initialize and warm shared AllGather/ReduceScatter before graph capture."""
    pg_manager.init_process_group(parallel.tp_group, backend=None)
    probe = torch.zeros((1, 1), dtype=torch.bfloat16, device="cuda")
    received = torch.empty(
        (parallel.tp_size, 1), dtype=probe.dtype, device=probe.device
    )
    all_gather_single(received, probe, parallel.tp_group, backend=None)
    reduce_scatter(received, parallel.tp_group, backend=None)


class SharedExpertCommunication:
    """Shared-expert collectives and scratch reused by sequential model layers.

    Allocate before memory profiling/capture; never alias main-stream attention
    buffers. Physical subgroup counts select the same path on every peer.
    """

    def __init__(self, parallel, capacity, hidden, device):
        self.parallel, self.capacity, self.hidden = parallel, capacity, hidden
        self.send = torch.empty(capacity, hidden, dtype=torch.bfloat16, device=device)
        self.received = torch.empty(
            parallel.tp_size * capacity, hidden, dtype=torch.bfloat16, device=device
        )
        self.gather = self.reduction = None
        # Native kernels support TP2/4/8/16; the gather wrapper requires
        # 128-aligned widths. Other geometries use NCCL.
        if parallel.tp_size in (2, 4, 8, 16) and hidden > 0 and hidden % 128 == 0:
            group = pg_manager.get_process_group("nccl", parallel.tp_group)
            self.gather = TrtllmAllGatherState(
                group, min(capacity, 128), hidden, device, True
            )
            self.reduction = TrtllmReduceScatterState(
                group, min(capacity, 128), hidden, device
            )
            probe = self.reduction.input_buffer(1)
            probe.zero_()
            trtllm_reduce_scatter(self.reduction, probe, 1)

    def gather_inputs(self, inputs, counts):
        """Return borrowed padded subgroup inputs before forking MLP compute.

        inputs is local BF16 [tokens,H]; counts contains physical world rows.
        Finish consuming this view before the next gather. It may overlap
        local routing on aux, but must finish before routed-MoE communication.
        """
        p = self.parallel
        if (
            counts is None
            or len(counts) != p.world_size
            or any(n < 0 for n in counts)
            or inputs.shape != (counts[p.rank], self.hidden)
            or inputs.dtype != torch.bfloat16
        ):
            raise ValueError(
                "Shared-expert TP requires matching BF16 inputs and world counts"
            )
        rows = max(counts[r] for r in p.tp_group)
        if rows > self.capacity:
            raise ValueError("Shared-expert TP exceeds prepared capacity")
        if rows == 0:
            return inputs.new_empty((0, self.hidden))
        local_rows = inputs.shape[0]
        if local_rows == rows and inputs.is_contiguous():
            send = inputs
        else:
            send = self.send[:rows]
            send.zero_()
            send[:local_rows].copy_(inputs)
        if self.gather is not None and rows <= 128:
            gathered = trtllm_allgather(self.gather, send)
        else:
            gathered = self.received[: p.tp_size * rows]
            all_gather_single(gathered, send, p.tp_group, backend=None)
        return gathered

    def reduce_outputs(self, partial, local_rows):
        """Restore owned rows after shared compute and routed dispatch finish.

        partial is BF16 [TP*padded_rows,H]; local_rows is this owner's valid
        count. Empty subgroups skip the collective; empty owners still join.
        Reduction may overlap finite routed BMM, but must finish before combine.
        """
        rows = partial.shape[0] // self.parallel.tp_size
        if (
            partial.shape != (self.parallel.tp_size * rows, self.hidden)
            or not 0 <= local_rows <= rows
        ):
            raise ValueError("Shared-expert partials must match padded subgroup rows")
        if rows == 0:
            return partial.new_empty((0, self.hidden))
        if self.reduction is not None and rows <= 128:
            owned = trtllm_reduce_scatter(self.reduction, partial, rows)
        else:
            owned = reduce_scatter(partial, self.parallel.tp_group, backend=None)
        return owned[:local_rows]

    def close(self):
        """Release collectively only after graphs and stream consumers finish."""
        if self.gather is not None:
            self.gather.close()
        if self.reduction is not None:
            self.reduction.close()
