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

"""Ascend V4.1 DP row layouts using the existing LongCat RSAG helpers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EngramRowExchange:
    """Gather hashes across DP replicas before a world-sharded table lookup.

    Only IDs cross the DP group. The embedding's ordinary TP all-reduce then
    returns exact BF16 table rows, from which each replica takes its own slice.
    Row counts come from the same live/padded forward table as MoE collectives.
    """

    rank: int
    world_size: int
    attn_tp_size: int
    dp_rank: int
    dp_group: tuple[int, ...]

    def counts(self, table, local_rows):
        if table is None or len(table) != self.world_size:
            raise ValueError("Shared Engram tables require the forward DP row table")
        if table[self.rank] != local_rows or any(n < 0 for n in table):
            raise ValueError("Shared Engram row counts disagree with local hashes")
        counts = []
        for first in range(0, self.world_size, self.attn_tp_size):
            cohort = table[first : first + self.attn_tp_size]
            if len(set(cohort)) != 1:
                raise ValueError("Engram attention TP peers disagree on row counts")
            counts.append(cohort[0])
        return counts

    def gather(self, hashes, table):
        from tokenspeed.runtime.distributed.comm_ops import token_all_gather_rows

        counts = self.counts(table, hashes.shape[0])
        gathered = token_all_gather_rows(hashes, self.dp_group, counts)
        offset = sum(counts[: self.dp_rank])
        return gathered, offset


def resolve_engram_table_mapping(mapping, requested, *, device_type, host_table, role):
    """Resolve explicit table sharing without changing attention/projection TP."""
    if requested is None or requested == mapping.attn.tp_size:
        return mapping, None
    if (
        requested != mapping.world_size
        or device_type != "npu"
        or host_table
        or role != "decode"
        or mapping.pp_size != 1
        or not mapping.attn.has_dp
        or mapping.attn.has_qcp
    ):
        raise ValueError(
            "Wider Engram TP requires Ascend V4.1 PD decode, device tables, "
            "PP1, attention DP without QCP, and engram_tp_size=world_size"
        )
    from tokenspeed.runtime.distributed.mapping import Mapping

    table_mapping = Mapping(
        rank=mapping.rank,
        world_size=mapping.world_size,
        attn_tp_size=requested,
        attn_dp_size=1,
    )
    exchange = EngramRowExchange(
        mapping.rank,
        mapping.world_size,
        mapping.attn.tp_size,
        mapping.attn.dp_rank,
        mapping.attn.dp_group,
    )
    return table_mapping, exchange


def supports_ascend_dp_moe(mapping, backend, is_npu):
    return (
        is_npu
        and backend in ("ascend_w4a4", "ascend_bf16")
        and mapping.attn.dp_size > 1
        and mapping.dense.tp_size in (mapping.attn.tp_size, 4, 8)
        and mapping.dense.tp_size >= mapping.attn.tp_size
        and mapping.attn.tp_size <= 8
        and mapping.moe.tp_size == 1
        and mapping.moe.ep_size == mapping.world_size
        and mapping.pp_size == 1
        and not mapping.attn.has_qcp
    )


def validate_replica_counts(counts, mapping, local_rows):
    if len(counts) != mapping.world_size or counts[mapping.rank] != local_rows:
        raise ValueError("V4.1 decoder DP row table disagrees with local rows")
    tp = mapping.attn.tp_size
    if any(len(set(counts[i : i + tp])) != 1 for i in range(0, len(counts), tp)):
        raise ValueError("V4.1 attention TP ranks disagree on decoder row counts")
    if any(x < 0 for x in counts):
        raise ValueError("V4.1 decoder row counts cannot be negative")
    return counts


def create_decoder_cpu_group(mapping):
    """Create a forward-only group before scheduler and forward threads start.

    Scheduler metadata collectives can overlap narrowed decoder-row exchange.
    They must not share a Gloo sequence space, even with identical rank sets.
    Every model rank calls this once during construction, in the same order.
    """
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    process_group_manager.init_process_group(mapping.attn.dp_group, backend="hccl")
    return torch.distributed.new_group(
        ranks=list(range(mapping.world_size)), backend="gloo"
    )


def decoder_global_rows(ctx, mapping, group):
    """Exchange narrowed prefill counts once; decode uses its existing table.

    The predicate is global scheduler metadata, so an idle/decode DP replica
    enters the same branch as a peer doing prefill. Decode graph warmup/capture
    explicitly sets all_decode_or_idle and uses padded token counts.
    """
    # Idle forwards skip attention metadata refresh; decoder_view may still
    # hold the final captured batch. They contribute zero actual rows.
    rows = (
        0
        if ctx.input_num_tokens == 0
        else int(ctx.attn_backend.decoder_view().metadata.positions.numel())
    )
    counts = ctx.global_num_tokens
    if counts is None:
        raise ValueError("Ascend DP MoE needs scheduler global token counts")
    if ctx.all_decode_or_idle:
        return validate_replica_counts(list(counts), mapping, rows)
    if group is None:
        raise ValueError("Ascend DP decoder row exchange needs its own CPU group")
    local = torch.tensor([rows], dtype=torch.int64, device="cpu")
    gathered = [torch.empty_like(local) for _ in range(mapping.world_size)]
    torch.distributed.all_gather(gathered, local, group=group)
    return validate_replica_counts([int(x.item()) for x in gathered], mapping, rows)


def replicated_rows_moe(x, image_mask, ctx, ffn, comm_manager):
    """Keep mHC rows local/full while routed experts consume EP-global rows."""
    if image_mask is not None:
        raise NotImplementedError(
            "Ascend V4.1 DP MoE currently requires text-only rows"
        )
    if comm_manager.moe_combine_order != "rank":
        raise ValueError("Ascend V4.1 DP MoE requires rank-ordered partial reduction")
    # Attention TP already replicates every local DP row. Exchange whole
    # replicas across corresponding TP ranks rather than re-gathering our
    # own rows through the global EP group. Metadata counts are static for
    # a captured graph and follow the original CED/padding row contract.
    from tokenspeed.runtime.distributed.comm_ops import token_all_gather

    mapping = comm_manager.mapping
    if ctx.query_shard is not None and ctx.query_shard.size != 1:
        raise ValueError("Replicated DP MoE cannot receive query-sharded rows")
    scattered_counts = comm_manager.scattered_num_tokens(ctx)
    tp = mapping.attn.tp_size
    dp_counts = [
        sum(scattered_counts[i : i + tp]) for i in range(0, len(scattered_counts), tp)
    ]
    if (
        len(dp_counts) != mapping.attn.dp_size
        or x.shape[0] != dp_counts[mapping.attn.dp_rank]
    ):
        raise ValueError("Replicated DP MoE input does not match decoder row counts")
    gathered = token_all_gather(
        x, group=mapping.attn.dp_group, scattered_num_tokens=dp_counts
    )
    total, maximum = comm_manager.get_num_tokens(ctx)
    partial = ffn(gathered, None, total, maximum, ctx=None, comm_manager=None)
    scattered, _ = comm_manager.post_mlp_comm(partial, None, ctx)
    return comm_manager.gather_scattered_rows(scattered, ctx)


@dataclass(frozen=True)
class V41OutputPartition:
    """Canonical grouped-output weight shard and its replicated TP cohort."""

    tp_rank: int
    tp_size: int
    tp_group: tuple[int, ...]


def grouped_output_partition(mapping, output_groups):
    """Collect only the neighboring head shards belonging to one WO-A group.

    For TP16/groups8, ranks(0,1) share output group0, ranks(2,3) group1, etc.
    Even/odd cohorts hold the eight canonical weight shards. Only the first
    rank in each head group contributes its projection to the final reduction.
    No checkpoint values, group ordering, or per-group GEMM dimensions change.
    """
    attn = mapping.attn
    if (
        attn.dp_size != 1
        or attn.tp_size <= output_groups
        or attn.tp_size % output_groups
    ):
        raise ValueError("Grouped output exchange requires DP1 and divisible wide TP")
    width = attn.tp_size // output_groups
    rank = attn.tp_rank
    first = (rank // width) * width
    heads = tuple(attn.tp_group[first : first + width])
    cohort = tuple(attn.tp_group[rank % width :: width])
    return V41OutputPartition(rank // width, output_groups, cohort), heads
