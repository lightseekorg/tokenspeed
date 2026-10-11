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

"""KV-page sharding and the retraction image's ops, across ranks on gloo.

Under ``--kv-parallel-size D`` a group's virtual block ``v > 0``
belongs to rank ``(v - 1) % D`` on the Device and on every Host tier (the
scheduler allocates the Host block in its Device block's residue class). Each
rank adapts the same wire batch (``cache_ops_from_plan``) and keeps, through
the one ownership translation (``BlockOwnerTranslation``), only the rows it
owns on both ends. Four ranks assert, with the kept positions gathered over
gloo: every kept row's Device id and Host id translate to this rank, the
sharded group's rows are covered exactly once across the ranks, and a
replicated group's rows are kept by every rank -- for a ``SnapshotOp`` (Device
to pool) and a ``RestoreOp`` reading both Host tiers.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.cache.transfer.ops import (  # noqa: E402
    HostTier,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.cache.transfer.ownership import (  # noqa: E402
    BlockOwnerTranslation,
)
from tokenspeed.runtime.engine import scheduler_utils  # noqa: E402

WORLD = 4
DEVICE_LCM_BLOCKS = 4
HOST_LCM_BLOCKS = 6
POOL_LCM_BLOCKS = 3
# Group 0 is page-cyclic over the four ranks, group 1 replicated; both pack
# four CacheBlocks per LCM block, so a sharded group's virtual packing is 16.
SHARD_COUNTS = (WORLD, 1)
PACKING = 4


def _layout():
    return SimpleNamespace(
        num_lcm_blocks=DEVICE_LCM_BLOCKS,
        groups=(
            SimpleNamespace(group_id="sharded", cache_blocks_per_lcm_block=PACKING),
            SimpleNamespace(group_id="replicated", cache_blocks_per_lcm_block=PACKING),
        ),
    )


def _contract(layout):
    return SimpleNamespace(
        group_specs=tuple(
            SimpleNamespace(group_id=group.group_id, shard_count=shard)
            for group, shard in zip(layout.groups, SHARD_COUNTS)
        ),
        virtual_block_counts={
            group.group_id: 1 + DEVICE_LCM_BLOCKS * PACKING * shard
            for group, shard in zip(layout.groups, SHARD_COUNTS)
        },
    )


def _owner(block: int, shard: int) -> int:
    return (block - 1) % shard


def _equal_residue_rows(shard: int, host_bound: int, *, count: int, seed: int):
    """``count`` (device, host) pairs of equal residue, one per owner in turn."""
    rows = []
    for i in range(count):
        device = 1 + (seed + i) % (DEVICE_LCM_BLOCKS * PACKING * shard)
        # The lowest Host block of the same residue above ``seed``.
        host = 1 + ((seed * 3 + i * shard) % (host_bound - 1))
        host += (_owner(device, shard) - _owner(host, shard)) % shard
        if host >= host_bound:
            host -= shard
        assert _owner(device, shard) == _owner(host, shard)
        rows.append((device, host))
    return rows


def _wire_batches():
    """One SnapshotOp and one RestoreOp wire batch, both groups, equal residues."""
    pool_bound = 1 + POOL_LCM_BLOCKS * PACKING * WORLD
    l2_bound = 1 + HOST_LCM_BLOCKS * PACKING * WORLD
    rep_pool_bound = 1 + POOL_LCM_BLOCKS * PACKING
    sharded_tail = _equal_residue_rows(WORLD, pool_bound, count=7, seed=5)
    replicated_tail = _equal_residue_rows(1, rep_pool_bound, count=3, seed=2)
    store_groups = [0] * len(sharded_tail) + [1] * len(replicated_tail)
    store = SimpleNamespace(
        op_ids=[11],
        request_ids=["victim"],
        request_pool_indices=[3],
        snapshot_slots=[0],
        group_ids=[store_groups],
        src_pages=[[d for d, _ in sharded_tail] + [d for d, _ in replicated_tail]],
        dst_pages=[[h for _, h in sharded_tail] + [h for _, h in replicated_tail]],
    )
    sharded_l2 = _equal_residue_rows(WORLD, l2_bound, count=5, seed=9)
    restore_rows = (
        [(0, h, d, HostTier.L2) for d, h in sharded_l2]
        + [(0, h, d, HostTier.SNAPSHOT_POOL) for d, h in sharded_tail]
        + [(1, h, d, HostTier.SNAPSHOT_POOL) for d, h in replicated_tail]
    )
    restore = SimpleNamespace(
        op_ids=[12],
        request_ids=["victim"],
        request_pool_indices=[6],
        snapshot_slots=[0],
        group_ids=[[g for g, _, _, _ in restore_rows]],
        src_pages=[[h for _, h, _, _ in restore_rows]],
        dst_pages=[[d for _, _, d, _ in restore_rows]],
        content_hashes=[
            [f"k{d}" if tier is HostTier.L2 else "" for _, _, d, tier in restore_rows]
        ],
        page_offsets=[[0] * len(restore_rows)],
        source_tiers=[[int(tier) for _, _, _, tier in restore_rows]],
    )
    return store, restore


class _StoreBatch(SimpleNamespace):
    pass


class _RestoreBatch(SimpleNamespace):
    pass


def _adapted_ops():
    """The two per-request ops every rank derives from the same wire plan."""
    store, restore = _wire_batches()
    store = _StoreBatch(**vars(store))
    restore = _RestoreBatch(**vars(restore))
    # The C++ op bindings have no Python constructor: dispatch on the fakes.
    fake_cache = SimpleNamespace(
        WriteBackOp=(), LoadBackOp=(), SnapshotOp=_StoreBatch, RestoreOp=_RestoreBatch
    )
    with patch.object(scheduler_utils, "Cache", fake_cache):
        ops = scheduler_utils.cache_ops_from_plan(
            SimpleNamespace(cache=[store, restore])
        )
    (snapshot_op,) = [op for op in ops if isinstance(op, SnapshotOp)]
    (restore_op,) = [op for op in ops if isinstance(op, RestoreOp)]
    return snapshot_op, restore_op


def _kept_positions(owners: BlockOwnerTranslation, rows, rank: int):
    """This rank's kept positions, asserting both ends translate to ``rank``."""
    kept = owners.owned_positions(rows)
    for position, (group, local_device, local_host) in kept:
        _, device, host = rows[position]
        shard = SHARD_COUNTS[group]
        assert _owner(device, shard) == rank % shard
        assert _owner(host, shard) == rank % shard
        assert local_device >= 1 and local_host >= 1
    return [position for position, _ in kept]


def _worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD)
    try:
        layout = _layout()
        contract = _contract(layout)
        pool_owners = BlockOwnerTranslation.for_host_pool(
            layout, contract, num_host_lcm_blocks=POOL_LCM_BLOCKS, rank=rank
        )
        l2_owners = BlockOwnerTranslation.for_host_pool(
            layout, contract, num_host_lcm_blocks=HOST_LCM_BLOCKS, rank=rank
        )
        snapshot_op, restore_op = _adapted_ops()

        # The store: Device source, pool destination.
        store_rows = [
            (t.group_id, t.source_page, t.destination_page)
            for t in snapshot_op.transfers
        ]
        store_kept = _kept_positions(pool_owners, store_rows, rank)
        # The restore: Host source per tier, Device destination.
        restore_kept = []
        for tier, owners in (
            (HostTier.L2, l2_owners),
            (HostTier.SNAPSHOT_POOL, pool_owners),
        ):
            positions = [i for i, t in enumerate(restore_op.source_tier) if t is tier]
            rows = [
                (
                    restore_op.transfers[i].group_id,
                    restore_op.transfers[i].destination_page,
                    restore_op.transfers[i].source_page,
                )
                for i in positions
            ]
            restore_kept.extend(
                positions[k] for k in _kept_positions(owners, rows, rank)
            )

        gathered: list = [None] * WORLD
        dist.all_gather_object(gathered, (store_kept, sorted(restore_kept)))

        for op, rows_fn, index in (
            (snapshot_op, lambda op: op.transfers, 0),
            (restore_op, lambda op: op.transfers, 1),
        ):
            transfers = rows_fn(op)
            counts = [0] * len(transfers)
            for per_rank in gathered:
                for position in per_rank[index]:
                    counts[position] += 1
            for position, transfer in enumerate(transfers):
                expected = 1 if SHARD_COUNTS[transfer.group_id] == WORLD else WORLD
                assert counts[position] == expected, (
                    f"op {op.op_id} row {position} ({transfer}) kept by "
                    f"{counts[position]} ranks, expected {expected}"
                )
        # Every rank adapted the same ops from the same plan.
        ops_by_rank: list = [None] * WORLD
        dist.all_gather_object(ops_by_rank, (snapshot_op, restore_op))
        assert all(pair == (snapshot_op, restore_op) for pair in ops_by_rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_every_rank_keeps_its_owned_rows_and_the_ranks_cover_each_op_once(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rv").as_uri(),), nprocs=WORLD, join=True)


def test_a_pair_of_different_owners_is_refused():
    """The scheduler's equal-residue invariant is asserted, not assumed."""
    layout = _layout()
    owners = BlockOwnerTranslation.for_host_pool(
        layout, _contract(layout), num_host_lcm_blocks=POOL_LCM_BLOCKS, rank=0
    )
    with pytest.raises(ValueError, match="residue class"):
        owners.owned_positions([(0, 1, 2)])
