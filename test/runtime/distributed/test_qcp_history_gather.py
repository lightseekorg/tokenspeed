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

"""The query-context-parallel history gather, on CPU over gloo.

Four ranks own the pages of a sharded cache group cyclically. Each rank holds
its owned rows of a deterministic reference plane; the gather must rebuild
every request group's history in position order on every rank, with the
per-rank split counted on the host from the page table. Also covered: a rank
that owns no row of a group still joins the collective, the replicated
(``placement=None``) path is a local gather, and the host owner rule agrees
with the kernel's per-rank translation.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import virtual_slots_to_local

from tokenspeed.runtime.layers.attention.dcp.cache import (
    gather_history_rows,
    plan_history_gather,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    cyclic_slot_owner,
    owned_history_rows,
)
from tokenspeed.runtime.layers.attention.page_table import (
    build_prefill_kv_workspace_slots,
)

WORLD = 4
PAGE = 2  # kernel page size
GRANULARITY = 4  # scheduler block: two kernel pages
VIRTUAL_BLOCKS = 1 + 12
DIM = 8


def _reference(virtual_slots: torch.Tensor) -> torch.Tensor:
    """The replicated plane: row v is ``[v, v + 1, ...]`` in bf16."""
    base = virtual_slots.to(torch.float32).unsqueeze(1)
    return (base + torch.arange(DIM, dtype=torch.float32)).to(torch.bfloat16)


def _page_table() -> torch.Tensor:
    """Three requests over distinct scheduler blocks (virtual kernel pages).

    Request 2 uses blocks 9 and 10 only, so rank 0 (owner of blocks 1, 5, 9)
    and rank 1 (owner of 2, 6, 10) hold its rows while ranks 2 and 3 own
    nothing of it.
    """
    blocks = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 0], [9, 10, 0, 0]], dtype=torch.int64
    )
    pages = blocks.unsqueeze(-1) * (GRANULARITY // PAGE) + torch.arange(
        GRANULARITY // PAGE
    )
    pages = torch.where(blocks.unsqueeze(-1) > 0, pages, 0)
    return pages.reshape(3, -1).to(torch.int32)


SEQ_LENS = torch.tensor([15, 11, 7], dtype=torch.int64)


def _placement(rank: int) -> CachePlacement:
    return CachePlacement(
        block_granularity=GRANULARITY,
        virtual_block_count=VIRTUAL_BLOCKS,
        group=tuple(range(WORLD)),
        rank=rank,
    )


def _local_plane(rank: int) -> torch.Tensor:
    """This rank's physical plane: every virtual slot it owns at its local slot."""
    placement = _placement(rank)
    local_pages = 1 + (VIRTUAL_BLOCKS - 1 + WORLD - 1) // WORLD
    plane = torch.full(
        (local_pages * GRANULARITY, DIM), float("nan"), dtype=torch.bfloat16
    )
    every = torch.arange(VIRTUAL_BLOCKS * GRANULARITY, dtype=torch.int64)
    local, owned = virtual_slots_to_local(
        every,
        rows_per_page=GRANULARITY,
        virtual_block_count=VIRTUAL_BLOCKS,
        degree=WORLD,
        rank=rank,
    )
    plane[local[owned]] = _reference(every)[owned]
    return plane


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, errors) -> None:
    try:
        _run(rank, port)
    except Exception:  # pragma: no cover - reported to the parent
        import traceback

        errors[rank] = traceback.format_exc()


def _run(rank: int, port: int) -> None:
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(
        rank=rank, world_size=WORLD, attn_tp_size=WORLD, attn_dcp_size=WORLD
    )
    pg_manager.init_distributed(
        mapping, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    group = mapping.attn.dcp_group
    pg_manager.init_process_group(group, backend="gloo")
    pg_manager.register_process_group(
        "nccl", group, pg_manager.get_process_group("gloo", group)
    )
    global_server_args_dict["force_deterministic_rsag"] = True

    placement = _placement(rank)
    table = _page_table()
    owned = owned_history_rows(table, SEQ_LENS, page_size=PAGE, placement=placement)
    assert owned.shape == (WORLD, 3)
    assert owned.sum(dim=0).tolist() == SEQ_LENS.tolist()
    plane = _local_plane(rank)
    workspace = torch.empty((int(SEQ_LENS.sum()), DIM), dtype=torch.bfloat16)

    # Group A: requests 0 and 1 together; group B: request 2 alone (two
    # owners hold rows, two own nothing and still join the gather).
    for requests in (slice(0, 2), slice(2, 3)):
        rows = int(SEQ_LENS[requests].sum())
        virtual_slots = build_prefill_kv_workspace_slots(
            page_table=table[requests],
            seq_lens=SEQ_LENS[requests],
            max_seq_len=int(SEQ_LENS[requests].max()),
            page_size=PAGE,
            device=torch.device("cpu"),
            num_tokens=rows,
        )
        counts = owned[:, requests].sum(dim=1).tolist()
        plan = plan_history_gather(
            virtual_slots, placement=placement, owned_rows_per_rank=counts
        )
        assert plan.rows == rows
        local = plane.index_select(0, plan.local_fetch_slots)
        assert local.shape[0] == counts[rank]
        assert not torch.isnan(
            local.float()
        ).any(), "fetched a row this rank does not own"
        out = gather_history_rows(plan, local, out=workspace)
        torch.testing.assert_close(out, _reference(virtual_slots), rtol=0, atol=0)
        if requests == slice(2, 3):
            assert counts[2] == 0 and counts[3] == 0

    dist.barrier()
    dist.destroy_process_group()


def test_history_gather_rebuilds_every_group_in_position_order():
    port = _get_open_port()
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(port, errors), nprocs=WORLD, join=True)
    if errors:
        raise RuntimeError("\n".join(f"rank {r}: {e}" for r, e in errors.items()))


def test_replicated_group_gathers_locally_without_a_collective():
    table = _page_table()
    owned = owned_history_rows(table, SEQ_LENS, page_size=PAGE, placement=None)
    assert owned.tolist() == [SEQ_LENS.tolist()]
    rows = int(SEQ_LENS.sum())
    virtual_slots = build_prefill_kv_workspace_slots(
        page_table=table,
        seq_lens=SEQ_LENS,
        max_seq_len=int(SEQ_LENS.max()),
        page_size=PAGE,
        device=torch.device("cpu"),
        num_tokens=rows,
    )
    plan = plan_history_gather(
        virtual_slots, placement=None, owned_rows_per_rank=[rows]
    )
    assert plan.group == (0,) and torch.equal(plan.local_fetch_slots, virtual_slots)
    plane = _reference(torch.arange(VIRTUAL_BLOCKS * GRANULARITY))
    out = gather_history_rows(
        plan,
        plane.index_select(0, plan.local_fetch_slots),
        out=torch.empty((rows, DIM), dtype=torch.bfloat16),
    )
    torch.testing.assert_close(out, _reference(virtual_slots), rtol=0, atol=0)


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_host_owner_rule_agrees_with_the_kernel_translation(degree):
    placement = CachePlacement(
        block_granularity=GRANULARITY,
        virtual_block_count=VIRTUAL_BLOCKS,
        group=tuple(range(degree)),
        rank=0,
    )
    slots = torch.arange(-3, (VIRTUAL_BLOCKS + 2) * GRANULARITY, dtype=torch.int64)
    owner = cyclic_slot_owner(slots, placement)
    for rank in range(degree):
        _local, owned = virtual_slots_to_local(
            slots,
            rows_per_page=GRANULARITY,
            virtual_block_count=VIRTUAL_BLOCKS,
            degree=degree,
            rank=rank,
        )
        assert torch.equal(owner == rank, owned)
    unowned = (slots < GRANULARITY) | (slots >= VIRTUAL_BLOCKS * GRANULARITY)
    assert torch.equal(owner == -1, unowned)
    assert cyclic_slot_owner(slots, None).eq(0).all()


def test_owned_history_rows_refuses_holes_below_the_length():
    table = _page_table().clone()
    table[0, 1] = 0  # a hole inside request 0's 15 rows
    with pytest.raises(ValueError, match="holes"):
        owned_history_rows(table, SEQ_LENS, page_size=PAGE, placement=_placement(0))
    with pytest.raises(ValueError, match="does not cover"):
        owned_history_rows(
            _page_table(), torch.tensor([99, 1, 1]), page_size=PAGE, placement=None
        )


def test_gather_plan_checks_its_counts():
    virtual_slots = torch.arange(GRANULARITY, 3 * GRANULARITY, dtype=torch.int64)
    with pytest.raises(ValueError, match="one owner"):
        plan_history_gather(virtual_slots, placement=None, owned_rows_per_rank=[4, 4])
    with pytest.raises(ValueError, match="sum to"):
        plan_history_gather(
            virtual_slots, placement=_placement(0), owned_rows_per_rank=[1, 1, 1, 1]
        )
