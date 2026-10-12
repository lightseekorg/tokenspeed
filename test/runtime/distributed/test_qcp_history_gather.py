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
that owns no row of a group still joins the collective, non-bf16 rows (packed
index-K bytes, fp32 scales, fp8 latent) travel through a bf16-only gather
backend byte-identically, the replicated (``placement=None``) path is a local
gather, the host owner rule agrees with the kernel's per-rank translation, and
the DSA leaf's index-K history gather rebuilds the plane in its own format
(``fp8_scaled`` bytes and scales, or ``bf16`` keys).
"""

from __future__ import annotations

import socket
from test.runtime.dsa_index_k_test_utils import (
    expected_index_k_rows,
    index_k_pool,
    write_index_k_plane,
)

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


class _Bf16OnlyBackend:
    """The production low-latency token all-gather moves bf16 rows only
    (``TritonRSAGBackend`` asserts the dtype); this stand-in enforces that
    and delegates the data movement to the NCCL-style padded gather."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.gathered_dtypes: list[torch.dtype] = []

    def token_all_gather(self, tensor, group, scattered_num_tokens):
        assert tensor.dtype == torch.bfloat16, f"RSAG gather of {tensor.dtype}"
        self.gathered_dtypes.append(tensor.dtype)
        return self.delegate.token_all_gather(tensor, group, scattered_num_tokens)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


def _run(rank: int, port: int) -> None:
    from tokenspeed.runtime.distributed.comm_backend import registry
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(
        rank=rank, world_size=WORLD, attn_tp_size=WORLD, attn_kvp_size=WORLD
    )
    pg_manager.init_distributed(
        mapping, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    group = mapping.attn.kvp_group
    pg_manager.init_process_group(group, backend="gloo")
    pg_manager.register_process_group(
        "nccl", group, pg_manager.get_process_group("gloo", group)
    )
    # Not the deterministic NCCL route: every history gather must be
    # dtype-safe for the bf16-only low-latency solution.
    global_server_args_dict["force_deterministic_rsag"] = False
    backend = _Bf16OnlyBackend(NcclBackend())
    registry._global_backend = backend

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

        # The non-bf16 payloads of the indexer (packed uint8 index-K rows,
        # fp32 scales) and an fp8 latent travel as bf16 pairs of their bytes
        # and land byte-identical.
        reference = _reference(virtual_slots)
        for dtype, width in (
            (torch.uint8, 132),
            (torch.float32, 3),
            (torch.float8_e4m3fn, DIM),
        ):
            full = _typed_rows(reference, dtype, width)
            local_typed = _typed_rows(local, dtype, width)
            out = gather_history_rows(
                plan,
                local_typed,
                out=torch.empty((rows + 3, width), dtype=dtype),
            )
            assert out.dtype == dtype and out.shape == (rows, width)
            assert torch.equal(out.view(torch.uint8), full.view(torch.uint8))
        with pytest.raises(ValueError, match="even byte width"):
            gather_history_rows(
                plan,
                torch.zeros((counts[rank], 3), dtype=torch.uint8),
                out=torch.empty((rows, 3), dtype=torch.uint8),
            )
    assert backend.gathered_dtypes and set(backend.gathered_dtypes) == {torch.bfloat16}

    dist.barrier()
    dist.destroy_process_group()


def _typed_rows(rows: torch.Tensor, dtype: torch.dtype, width: int) -> torch.Tensor:
    """Rows of ``dtype`` whose bytes derive from ``rows``' bf16 bytes: the
    row's byte pattern repeated to ``width`` elements."""
    raw = rows.contiguous().view(torch.uint8)
    width_bytes = width * torch.tensor([], dtype=dtype).element_size()
    repeats = -(-width_bytes // raw.shape[1])
    return raw.repeat(1, repeats)[:, :width_bytes].contiguous().view(dtype)


def test_history_gather_rebuilds_every_group_in_position_order():
    port = _get_open_port()
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(port, errors), nprocs=WORLD, join=True)
    if errors:
        raise RuntimeError("\n".join(f"rank {r}: {e}" for r, e in errors.items()))


# --- the DSA leaf's index-K history gather, per plane format ------------------

INDEX_HEAD_DIM = 128


def _index_k_keys() -> torch.Tensor:
    """The replicated index-K history: virtual slot ``v`` holds the key
    ``v + i / 4`` (``i`` the element), in bf16."""
    v = torch.arange(VIRTUAL_BLOCKS * GRANULARITY, dtype=torch.float32).unsqueeze(1)
    return (v + torch.arange(INDEX_HEAD_DIM, dtype=torch.float32) / 4).to(
        torch.bfloat16
    )


def _local_index_k_planes(index_k_format: str) -> list[torch.Tensor]:
    """Every rank's physical index-K plane: the virtual slots it owns, written
    at their local slots through the pool's production write path (built in
    the parent, so the gloo workers stay host-only)."""
    local_pages = 1 + (VIRTUAL_BLOCKS - 1 + WORLD - 1) // WORLD
    every = torch.arange(VIRTUAL_BLOCKS * GRANULARITY, dtype=torch.int64)
    keys = _index_k_keys()
    planes = []
    for rank in range(WORLD):
        local, owned = virtual_slots_to_local(
            every,
            rows_per_page=GRANULARITY,
            virtual_block_count=VIRTUAL_BLOCKS,
            degree=WORLD,
            rank=rank,
        )
        planes.append(
            write_index_k_plane(
                index_k_format,
                head_dim=INDEX_HEAD_DIM,
                page_size=GRANULARITY,
                slots=local_pages * GRANULARITY,
                loc=local[owned],
                keys=keys[owned],
            )
        )
    return planes


def _dsa_leaf(rank: int, index_k_format: str, *, max_model_len: int):
    """A GPU DSA leaf over the sharded cache, with the fields the index-K
    history gather reads (no device, no kernels)."""
    from tokenspeed.runtime.layers.attention.backends.paged import dsa

    backend = object.__new__(dsa.DSABackend)
    backend.kernel_page_size = PAGE
    backend.device = "cpu"
    backend.data_type = torch.bfloat16
    backend.kv_cache_dim = DIM
    backend.index_head_dim = INDEX_HEAD_DIM
    backend.index_k_format = index_k_format
    backend.kvp_group = tuple(range(WORLD))
    backend.kvp_rank = rank
    backend.kvp_block_granularity = GRANULARITY
    backend.kvp_virtual_block_count = VIRTUAL_BLOCKS
    backend._history_workspace = None
    backend.preallocate_history_gather_workspace(max_model_len)
    return backend


def _run_index_k(
    rank: int,
    port: int,
    index_k_format: str,
    plane: torch.Tensor,
    expected: tuple[torch.Tensor, torch.Tensor | None],
) -> None:
    from tokenspeed.runtime.distributed.comm_backend import registry
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.dsa import (
        QueryShardHistoryGroup,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(
        rank=rank, world_size=WORLD, attn_tp_size=WORLD, attn_kvp_size=WORLD
    )
    pg_manager.init_distributed(
        mapping, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    group = mapping.attn.kvp_group
    pg_manager.init_process_group(group, backend="gloo")
    pg_manager.register_process_group(
        "nccl", group, pg_manager.get_process_group("gloo", group)
    )
    global_server_args_dict["force_deterministic_rsag"] = False
    backend_comm = _Bf16OnlyBackend(NcclBackend())
    registry._global_backend = backend_comm

    leaf = _dsa_leaf(rank, index_k_format, max_model_len=int(SEQ_LENS.sum()))
    pool = index_k_pool(plane, head_dim=INDEX_HEAD_DIM, page_size=GRANULARITY)
    expected_keys, expected_scale = expected
    placement = leaf.cache_placement(None)
    assert placement == _placement(rank)
    table = _page_table()
    owned = owned_history_rows(table, SEQ_LENS, page_size=PAGE, placement=placement)
    row_base = 0
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
        group_ = QueryShardHistoryGroup(
            requests=requests,
            row_base=row_base,
            rows=rows,
            local_query=slice(0, 0),
            gather=plan_history_gather(
                virtual_slots,
                placement=placement,
                owned_rows_per_rank=owned[:, requests].sum(dim=1).tolist(),
            ),
        )
        keys, scale = leaf.gather_history_index_k(0, pool, group_)
        workspace = leaf.history_gather_workspace()
        assert keys.shape == (rows, INDEX_HEAD_DIM)
        assert keys.data_ptr() == workspace.index_k.data_ptr()
        assert torch.equal(keys, expected_keys[virtual_slots])
        if index_k_format == "fp8_scaled":
            assert keys.dtype == torch.uint8 and scale.dtype == torch.float32
            assert torch.equal(scale, expected_scale[virtual_slots])
        else:
            assert keys.dtype == torch.bfloat16 and scale is None
        row_base += rows
    assert set(backend_comm.gathered_dtypes) == {torch.bfloat16}

    dist.barrier()
    dist.destroy_process_group()


def _index_k_worker(
    rank: int, port: int, index_k_format: str, planes, expected, errors
):
    try:
        _run_index_k(rank, port, index_k_format, planes[rank], expected)
    except Exception:  # pragma: no cover - reported to the parent
        import traceback

        errors[rank] = traceback.format_exc()


@pytest.mark.parametrize("index_k_format", ["fp8_scaled", "bf16"])
def test_the_dsa_leaf_gathers_the_index_k_history_in_the_planes_format(
    index_k_format,
):
    """``DSABackend.gather_history_index_k`` over page-sharded planes rebuilds
    every group's index-K history in position order on every rank in the
    plane's own format -- the bytes the pool's write path stored: FP8 bytes
    plus fp32 scales, or bf16 keys with no scale (the
    ``index_k_fp8``/``index_k_scale`` and ``index_k_bf16`` rows of
    ``dsa_prefill_topk``) -- through the bf16-only collective."""
    planes = _local_index_k_planes(index_k_format)
    expected = expected_index_k_rows(index_k_format, _index_k_keys())
    port = _get_open_port()
    errors = mp.Manager().dict()
    mp.spawn(
        _index_k_worker,
        args=(port, index_k_format, planes, expected, errors),
        nprocs=WORLD,
        join=True,
    )
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
