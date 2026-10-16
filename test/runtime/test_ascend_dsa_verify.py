# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Ascend DSA gives each verify candidate its own causal DCP metadata."""

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry

from tokenspeed.runtime.layers.attention import backends
from tokenspeed.runtime.layers.attention.backends.paged import ascend_dsa
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.registry import _BACKEND_REGISTRY


def test_ascend_backend_reuses_the_common_dsa_base():
    assert issubclass(ascend_dsa.AscendDSABackend, DSABackend)


def test_both_dsa_modules_are_imported():
    assert backends.ascend_dsa is ascend_dsa
    assert backends.dsa.DSABackend is DSABackend
    expected = ascend_dsa.AscendDSABackend if current_platform().is_npu else DSABackend
    assert _BACKEND_REGISTRY["dsa"][1] is expected


def test_ascend_mla_prologue_supports_native_prefill_and_decode():
    spec = KernelRegistry.get().get_by_name("ascend_composite_mla_prologue")
    assert spec is not None
    assert spec.capability.vendors == frozenset({"ascend"})
    assert spec.traits["expanded"] == frozenset({False, True})
    assert spec.traits["kv_format"] == frozenset({"native"})
    assert spec.traits["store"] == frozenset({True})


@pytest.mark.parametrize("degree,rank", [(1, 0), (8, 3)])
def test_ascend_dsa_consumes_host_kvp_topology(monkeypatch, degree, rank):
    # Mock device initialization, but keep the NPU constructor and its topology
    # checks. The config deliberately has no legacy dcp_* attributes.
    def initialize_base(self, config, spec, *, kernel_page_size):
        self.is_draft = config.is_draft
        self.spec_num_tokens = 8
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size

    monkeypatch.setattr(DSABackend, "__init__", initialize_base)
    monkeypatch.setattr(ascend_dsa, "new_device_stream", lambda: None)
    monkeypatch.setattr(ascend_dsa, "StreamFork", lambda stream: None)
    monkeypatch.setattr(
        ascend_dsa,
        "ascend_dsa_kernels",
        lambda: SimpleNamespace(require_context_parallel=lambda: None),
    )
    config = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method=None,
        is_draft=False,
        kvp_size=degree,
        kvp_rank=rank,
        kvp_group=tuple(range(degree)),
    )
    spec = SimpleNamespace(
        is_dsa=True,
        index_k_format="bf16",
        attn_tp_size=degree,
        num_attention_heads=2 * degree,
    )
    backend = ascend_dsa.AscendDSABackend(config, spec, kernel_page_size=128)
    assert (backend._dcp.degree, backend._dcp.rank, backend._dcp.ranks) == (
        degree,
        rank,
        config.kvp_group,
    )
    if degree > 1:
        spec.attn_tp_size = 1
        with pytest.raises(ValueError, match="complete head-TP group"):
            ascend_dsa.AscendDSABackend(config, spec, kernel_page_size=128)


def test_ascend_write_translation_is_shared_and_refreshed(monkeypatch):
    placement = ascend_dsa._AscendCachePlacement(128, 17, tuple(range(8)), 1)
    slots = torch.tensor([128, 256, 0, 257], dtype=torch.int64)
    calls = []
    resolve = ascend_dsa.resolve_cache_slots

    def counted(locations, geometry):
        calls.append(locations)
        return resolve(locations, geometry)

    monkeypatch.setattr(ascend_dsa, "resolve_cache_slots", counted)
    first = ascend_dsa.ascend_cache_write_slots(slots, placement)
    for _ in range(12):
        assert ascend_dsa.ascend_cache_write_slots(slots[:], placement) is first
    assert len(calls) == 1
    assert first.tolist() == [-1, 128, -1, 129]
    # A new round can mutate the same persistent source address.
    backend = SimpleNamespace(
        _ascend_placement=placement,
        _dcp=SimpleNamespace(clear_forward_metadata=lambda: None),
    )
    ascend_dsa.AscendDSABackend._clear_forward_scratch(backend)
    slots[0] = 258
    second = ascend_dsa.ascend_cache_write_slots(slots, placement)
    assert len(calls) == 2
    assert second.tolist() == [130, 128, -1, 129]
    # Another geometry owns different rows, never borrowing this rank's result.
    other = ascend_dsa._AscendCachePlacement(128, 17, tuple(range(8)), 0)
    assert ascend_dsa.ascend_cache_write_slots(slots, other).tolist() == [-1] * 4


def test_indexer_writes_only_its_own_cache_after_mla_prologue():
    writes = []
    index_cache = torch.empty((1, 128, 1, 32))

    class Kernels:
        def scatter(self, rows, cache, locations):
            writes.append((rows, cache, locations))

        def index(self, *args):
            return torch.zeros((1, 1), dtype=torch.int32), torch.ones(
                (1,), dtype=torch.int32
            )

    class Pool:
        def get_key_buffer(self, layer_id):
            raise AssertionError("MLA prologue already wrote the KV cache")

        def get_component(self, layer_id, name):
            assert name == "dsa_index_k"
            return index_cache

    backend = SimpleNamespace(
        _indexer_spec=SimpleNamespace(index_head_dim=32, index_topk=1),
        _indexer_kernels=Kernels(),
        kernel_page_size=128,
        step_counter=None,
        _dcp=SimpleNamespace(degree=1),
        cache_placement=lambda layer: None,
    )
    key = torch.ones((1, 1, 32))
    locations = torch.tensor([3], dtype=torch.int32)
    ascend_dsa.AscendDSABackend._select_indexed(
        backend,
        torch.ones((1, 1, 8)),
        None,
        SimpleNamespace(layer_id=0),
        locations,
        Pool(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        {
            "index_key": key,
            "index_query": torch.ones((1, 1, 32)),
            "index_weights": torch.ones((1, 1)),
            "index_init_tokens": 0,
            "index_local_tokens": 0,
        },
    )
    assert len(writes) == 1
    assert writes[0][1].data_ptr() == index_cache.data_ptr()
    assert writes[0][2] is locations


def test_each_candidate_keeps_its_own_local_page_prefix():
    table = torch.tensor([[1, 2, 3]] * 4, dtype=torch.int32)
    dcp = ascend_dsa._AscendDSAContextParallel(
        virtual_block_count=8,
        degree=8,
        rank=1,
        ranks=tuple(range(8)),
        auxiliary_namespace="test_dsa_cp",
    )
    dcp.refresh_metadata(
        page_table=table,
        seq_lens=torch.tensor([254, 255, 256, 257], dtype=torch.int32),
        page_size=128,
        block_granularity=128,
        init_tokens=0,
        local_tokens=0,
    )
    compact = dcp.metadata(start=0, end=4)
    assert compact.seq_lens.tolist() == [126, 127, 128, 128]
    assert compact.page_table[:, 0].tolist() == [1, 1, 1, 1]


def test_dcp_metadata_is_shared_only_within_matching_forward(monkeypatch):
    dcp = ascend_dsa._AscendDSAContextParallel(
        virtual_block_count=17,
        degree=8,
        rank=1,
        ranks=tuple(range(8)),
        auxiliary_namespace="test_dsa_cp",
    )
    table = torch.tensor([[1, 2, 3]] * 4, dtype=torch.int32)
    lengths = torch.tensor([254, 255, 256, 257], dtype=torch.int32)
    calls = []
    refresh = dcp.refresh_metadata

    def counted(**kwargs):
        calls.append(kwargs)
        refresh(**kwargs)

    monkeypatch.setattr(dcp, "refresh_metadata", counted)

    def prepare(local=0, rows=4):
        dcp.prepare_metadata(
            page_table=table[:rows],
            seq_lens=lengths[:rows],
            page_size=128,
            block_granularity=128,
            init_tokens=0,
            local_tokens=local,
        )

    for _ in range(6):
        prepare()
    assert len(calls) == 1
    assert dcp.seq_lens.tolist() == [126, 127, 128, 128]
    # The next round can update the same device addresses in place.
    backend = SimpleNamespace(_ascend_placement=None, _dcp=dcp)
    ascend_dsa.AscendDSABackend._clear_forward_scratch(backend)
    lengths.fill_(130)
    table[:, 1] = 10
    prepare()
    assert len(calls) == 2
    assert dcp.seq_lens.tolist() == [2] * 4
    assert dcp.page_table[:, 0].tolist() == [2] * 4
    prepare(local=1)
    assert len(calls) == 3
    assert dcp.local_counts.tolist() == [1] * 4
    prepare(local=1, rows=2)
    assert len(calls) == 4
    assert dcp.seq_lens.tolist() == [2] * 2
    # Rebinding, even to the same capacity, invalidates the forward result.
    dcp.bind_virtual_block_count(17)
    prepare(local=1, rows=2)
    assert len(calls) == 5


def test_decode_metadata_storage_is_stable_across_batch_sizes():
    dcp = ascend_dsa._AscendDSAContextParallel(
        degree=1,
        rank=0,
        ranks=(0,),
        auxiliary_namespace="test_dsa_cp",
        virtual_block_count=8,
    )
    dcp.allocate_decode_buffers(rows=8, columns=3, device="cpu")
    large = dcp._copy_buffer("query_page_table", torch.ones((8, 3), dtype=torch.int32))
    small = dcp._copy_buffer("query_page_table", torch.zeros((2, 3), dtype=torch.int32))
    assert large.data_ptr() == small.data_ptr()
    assert small.tolist() == [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(RuntimeError, match="preallocated decode geometry"):
        dcp._copy_buffer("query_page_table", torch.ones((9, 3), dtype=torch.int32))


@pytest.mark.parametrize("prefill_rows,prefill_columns", [(17, 3), (4, 4)])
def test_large_prefill_does_not_replace_decode_metadata_buffers(
    prefill_rows, prefill_columns
):
    dcp = ascend_dsa._AscendDSAContextParallel(
        degree=8,
        rank=0,
        ranks=tuple(range(8)),
        auxiliary_namespace="test_dsa_cp",
        virtual_block_count=17,
    )
    dcp.allocate_decode_buffers(8, 3, "cpu")

    def refresh(rows, columns):
        dcp.refresh_metadata(
            seq_lens=torch.full((rows,), 260, dtype=torch.int32),
            page_table=torch.tensor(
                [[1, 2, 9] + [0] * (columns - 3)] * rows, dtype=torch.int32
            ),
            page_size=128,
            block_granularity=128,
            init_tokens=0,
            local_tokens=0,
        )
        meta = dcp.metadata(start=0, end=rows)
        assert meta.seq_lens.tolist() == [132] * rows
        assert meta.page_table.tolist() == [[1, 2] + [0] * (columns - 2)] * rows
        return meta

    small = refresh(8, 3)
    pointers = (small.page_table.data_ptr(), small.seq_lens.data_ptr())
    refresh(prefill_rows, prefill_columns)
    restored = refresh(8, 3)
    assert (restored.page_table.data_ptr(), restored.seq_lens.data_ptr()) == pointers


@pytest.mark.parametrize("degree", [1, 2, 8])
@pytest.mark.parametrize("block_size", [128, 256])
def test_sharded_index_write_and_read_use_local_storage(degree, block_size):
    """Virtual IDs exceed each rank's capacity; replicated addressing fails."""

    class ReachedIndexer(Exception):
        pass

    subpages = block_size // 128
    blocks = [degree * 2, 1, degree + 1, 2]
    # No repeated block within a request, including degree=1.
    blocks = list(dict.fromkeys(blocks))
    pages = [b * subpages + s for b in blocks for s in range(subpages)]
    table = torch.tensor([pages] * 8, dtype=torch.int32)
    lengths = torch.arange(
        len(pages) * 128 - 7, len(pages) * 128 + 1, dtype=torch.int32
    )
    virtual_count = degree * 2 + 1
    slots = torch.arange(virtual_count * block_size, dtype=torch.int64)
    keys = slots.float().view(-1, 1, 1).expand(-1, 1, 8).contiguous()
    for rank in range(degree):
        placement = CachePlacement(
            block_size, virtual_count, tuple(range(degree)), rank
        )
        cache = torch.full((3 * block_size, 1, 8), -99.0)
        dcp = ascend_dsa._AscendDSAContextParallel(
            degree=degree,
            rank=rank,
            ranks=tuple(range(degree)),
            auxiliary_namespace="test_dsa_cp",
            virtual_block_count=virtual_count,
        )
        # Stop before collectives: this test exercises the real write and
        # metadata paths independently for every owner.
        dcp.primary_process_group = object()

        class Kernels:
            def scatter(self, rows, destination, locations):
                assert int(locations.max()) < cache.shape[0]
                virtual_blocks = slots // block_size
                owners = (virtual_blocks > 0) & ((virtual_blocks - 1) % degree == rank)
                assert (locations[~owners] == -1).all()
                valid = locations >= 128
                destination.view_as(cache)[locations[valid]] = rows[valid]

            def index_partial(self, *args, **kwargs):
                raise ReachedIndexer

            def index(self, *args):
                raise ReachedIndexer

        backend = SimpleNamespace(
            _indexer_spec=SimpleNamespace(index_head_dim=8, index_topk=512),
            _indexer_kernels=Kernels(),
            kernel_page_size=128,
            step_counter=None,
            _dcp=dcp,
            cache_placement=lambda layer: placement,
        )
        with pytest.raises(ReachedIndexer):
            ascend_dsa.AscendDSABackend._select_indexed(
                backend,
                None,
                None,
                SimpleNamespace(layer_id=0),
                slots,
                SimpleNamespace(get_component=lambda *args: cache),
                torch.arange(1, 9, dtype=torch.int32),
                lengths,
                table,
                dict(
                    index_key=keys,
                    index_query=None,
                    index_weights=None,
                    index_init_tokens=131,
                    index_local_tokens=129,
                ),
                context_parallel=True,
            )
        assert (cache[:block_size] == -99).all()
        for local_block in (1, 2):
            v = (local_block - 1) * degree + rank + 1
            torch.testing.assert_close(
                cache[local_block * block_size : (local_block + 1) * block_size],
                keys[v * block_size : (v + 1) * block_size],
            )
        # Check the same ABI also for degree=1 (no collective branch).
        dcp.refresh_metadata(
            seq_lens=lengths,
            page_table=table,
            page_size=128,
            block_granularity=block_size,
            init_tokens=131,
            local_tokens=129,
        )
        meta = dcp.metadata(start=0, end=8)
        for row, end in enumerate(lengths.tolist()):
            expected = []
            positions = []
            for pos in range(end):
                page = pages[pos // 128]
                block = page // subpages
                if (block - 1) % degree == rank:
                    expected.append(page * 128 + pos % 128)
                    positions.append(pos)
            n = int(meta.seq_lens[row])
            assert n == len(expected)
            assert int(meta.init_counts[row]) == sum(p < 131 for p in positions)
            assert int(meta.local_counts[row]) == sum(
                p >= max(0, end - 129) for p in positions
            )
            indices = torch.arange(n)
            addresses = (
                meta.page_table[row, indices // 128].long() * 128 + indices % 128
            )
            torch.testing.assert_close(
                cache[addresses, 0, 0], torch.tensor(expected, dtype=torch.float32)
            )


@pytest.mark.parametrize("degree", [1, 8])
def test_prefill_uses_global_query_visibility_before_dcp_compaction(degree):
    records = []
    backend = SimpleNamespace(
        _validate_logit_cap=lambda cap: None,
        _dcp=SimpleNamespace(degree=degree),
        forward_prefill_metadata=SimpleNamespace(
            cum_extend_seq_lens=torch.tensor([0, 3, 5]),
            seq_lens=torch.tensor([130, 257]),
            page_table=torch.tensor([[9, 2, 0], [8, 1, 10]], dtype=torch.int32),
        ),
        _select_indexed=lambda *args, **kw: records.append((args, kw)),
        _run_indexed_sparse_attention=lambda *args, **kw: None,
    )
    ascend_dsa.AscendDSABackend.forward_extend(
        backend,
        torch.zeros(5, 1, 8),
        None,
        None,
        SimpleNamespace(logit_cap=0),
        torch.arange(5),
        None,
        2,
    )
    args, options = records[0]
    assert options["context_parallel"] == (degree > 1)
    if degree > 1:
        assert args[5].tolist() == [1, 2, 3, 4, 5]
        assert args[6].tolist() == [128, 129, 130, 256, 257]
        assert args[7].tolist() == [[9, 2, 0]] * 3 + [[8, 1, 10]] * 2
    else:
        assert args[5].tolist() == [3, 5]
        assert args[6].tolist() == [130, 257]


@pytest.mark.parametrize("width", [2, 4, 8])
def test_dcp_verify_batches_query_rows_with_causal_lengths(width):
    lengths = torch.tensor([20, 129, 260], dtype=torch.int32)
    table = torch.arange(9, dtype=torch.int32).view(3, 3)
    metadata = SimpleNamespace(seq_lens=lengths, page_table=table, num_extends=1)

    class FakeDcp:
        degree = 8

        def __init__(self):
            self.refreshed = []

        def _copy_buffer(self, name, value):
            del name
            return value.clone()

        def refresh_metadata(self, *, seq_lens, page_table, **kwargs):
            del kwargs
            self.refreshed.append((seq_lens.clone(), page_table.clone()))

    class FakeBackend:
        spec_num_tokens = width
        kernel_page_size = 128
        forward_decode_metadata = metadata

        def _clear_forward_scratch(self):
            pass

        def __init__(self):
            self._dcp = FakeDcp()
            self._dense_backend = SimpleNamespace(
                refresh_decode_metadata=lambda *args, **kwargs: None
            )
            self._decode_query_seq_lens = None
            self._decode_query_page_table = None
            self.selections = []

        def _validate_logit_cap(self, value):
            assert value == 0

        def _select_indexed(
            self,
            q,
            k,
            layer,
            locations,
            pool,
            q_ends,
            kv_lengths,
            pages,
            kwargs,
            **options,
        ):
            del k, layer, pool, kwargs
            self.selections.append(
                (
                    q_ends.clone(),
                    kv_lengths.clone(),
                    pages.clone(),
                    locations.clone(),
                    options,
                )
            )
            return q

        def _run_indexed_sparse_attention(
            self, selection, layer, pool, *, head_major_output
        ):
            del layer, pool, head_major_output
            return selection.flatten(1)

    backend = FakeBackend()
    ascend_dsa.AscendDSABackend.refresh_decode_metadata(backend, 3, 3, lengths, table)
    expected_lengths = torch.cat(
        [torch.arange(int(end) - width + 1, int(end) + 1) for end in lengths]
    ).to(torch.int32)
    torch.testing.assert_close(backend._decode_query_seq_lens, expected_lengths)
    torch.testing.assert_close(
        backend._decode_query_page_table, table.repeat_interleave(width, dim=0)
    )
    # The model supplies selection policy at execution, not metadata refresh.
    assert not backend._dcp.refreshed

    rows = 2 * width
    query = torch.arange(rows * 2, dtype=torch.float32).view(rows, 1, 2)
    locations = torch.arange(rows, dtype=torch.int32)
    output = ascend_dsa.AscendDSABackend.forward_decode(
        backend,
        query,
        query,
        None,
        SimpleNamespace(logit_cap=0),
        locations,
        None,
        2,
    )
    torch.testing.assert_close(output, query.flatten(1))
    assert len(backend.selections) == 1
    q_ends, visible, pages, selected_locations, options = backend.selections[0]
    assert q_ends.tolist() == list(range(1, rows + 1))
    torch.testing.assert_close(visible, expected_lengths[width:])
    torch.testing.assert_close(pages, table[1:].repeat_interleave(width, dim=0))
    torch.testing.assert_close(selected_locations, locations)
    assert options["context_parallel"] is True


@pytest.mark.parametrize("initial,local", [(-1, 0), (0, -1), (3, 2)])
def test_index_policy_is_validated_at_execution(initial, local):
    backend = SimpleNamespace(_indexer_spec=SimpleNamespace(index_topk=4))
    with pytest.raises(ValueError):
        ascend_dsa.AscendDSABackend._select_indexed(
            backend,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            {"index_init_tokens": initial, "index_local_tokens": local},
        )


@pytest.mark.parametrize("width", [2, 4, 8])
def test_dcp_indexer_receives_model_policy_for_each_candidate(width):
    class ReachedIndexer(Exception):
        pass

    lengths = torch.arange(253, 253 + width, dtype=torch.int32)
    table = torch.tensor([[1, 2, 3]] * width, dtype=torch.int32)
    initial, local = 130, 129
    records = []

    class Dcp(ascend_dsa._AscendDSAContextParallel):
        @property
        def ready(self):
            return True

        def refresh_metadata(
            self,
            *,
            seq_lens,
            page_table,
            page_size,
            block_granularity,
            init_tokens,
            local_tokens,
        ):
            assert init_tokens == initial and local_tokens == local
            super().refresh_metadata(
                page_table=page_table,
                seq_lens=seq_lens,
                page_size=page_size,
                block_granularity=block_granularity,
                init_tokens=init_tokens,
                local_tokens=local_tokens,
            )

    class Kernels:
        def scatter(self, *args):
            pass

        def index_partial(self, *args, **kwargs):
            records.append(args)
            raise ReachedIndexer

    backend = SimpleNamespace(
        _indexer_spec=SimpleNamespace(index_head_dim=8, index_topk=512),
        _indexer_kernels=Kernels(),
        kernel_page_size=128,
        step_counter=None,
        _dcp=Dcp(
            degree=2,
            rank=1,
            ranks=(0, 1),
            auxiliary_namespace="test_dsa_cp",
            virtual_block_count=8,
        ),
        cache_placement=lambda layer: CachePlacement(128, 8, (0, 1), 1),
    )
    with pytest.raises(ReachedIndexer):
        ascend_dsa.AscendDSABackend._select_indexed(
            backend,
            torch.ones(width, 1, 8),
            None,
            SimpleNamespace(layer_id=0),
            torch.arange(width),
            SimpleNamespace(get_component=lambda *args: torch.zeros(3, 128, 1, 8)),
            torch.arange(1, width + 1, dtype=torch.int32),
            lengths,
            table,
            {
                "index_key": torch.ones(width, 1, 8),
                "index_query": torch.ones(width, 1, 8),
                "index_weights": torch.ones(width, 1),
                "index_init_tokens": initial,
                "index_local_tokens": local,
            },
            context_parallel=True,
        )
    args = records[0]
    expected_initial = [
        sum(128 <= p < 256 for p in range(min(initial, n))) for n in lengths.tolist()
    ]
    expected_local = [
        sum(128 <= p < 256 for p in range(max(n - local, 0), n))
        for n in lengths.tolist()
    ]
    assert args[7].tolist() == expected_initial
    assert args[8].tolist() == expected_local
    assert args[4].tolist() == [min(max(n - 128, 0), 128) for n in lengths.tolist()]


def test_single_token_mixed_decode_selects_request_shaped_query_metadata():
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([12, 24], dtype=torch.int32),
        page_table=torch.tensor([[1], [2]], dtype=torch.int32),
        num_extends=1,
    )

    class FakeDcp:
        degree = 8

        def __init__(self):
            self.refreshed = []

        def refresh_metadata(self, *, seq_lens, page_table, **kwargs):
            del kwargs
            self.refreshed.append((seq_lens.clone(), page_table.clone()))

    class FakeBackend:
        spec_num_tokens = 4
        kernel_page_size = 128
        forward_decode_metadata = metadata
        _decode_query_ends = torch.arange(1, 9, dtype=torch.int32)

        def __init__(self):
            self._dcp = FakeDcp()

        def _validate_logit_cap(self, value):
            assert value == 0

        def _select_indexed(
            self,
            q,
            k,
            layer,
            locations,
            pool,
            q_ends,
            kv_lengths,
            pages,
            kwargs,
            **options,
        ):
            del k, layer, locations, pool, kwargs
            assert q_ends.tolist() == [1]
            assert kv_lengths.tolist() == [24]
            assert pages.tolist() == [[2]]
            assert options["context_parallel"] is True
            return q

        def _run_indexed_sparse_attention(
            self, selection, layer, pool, *, head_major_output
        ):
            del layer, pool, head_major_output
            return selection.flatten(1)

    backend = FakeBackend()
    query = torch.ones((1, 1, 2))
    ascend_dsa.AscendDSABackend.forward_decode(
        backend,
        query,
        query,
        None,
        SimpleNamespace(logit_cap=0),
        torch.tensor([7], dtype=torch.int32),
        None,
        1,
    )
    assert not backend._dcp.refreshed
