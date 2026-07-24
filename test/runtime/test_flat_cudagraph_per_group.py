"""Focused flat CUDA-graph contracts: null padding, aliasing, and refresh."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

torch = pytest.importorskip("torch")
_mha = pytest.importorskip("tokenspeed.runtime.layers.attention.backends.mha")
_flat_groups = pytest.importorskip(
    "tokenspeed.runtime.layers.attention.backends.flat_groups"
)

from tokenspeed.runtime.configs.paged_cache_spec import (  # noqa: E402
    PagedCacheGroupSpec,
)
from tokenspeed.runtime.execution.cuda_graph_wrapper import (  # noqa: E402
    CudaGraphWrapper,
)
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.flat_cache_tables import CacheTableSource  # noqa: E402

MAX_BS = 4
MAX_NUM_PAGES = 6
GROUP_IDS = ("sliding_attention", "full_attention")


def _decode_mode() -> SimpleNamespace:
    return SimpleNamespace(is_extend_or_mixed=lambda: False)


def _history_spec(group_id: str) -> PagedCacheGroupSpec:
    return PagedCacheGroupSpec(
        group_id=group_id,
        retention="full_history",
        rows_per_page=2,
        entry_stride_tokens=1,
        sliding_window_tokens=None,
        block_size=2,
        pool_id=f"pool.{group_id}",
        owner_mask=1,
    )


def _backend(*, draft_lookback: int = 0):
    backend = _mha.MHAAttnBackend.__new__(_mha.MHAAttnBackend)
    backend.spec_num_tokens = 1
    backend.is_draft = False
    backend.draft_block_decode = False
    backend.flat_draft_lookback = draft_lookback
    backend.max_num_pages = MAX_NUM_PAGES
    backend.max_context_len = MAX_NUM_PAGES * 2
    backend.page_size = 2
    backend.device = "cpu"
    backend.forward_decode_metadata = None
    backend.forward_extend_metadata = None
    backend.init_cuda_graph_state(
        max_bs=MAX_BS,
        seq_lens_buf=torch.ones(MAX_BS, dtype=torch.int32),
        paged_cache_group_specs=tuple(_history_spec(gid) for gid in GROUP_IDS),
    )
    return backend


def _capture(backend, bs: int = 2):
    backend.init_forward_metadata_capture_cuda_graph(
        bs,
        torch.arange(bs, dtype=torch.int64),
        torch.ones(bs, dtype=torch.int32),
        _decode_mode(),
        flat_cache_group_ids=GROUP_IDS,
    )
    return backend.forward_decode_metadata


@pytest.mark.parametrize("source_generation", (None, 3), ids=("missing", "stale"))
def test_stale_generation_fails_at_wrapper_entry_before_graph_selection(
    source_generation: int | None,
) -> None:
    wrapper = CudaGraphWrapper.__new__(CudaGraphWrapper)
    wrapper._flat_generation_pool = SimpleNamespace(
        flat_memory_plan=object(),
        arena_generation=4,
    )
    wrapper._can_use_graph = mock.Mock()
    source = CacheTableSource(
        kind="flat",
        tables={"history": object()},
        base_offsets={"history": object()},
        generation=source_generation,
    )
    with pytest.raises(RuntimeError, match="stale"):
        wrapper(
            bs=2,
            ctx=SimpleNamespace(),
            sampling_info=mock.sentinel.sampling_info,
            req_to_page=torch.zeros((2, 2), dtype=torch.int32),
            flat_cache_table_source=source,
        )
    wrapper._can_use_graph.assert_not_called()


def test_stale_generation_fails_at_executor_entry_before_input_fill() -> None:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor._flat_generation_pool = SimpleNamespace(arena_generation=4)
    forward_op = SimpleNamespace(cache_generation=3, num_extends=mock.Mock())

    with pytest.raises(RuntimeError, match="stale"):
        executor.execute_forward_op(forward_op, [])
    forward_op.num_extends.assert_not_called()


def test_replay_padding_routes_dummy_rows_to_null_page() -> None:
    tables = {
        "sliding_attention": torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        "full_attention": torch.tensor([[7, 8], [9, 10]], dtype=torch.int32),
    }
    bases = {gid: torch.tensor([3, 4], dtype=torch.int32) for gid in tables}

    padded_tables = CudaGraphWrapper._pad_block_tables_to_padded_bs(
        tables,
        actual_bs=2,
        padded_bs=4,
        pad_value=0,
    )
    padded_bases = CudaGraphWrapper._pad_offsets_to_padded_bs(
        bases,
        actual_bs=2,
        padded_bs=4,
    )

    for group_id, table in padded_tables.items():
        assert torch.equal(table[:2], tables[group_id])
        assert table[2:].tolist() == [[0, 0], [0, 0]]
        assert padded_bases[group_id].tolist() == [3, 4, 0, 0]


def test_capture_metadata_aliases_persistent_group_buffers() -> None:
    backend = _backend()
    metadata = _capture(backend)

    assert metadata.page_table is None
    assert set(metadata.page_tables) == set(GROUP_IDS)
    for group_id in GROUP_IDS:
        table_buffer = backend.cuda_graph_flat_page_tables[group_id]
        base_buffer = backend.cuda_graph_flat_block_table_base_offsets[group_id]
        assert metadata.page_tables[group_id].data_ptr() == table_buffer.data_ptr()
        assert (
            metadata.block_table_base_offsets[group_id].data_ptr()
            == base_buffer.data_ptr()
        )


@pytest.mark.parametrize(
    ("module_name", "backend_name"),
    (
        ("tokenspeed.runtime.layers.attention.backends.mha", "MHAAttnBackend"),
        ("tokenspeed.runtime.layers.attention.backends.msa", "MSAAttnBackend"),
    ),
    ids=("mha", "msa"),
)
def test_repeated_capture_reuses_metadata_object(
    module_name: str,
    backend_name: str,
) -> None:
    module = pytest.importorskip(module_name)
    backend_type = getattr(module, backend_name)
    backend = backend_type.__new__(backend_type)
    backend.spec_num_tokens = 1
    backend.is_draft = False
    backend.draft_block_decode = False
    backend.max_num_pages = MAX_NUM_PAGES
    backend.max_context_len = MAX_NUM_PAGES * 2
    backend.page_size = 2
    backend.device = "cpu"
    backend.forward_decode_metadata = None
    backend.forward_extend_metadata = None
    if backend_name == "MSAAttnBackend":
        backend.decode_score_buffer = torch.empty(
            (MAX_BS, 1, MAX_NUM_PAGES), dtype=torch.float32
        )
    backend.init_cuda_graph_state(
        max_bs=MAX_BS,
        seq_lens_buf=torch.ones(MAX_BS, dtype=torch.int32),
        paged_cache_group_specs=tuple(_history_spec(gid) for gid in GROUP_IDS),
    )

    first = _capture(backend)
    second = _capture(backend)

    assert second is first
    assert second is backend.cuda_graph_decode_metadata[2]
    assert set(second.block_table_base_offsets) == set(GROUP_IDS)


def test_capture_init_restores_main_locs_after_lookback() -> None:
    backend = _backend(draft_lookback=2)
    metadata = _capture(backend)
    main_locs = dict(metadata.out_cache_locs)

    assert backend.flat_enter_draft_lookback(2)
    assert backend.forward_decode_metadata is metadata
    assert _capture(backend) is metadata
    for group_id, locs in main_locs.items():
        assert metadata.out_cache_locs[group_id].is_set_to(locs)


def test_replay_refreshes_persistent_tables_without_rebinding_metadata() -> None:
    backend = _backend()
    metadata = _capture(backend)
    backend._flat_try_packed_unpack = lambda bs, tables: False

    with mock.patch.object(_flat_groups, "flat_decode_locs") as fill_locs:
        for value, width, bases in ((10, 2, (0, 1)), (20, 3, (2, 3))):
            tables = {
                group_id: torch.full((2, width), value, dtype=torch.int32)
                for group_id in GROUP_IDS
            }
            base_offsets = {
                group_id: torch.tensor(bases, dtype=torch.int32)
                for group_id in GROUP_IDS
            }
            backend.init_forward_metadata_replay_cuda_graph(
                2,
                torch.arange(MAX_BS, dtype=torch.int64),
                torch.tensor([3, 4, 1, 1], dtype=torch.int32),
                torch.zeros((MAX_BS, MAX_NUM_PAGES), dtype=torch.int32),
                _decode_mode(),
                flat_block_tables=tables,
                flat_block_table_base_offsets=base_offsets,
            )

            for group_id in GROUP_IDS:
                persistent = backend.cuda_graph_flat_page_tables[group_id]
                assert torch.equal(persistent[:2, :width], tables[group_id])
                assert (persistent[:2, width:] == -1).all()
                assert backend.cuda_graph_flat_block_table_base_offsets[group_id][
                    :2
                ].tolist() == list(bases)
            assert backend.forward_decode_metadata is metadata

    assert fill_locs.call_count == 2
