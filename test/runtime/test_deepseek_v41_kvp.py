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


import sys
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.layers.attention import deepseek_v41_kvp as kvp
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")


@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [0, 1, 63, 64, 65, 193])
@pytest.mark.parametrize("width", [68, 272])
def test_packed_history_preserves_bytes_and_strides(monkeypatch, size, rows, width):
    # Non-monotonic virtual allocation; some ranks contribute no rows.
    table = torch.tensor([7, 2, 13, 4], dtype=torch.int32)
    # Use arbitrary nonzero bytes through the final page, including scales
    # after the last valid token. Those bytes must survive a partial history.
    padded_rows = (rows + 63) // 64 * 64
    logical = torch.arange(padded_rows)
    virtual = table[logical // 64].long() * 64 + logical % 64
    expected = (
        (logical[:, None] * 13 + torch.arange(width)[None, :]).remainder(256).byte()
    )
    order = torch.argsort((virtual // 64 - 1) % size, stable=True)
    for rank in range(size):
        placement = CachePlacement(64, 17, tuple(range(size)), rank)
        # Simulate LCM planes whose page stride exceeds the field width.
        storage = torch.zeros((18, 64, width + 16), dtype=torch.uint8)
        cache = storage[:, :, :width]
        owner = (virtual // 64 - 1) % size
        local = ((virtual // 64 - 1) // size + 1) * 64 + virtual % 64
        mine = owner == rank
        cache[local[mine] // 64, local[mine] % 64] = expected[mine]

        def gather(plan, local_rows, *, out):
            assert plan.rank == rank
            torch.testing.assert_close(local_rows, expected[mine], rtol=0, atol=0)
            assert sum(plan.owned_rows_per_rank) == padded_rows
            torch.testing.assert_close(plan.order, order)
            out[:padded_rows].index_copy_(0, plan.order, expected[order])
            return out[:padded_rows]

        monkeypatch.setattr(kvp, "gather_history_rows", gather)
        result = kvp.gather_paged_history(
            cache, table, table, rows=rows, placement=placement
        )
        assert result.is_contiguous()
        assert result.shape == ((rows + 63) // 64 + 1, 64, width)
        assert not result[0].any()
        flat = result[1:].reshape(-1, width)
        torch.testing.assert_close(flat, expected, rtol=0, atol=0)


def test_history_rejects_missing_page():
    cache = torch.zeros((4, 64, 68), dtype=torch.uint8)
    table = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(ValueError):
        kvp.gather_paged_history(
            cache,
            table,
            table,
            rows=1,
            placement=CachePlacement(64, 9, (0, 1), 0),
        )


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_recipe_shards_history_not_swa_or_tails(size):
    from dataclasses import replace
    from test.runtime.test_deepseek_v41_cache import R1, R2, _layout, _recipe

    recipe = _recipe("cpu")
    recipe.attn_config = replace(
        recipe.attn_config,
        kvp_size=size,
        kvp_rank=0,
        kvp_group=tuple(range(size)),
    )
    groups = recipe.groups()
    for spec, _ in groups:
        assert spec.shard_count == (size if spec.group_id in (R1, R2) else 1)
    recipe.check_layout(_layout(recipe))
    assert recipe.workspace_bytes() > 0


def test_remote_selection_is_valid_but_remote_write_is_masked():
    from dataclasses import replace
    from test.runtime.test_deepseek_v41_cache import (
        R1,
        _extend,
        _pool,
        _recipe,
        _tables,
    )

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
        DeepseekV41AttentionBackend,
    )
    from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import (
        DeepseekV41Config,
    )

    recipe = _recipe("cpu")
    recipe.attn_config = replace(
        recipe.attn_config, kvp_size=2, kvp_rank=1, kvp_group=(0, 1)
    )
    backend = DeepseekV41AttentionBackend(
        recipe.attn_config, recipe.attn_config.component(DeepseekV41Config)
    )
    backend.set_cache_pool(_pool(recipe, "cpu"))
    assert backend.cuda_graph_support.decode_graph
    assert not backend.cuda_graph_support.prefill_graph
    tables = _tables("cpu")
    tables[R1][0, :2] = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="host block tables"):
        _extend(backend, tables, [128], [0], [0], [128])
    # Use the existing fixture to initialize all baseline metadata, then restore KVP.
    backend.kvp_size = 1
    _extend(backend, tables, [128], [0], [0], [128])
    backend.kvp_size = 2
    backend._kvp_tables_cpu = tables
    positions = torch.tensor([0, 64], dtype=torch.int64)
    requests = torch.zeros(2, dtype=torch.int64)
    writes = backend.cache_slots(R1, positions, requests, ForwardMode.EXTEND)
    assert writes.tolist() == [-1, 64]
    reads = backend.global_read_slots(
        20, positions[:, None], positions, requests, ForwardMode.EXTEND
    )
    assert reads.flatten().tolist() == [64, 128]


@pytest.mark.parametrize(
    "role,speculative,mixed,graph",
    [
        ("encode", None, False, False),
        ("decode", None, False, False),
        ("prefill", "MTP", False, False),
        ("prefill", None, True, False),
        ("prefill", None, False, True),
    ],
)
def test_unsupported_kvp_launches_fail_before_model_construction(
    role, speculative, mixed, graph
):
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import (
        DeepseekV41Config,
    )

    args = SimpleNamespace(
        mapping=SimpleNamespace(attn=SimpleNamespace(kvp_size=2)),
        disaggregation_mode=role,
        speculative_algorithm=speculative,
        enable_mixed_batch=mixed,
        disable_prefill_graph=not graph,
    )
    with pytest.raises(ValueError, match="V4.1 KVP"):
        DeepseekV41Config.generate(args, None, False)


@pytest.mark.parametrize("local_reindex", [False, True])
@pytest.mark.parametrize("width", [1, 6])
@pytest.mark.parametrize("graph", [False, True])
def test_kvp_decode_refresh_has_no_host_history_plan(
    width, graph, local_reindex, monkeypatch
):
    from dataclasses import replace
    from test.runtime.test_deepseek_v41_cache import _pool, _recipe, _tables

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
        DeepseekV41AttentionBackend,
    )
    from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import (
        DeepseekV41Config,
    )

    recipe = _recipe("cpu")
    recipe.attn_config = replace(
        recipe.attn_config,
        kvp_size=2,
        kvp_rank=1,
        kvp_group=(0, 1),
        speculative_num_draft_tokens=width,
    )
    recipe.decode_input_tokens = width
    backend = DeepseekV41AttentionBackend(
        recipe.attn_config, recipe.attn_config.component(DeepseekV41Config)
    )
    pool = _pool(recipe, "cpu")
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(2, max_tokens_per_req=width, overlap_schedule_depth=1)
    tables = _tables("cpu")
    from tokenspeed_kernel.ops.attention import dsv41

    import tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 as backend_module

    history_calls = []
    local_calls = []

    def history(cache, table, visible_rows, *, placement):
        history_calls.append(visible_rows)
        return cache.new_zeros((table.numel() + 1, 64, cache.shape[-1]))

    calls = []

    def select(*args):
        page_table, visible, candidates = args[3:6]
        # Keep decode eligible for paged Full and sparse Reindex, even when
        # several verify tokens share the same request's logical page table.
        assert page_table.stride(0) != 0
        expected = torch.arange(1, page_table.shape[1] + 1, dtype=page_table.dtype)
        torch.testing.assert_close(page_table, expected.expand_as(page_table))
        calls.append(candidates is not None)
        rows, lens, blocks, block_lens = args[12]
        rows.fill_(-1)
        lens.zero_()
        blocks.fill_(-1)
        block_lens.zero_()
        return rows, lens, blocks, block_lens

    def local_candidates(q, weights, cache, page_table, visible, candidates, topk):
        from tokenspeed.runtime.layers.attention.dcp.placement import (
            resolve_cache_slots,
        )

        request = len(local_calls) % 2
        group_id = backend._owner_group(20)
        slots, owned = resolve_cache_slots(
            tables[group_id][request : request + 1].long() * 64,
            backend._kvp_placement(group_id),
        )
        expected_table = (slots // 64).to(torch.int32).masked_fill(~owned, -1)
        expected_table = torch.nn.functional.pad(
            expected_table, (0, page_table.shape[1] - expected_table.shape[1]), value=-1
        )
        if request >= len(lengths):
            expected_table.fill_(-1)
        torch.testing.assert_close(page_table, expected_table.expand_as(page_table))
        local_calls.append(visible)
        return torch.zeros(q.shape[0], topk, 2, dtype=torch.int32)

    def merge(packed, rows, lens):
        rows.fill_(-1)
        lens.zero_()

    monkeypatch.setattr(
        dsv41,
        "local_index_supported",
        lambda q, cache, candidates: local_reindex and candidates is not None,
    )
    monkeypatch.setattr(dsv41, "index_candidates", local_candidates)
    monkeypatch.setattr(dsv41, "merge_index_candidates", merge)
    monkeypatch.setattr(
        backend_module, "all_gather", lambda x, group, dim: torch.cat([x, x], dim=dim)
    )
    monkeypatch.setattr(backend_module, "gather_decode_history", history)
    monkeypatch.setattr(dsv41, "index_topk", select)
    for lengths in ([130, 65], [7], []):
        backend.refresh_decode_metadata(
            2,
            len(lengths),
            torch.arange(len(lengths)),
            torch.tensor(lengths, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=graph,
        )
        meta = backend.query_metadata(ForwardMode.DECODE)
        expected = [p for length in lengths for p in range(length - width, length)]
        assert meta.positions.tolist() == expected + [-1] * ((2 - len(lengths)) * width)
        assert not backend._kvp_tables_cpu
        calls.clear()
        history_calls.clear()
        local_calls.clear()
        for layer in (2, 20, 24):
            backend.select_global(
                layer,
                torch.zeros(2 * width, 32, 128),
                torch.zeros(2 * width, 32),
                meta.positions,
                meta.request_indices,
                ForwardMode.DECODE,
                None,
                projected_partition=None,
            )
        assert calls == ([False] * 4 if local_reindex else [False] * 4 + [True] * 2)
        assert len(history_calls) == (4 if local_reindex else 6)
        assert len(local_calls) == (2 if local_reindex else 0)
    # Eager model startup seeds metadata through the inherited capture API.
    backend.init_forward_metadata_capture_cuda_graph(
        2,
        torch.arange(2),
        torch.ones(2, dtype=torch.int32),
        ForwardMode.DECODE,
        block_tables=tables,
    )
    assert not backend._kvp_tables_cpu
    backend.set_cache_pool(_pool(recipe, "cpu"))
    assert not backend._kvp_tables_cpu


@pytest.mark.parametrize("rows", [0, 1, 64, 65, 128])
def test_decode_history_fixed_capacity_preserves_packed_pages(monkeypatch, rows):
    table = torch.tensor([2, 1, 0], dtype=torch.int32)
    cache = torch.arange(4 * 64 * 68).remainder(256).byte().reshape(4, 64, 68)
    expected = cache.new_zeros((4, 64, 68))
    page_count = (rows + 63) // 64
    expected[1 : page_count + 1] = cache[table[:page_count]]
    contributions = []

    def reduce(tensor, group, *, op):
        assert op == torch.distributed.ReduceOp.MAX
        contributions.append(tensor.clone())
        return expected[1:]

    monkeypatch.setattr(kvp, "all_reduce", reduce)
    for rank in (0, 1):
        placement = CachePlacement(64, 7, (0, 1), rank)
        local = cache.new_zeros(cache.shape)
        for virtual in (1, 2):
            if (virtual - 1) % 2 == rank:
                local[(virtual - 1) // 2 + 1] = cache[virtual]
        out = kvp.gather_decode_history(
            local, table, torch.tensor(rows), placement=placement
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.maximum(*contributions), expected[1:], rtol=0, atol=0
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_local_decode_selection_reuse_and_refresh(rank, monkeypatch):
    from dataclasses import replace
    from test.runtime.test_deepseek_v41_cache import _pool, _recipe, _tables

    from tokenspeed_kernel.ops.attention import dsv41

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
        DeepseekV41AttentionBackend,
    )
    from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import (
        DeepseekV41Config,
    )

    recipe = _recipe("cpu")
    recipe.attn_config = replace(
        recipe.attn_config,
        kvp_size=2,
        kvp_rank=rank,
        kvp_group=(0, 1),
    )
    backend = DeepseekV41AttentionBackend(
        recipe.attn_config, recipe.attn_config.component(DeepseekV41Config)
    )
    backend.set_cache_pool(_pool(recipe, "cpu"))
    backend.init_cuda_graph_state(2, max_tokens_per_req=1, overlap_schedule_depth=1)
    calls = []

    def compact(slots, lengths):
        calls.append(1)
        valid = (slots >= 0) & (torch.arange(slots.shape[1])[None] < lengths[:, None])
        out = torch.full_like(slots, -1)
        for i in range(slots.shape[0]):
            out[i, : int(valid[i].sum())] = slots[i, valid[i]]
        return out, valid.sum(-1).int()

    monkeypatch.setattr(dsv41, "compact_slots", compact)
    monkeypatch.setattr(dsv41, "new_attention_schedule", object)
    slots = torch.tensor([[64, 128, 192, -1], [256, -1, -1, -1]])
    lengths = torch.tensor([3, 1], dtype=torch.int32)
    swa = torch.tensor([128, 64], dtype=torch.int32)
    first = backend._local_decode_selection(20, slots, lengths, swa)
    second = backend._local_decode_selection(20, slots, lengths, swa)
    assert all(a is b for a, b in zip(first, second))
    assert len(calls) == 1
    assert first[1].tolist() == ([2, 0] if rank == 0 else [1, 1])
    assert first[2].tolist() == (swa.tolist() if rank == 0 else [0, 0])
    assert backend._decode_schedule(first[2], first[1]) is backend._decode_schedule(
        second[2], second[1]
    )
    # A subset and a fresh Reindex selection must not reuse the parent's result.
    subset = backend._local_decode_selection(20, slots[:1], lengths[:1], swa[:1])
    assert subset[0].shape[0] == 1
    backend._local_decode_selection(20, slots.clone(), lengths, swa)
    assert len(calls) == 3
    backend.refresh_decode_metadata(
        2,
        0,
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables=_tables("cpu"),
        for_graph_replay=True,
    )
    # Even pointer-stable input buffers can change on the next forward.
    lengths.zero_()
    refreshed = backend._local_decode_selection(20, slots, lengths, swa)
    assert len(calls) == 4
    assert not refreshed[1].any()
    assert (refreshed[0] == -1).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
