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


import pytest
import torch

from tokenspeed.runtime.layers.attention import deepseek_v41_kvp as kvp
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement


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
    assert not backend.cuda_graph_support.decode_graph
    assert not backend.cuda_graph_support.prefill_graph
    tables = _tables("cpu")
    tables[R1][0, :2] = torch.tensor([1, 2], dtype=torch.int32)
    # Supply the CPU mirror explicitly, as forward_step does in production.
    with pytest.raises(NotImplementedError):
        backend.refresh_decode_metadata(
            1,
            1,
            None,
            None,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=True,
        )
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
        ("prefill", "DSPARK", False, False),
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


def test_kvp_decode_refresh_replaces_request_history_plan():
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
        recipe.attn_config, kvp_size=2, kvp_rank=1, kvp_group=(0, 1)
    )
    backend = DeepseekV41AttentionBackend(
        recipe.attn_config, recipe.attn_config.component(DeepseekV41Config)
    )
    pool = _pool(recipe, "cpu")
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(2, max_tokens_per_req=1, overlap_schedule_depth=1)
    tables = _tables("cpu")
    for lengths in ([130, 65], [7], []):
        backend.refresh_decode_metadata(
            2,
            len(lengths),
            torch.arange(len(lengths)),
            torch.tensor(lengths, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=False,
        )
        assert [
            (s.request, s.offset, s.prefix, s.count) for s in backend._kvp_decode_spans
        ] == [(i, i, length - 1, 1) for i, length in enumerate(lengths)]
        assert all(t.device.type == "cpu" for t in backend._kvp_tables_cpu.values())
    # Eager model startup seeds metadata through the inherited capture API.
    backend.init_forward_metadata_capture_cuda_graph(
        2,
        torch.arange(2),
        torch.ones(2, dtype=torch.int32),
        ForwardMode.DECODE,
        block_tables=tables,
    )
    assert backend._kvp_decode_spans == ()
    assert not backend._kvp_tables_cpu
    backend.set_cache_pool(_pool(recipe, "cpu"))
    assert backend._kvp_decode_spans == ()
    assert not backend._kvp_tables_cpu
