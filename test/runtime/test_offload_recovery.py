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

"""Recovery chunks use the same canonical history and bounded compute storage."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.offload_adapter import (
    KVOffloadAdapter,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.offload_config import KVOffloadPolicy
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.storage import (
    plan_cache_storage,
)


def storage(queries):
    group = CacheGroupSpec(
        group_id="history",
        retention="full_history",
        rows_per_page=64,
        entry_stride_tokens=1,
        transfer_policy="full_suffix",
        replayable=False,
    )
    names = ("layer.0.latent_kv", "layer.1.latent_kv")
    layout = pack(
        (
            (
                group,
                tuple(CacheFieldSpec(n, n, (64, 1, 576), "bfloat16") for n in names),
            ),
        ),
        prefix_granularity=32,
        cache_blocks_per_lcm_block={"history": 1},
        alignment=256,
        max_padding_fraction=0.25,
    )
    policy = KVOffloadPolicy(
        (names[1],),
        512,
        queries,
        512,
        queries,
        1 << 24,
        True,
        0,
        ((names[0], names[1]),),
    )
    config = policy.bind(request_slots=4, device_rows=2112, max_extend_tokens=64)
    plan = layout.bind(128)
    return names, group, plan, plan_cache_storage(plan, (group,), offload=config)


def test_row_bytes_and_alignment_do_not_use_prefix_identity():
    _, _, plan, physical = storage(1)
    workspace = physical.workspaces[0]
    assert plan.prefix_granularity == 32
    assert workspace.row_bytes == 576 * 2
    assert workspace.device_rows % 64 == 0
    assert physical.offload.reserved_tokens == 1
    assert physical.host_bytes + physical.device_history_bytes == plan.arena_bytes
    assert (
        physical.fixed_device_bytes
        == workspace.persistent_bytes + workspace.temporary_bytes
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("queries", [1, 4])
@pytest.mark.parametrize("solution", ["triton", "flashmla"])
def test_recovery_chunks_commit_all_rows_and_resume_decode_bitwise(queries, solution):
    from tokenspeed_kernel.ops.attention.dsa import dsa_prefill

    torch.manual_seed(103)
    names, group, plan, physical = storage(queries)
    arena = CacheArena(plan, "cuda", cache_group_specs=(group,), storage_plan=physical)
    host = arena.field(names[1])
    host.copy_(torch.randn_like(host))
    baseline = host.to("cuda")
    table = (torch.randperm(80, device="cuda", dtype=torch.int32) + 1).view(1, -1)
    full = (
        table[:, :, None] * 64 + torch.arange(64, device="cuda", dtype=torch.int32)
    ).flatten()
    pool = SimpleNamespace(
        arena=arena,
        layer_fields={(0, "latent_kv"): names[0], (1, "latent_kv"): names[1]},
    )
    router = SimpleNamespace(
        spec_num_tokens=queries,
        group_view=lambda group, bs: SimpleNamespace(
            page_table=table, kernel_page_size=64
        ),
    )
    adapter = KVOffloadAdapter(router, pool)
    adapter.validate_slots(4)
    with pytest.raises(ValueError, match="slots"):
        adapter.validate_slots(3)
    layer = SimpleNamespace(layer_id=1, group_id="history")
    rids = torch.tensor([2], device="cuda", dtype=torch.int32)
    device = arena.compute_field(names[1])
    pointer = device.data_ptr()
    for prefix in (0, 2048, 4096):
        count = 63
        writes = full[prefix : prefix + count]
        adapter.set_extend_writes({"history": writes})
        adapter.begin(rids, num_extends=1, stream=torch.cuda.current_stream())
        positions = torch.arange(
            prefix, prefix + count, device="cuda", dtype=torch.int32
        )
        logical = torch.stack(
            [
                torch.randint(0, p + 1, (512,), device="cuda")
                for p in range(prefix, prefix + count)
            ]
        )
        selected = full[logical]
        selected[:, -1] = writes
        selected[:, 0] = -1
        selected[:, 1] = selected[:, 2]
        compute_writes = adapter.prepare(layer, selected, positions, ForwardMode.EXTEND)
        values = torch.randn((count, 1, 576), device="cuda", dtype=torch.bfloat16)
        device[compute_writes.long()] = values
        baseline[writes.long()] = values
        q = torch.randn((count, 8, 576), device="cuda", dtype=torch.bfloat16)

        def attend(cache, indices, query_slice, query, query_positions):
            return dsa_prefill(
                q=query[query_slice],
                kv_cache=cache,
                sparse_kv_cache=None,
                topk_slots=indices,
                topk_lens=torch.full(
                    (indices.shape[0],), 511, device="cuda", dtype=torch.int32
                ),
                kv_seq_lens=query_positions[query_slice] + 1,
                max_seqlen_k=8192,
                qk_nope_head_dim=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                softmax_scale=192**-0.5,
                page_size=64,
                solution=solution,
            )

        expected = attend(baseline, selected, slice(None), q, positions)
        actual = torch.cat(
            [
                attend(device, mapped, query_slice, q, positions)
                for query_slice, mapped in adapter.prefill_tiles(layer, selected)
            ]
        )
        assert torch.isfinite(actual).all()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        # MTP's decode acceptance never truncates a computed recovery chunk.
        arena.offload.commit(torch.ones(1, device="cuda", dtype=torch.int32))
        arena.offload.synchronize()
        assert torch.equal(host[writes.cpu().long()], values.cpu())
        assert device.data_ptr() == pointer

    # Recovery invalidated every hot partition, including other suspended slots.
    assert arena.offload.fields[names[1]].keys.eq(-1).all()
    current = full[4159 : 4159 + queries]
    selected = full[:64].expand(queries, -1).clone()
    selected[:, -1] = current
    router.write_locations = lambda layer, mode: current
    adapter.begin(rids, num_extends=0, stream=torch.cuda.current_stream())
    adapter.prepare(
        layer,
        selected,
        torch.arange(4159, 4159 + queries, device="cuda", dtype=torch.int32),
        ForwardMode.DECODE,
    )
    device[adapter.writes[1].long()] = baseline[current.long()]
    assert torch.equal(
        device[adapter.read_indices(layer, selected).long()],
        baseline[selected.long()],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_seed_mapping_runtime_table_extent_does_not_recompile():
    from unittest.mock import patch

    from tokenspeed_kernel.ops.kvcache.offload import seed_locations

    requests = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    positions = torch.tensor([12, 19], device="cuda", dtype=torch.int32)

    def run(width):
        table = torch.arange(
            1, 2 * width + 1, device="cuda", dtype=torch.int32
        ).reshape(2, width)
        history, hot_rows = seed_locations(
            table,
            positions,
            requests,
            page_size=8,
            hot=32,
            stride=33,
            queries=1,
            cyclic=0,
        )
        for row, length in enumerate((12, 19)):
            logical = torch.arange(length, device="cuda")
            assert torch.equal(
                history[row, :length], table[row, logical // 8] * 8 + logical % 8
            )
            assert history[row, length:].eq(-1).all()
            assert torch.equal(hot_rows[row, :length], requests[row] * 33 + logical)

    run(4)
    from cutlass import cute

    with patch.object(cute, "compile", wraps=cute.compile) as compile_call:
        for width in (5, 16, 17, 32):
            run(width)
    assert compile_call.call_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_unguarded_step_consumption_fails_fast(monkeypatch):
    """History queries remain available; offload compute requires preparation.

    A missing prepare_sparse_kv_access must fail loudly: the silent fallbacks
    (unmapped indices, history write slots) are correct only for layers
    that consume no offloaded plane.
    """
    names, group, plan, physical = storage(1)
    arena = CacheArena(plan, "cuda", cache_group_specs=(group,), storage_plan=physical)
    table = (torch.randperm(80, device="cuda", dtype=torch.int32) + 1).view(1, -1)
    pool = SimpleNamespace(
        arena=arena,
        layer_fields={(0, "latent_kv"): names[0], (1, "latent_kv"): names[1]},
    )
    router = CacheGroupRouter(
        lambda group, page_size: None,
        is_draft=False,
        spec_num_tokens=1,
        device="cuda",
        consumed_group_ids=("history",),
    )
    router.leaves = {"history": object()}
    monkeypatch.setattr(
        router,
        "group_view",
        lambda group, bs: SimpleNamespace(page_table=table, kernel_page_size=64),
    )
    history = torch.tensor([64], device="cuda", dtype=torch.int32)
    router.decode_write_locations = SimpleNamespace(
        by_group={"history": history}, tokens_per_req=1
    )
    router._extend_write_locations = {"history": history + 1}
    adapter = KVOffloadAdapter(router, pool)
    layer = SimpleNamespace(layer_id=1, group_id="history")
    bystander = SimpleNamespace(layer_id=7, group_id="history")
    router._offload_adapter = adapter
    rids = torch.tensor([2], device="cuda", dtype=torch.int32)
    adapter.begin(rids, num_extends=0, stream=torch.cuda.current_stream())

    # A layer that consumes no offloaded plane keeps its graceful fallback.
    sentinel = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    assert adapter.read_indices(bystander, sentinel) is sentinel
    assert adapter.compute_write_locations(bystander, ForwardMode.DECODE) is history
    assert router.write_locations(layer, ForwardMode.DECODE) is history

    # The offload layer must fail loudly before its prepare.
    with pytest.raises(RuntimeError, match="prepare_sparse_kv_access"):
        adapter.read_indices(layer, sentinel)
    with pytest.raises(RuntimeError, match="prepare_sparse_kv_access"):
        router.forward(
            sentinel,
            None,
            None,
            layer,
            pool,
            ForwardMode.DECODE,
            1,
            save_kv_cache=False,
        )

    # After the decode prepare, same-family consumption passes.
    selected = table[0, :64].view(1, -1).clone()
    compute_writes = router.prepare_sparse_kv_access(
        layer,
        selected,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    assert adapter.read_indices(layer, sentinel) is not sentinel
    assert adapter.compute_write_locations(layer, ForwardMode.DECODE) is compute_writes
    assert router.write_locations(layer, ForwardMode.DECODE) is history
    assert not torch.equal(compute_writes, history)

    # Cross-family consumption stays rejected.
    with pytest.raises(RuntimeError, match="prepare_sparse_kv_access"):
        router.forward(
            sentinel,
            None,
            None,
            layer,
            pool,
            ForwardMode.EXTEND,
            1,
            save_kv_cache=False,
        )
    # Recovery attention cannot consume a decode prepare's staging map.
    with pytest.raises(RuntimeError, match="prepare_sparse_kv_access"):
        adapter.prefill_tiles(layer, selected)
    adapter.begin(rids, num_extends=0, stream=torch.cuda.current_stream())
    with pytest.raises(RuntimeError, match="prepare_sparse_kv_access"):
        router.forward(
            sentinel,
            None,
            None,
            layer,
            pool,
            ForwardMode.DECODE,
            1,
            save_kv_cache=False,
        )
    assert router.write_locations(layer, ForwardMode.DECODE) is history


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("queries", [1, 4])
@pytest.mark.parametrize("use_graph", [False, True])
def test_router_keeps_history_queries_and_compute_access_distinct(
    monkeypatch, queries, use_graph
):
    """Projection and leaf reads use hot rows while Index-K keeps history IDs."""
    names, group, plan, physical = storage(queries)
    arena = CacheArena(plan, "cuda", cache_group_specs=(group,), storage_plan=physical)
    host = arena.field(names[1])
    host.copy_(torch.randn_like(host))
    device = arena.compute_field(names[1])
    table = torch.arange(1, 81, device="cuda", dtype=torch.int32).view(1, -1)
    full = (table[:, :, None] * 64 + torch.arange(64, device="cuda")).flatten()
    writes = full[4159 : 4159 + queries].to(torch.int32).clone()
    positions = torch.arange(4159, 4159 + queries, device="cuda", dtype=torch.int32)
    selection = full[:64].to(torch.int32).expand(queries, -1).clone()
    selection[:, 0] = -1
    selection[:, 1] = selection[:, 2]
    selection[:, -1] = writes
    values = torch.ones((queries, 1, 576), device="cuda", dtype=torch.bfloat16)
    accepted_count = 1 if queries == 1 else 2
    accepted = torch.tensor([accepted_count], device="cuda", dtype=torch.int32)
    request_slots = torch.tensor([2], device="cuda", dtype=torch.int32)
    index_k = torch.zeros(host.shape[0], device="cuda", dtype=torch.bfloat16)
    router = CacheGroupRouter(
        lambda group, page_size: None,
        is_draft=False,
        spec_num_tokens=queries,
        device="cuda",
        consumed_group_ids=("history",),
    )
    monkeypatch.setattr(
        router,
        "group_view",
        lambda group, bs: SimpleNamespace(page_table=table, kernel_page_size=64),
    )
    router.decode_write_locations = SimpleNamespace(
        by_group={"history": writes}, tokens_per_req=queries
    )
    pool = SimpleNamespace(
        arena=arena,
        layer_fields={(0, "latent_kv"): names[0], (1, "latent_kv"): names[1]},
    )
    router._offload_adapter = KVOffloadAdapter(router, pool)
    owner = SimpleNamespace(layer_id=0, group_id="history")
    consumer = SimpleNamespace(layer_id=1, group_id="history")

    def leaf_decode(q, k, v, layer, out_cache_loc, pool, bs, **kwargs):
        indices = kwargs["topk_indices"]
        gathered = device[indices.clamp_min(0).long()]
        return torch.where(indices[..., None, None] >= 0, gathered, 0)

    router.leaves = {"history": SimpleNamespace(forward_decode=leaf_decode)}

    def step():
        router.prepare_cache_batch(
            request_slots, num_extends=0, stream=torch.cuda.current_stream()
        )
        history_writes = router.write_locations(owner, ForwardMode.DECODE)
        index_k[history_writes.long()] = values[:, 0, 0]
        router.prefetch_sparse_kv(owner, selection, positions)
        compute_writes = router.prepare_sparse_kv_access(
            consumer, selection, positions, forward_mode=ForwardMode.DECODE
        )
        assert (
            router.forward_write_locations(consumer, ForwardMode.DECODE)
            is compute_writes
        )
        assert (
            router.padded_write_locations(consumer, ForwardMode.DECODE, queries)
            is compute_writes
        )
        device[compute_writes.long()] = values
        output = router.forward(
            values,
            values,
            None,
            consumer,
            None,
            ForwardMode.DECODE,
            1,
            save_kv_cache=False,
            topk_indices=selection,
        )
        router.writeback_accepted_kv(accepted)
        return output

    if use_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = step()
        torch.cuda.synchronize()
    for offset in range(3):
        positions.copy_(
            torch.arange(4163 + offset, 4163 + offset + queries, device="cuda")
        )
        writes.copy_(full[4163 + offset : 4163 + offset + queries])
        selection[:, -1] = writes
        original_selection = selection.clone()
        values.fill_(offset + 11)
        before = host[writes.cpu().long()].clone()
        expected_cache = host.to("cuda")
        expected_cache[writes.long()] = values
        expected = expected_cache[selection.clamp_min(0).long()]
        expected = torch.where(selection[..., None, None] >= 0, expected, 0)
        if use_graph:
            graph.replay()
        else:
            actual = step()
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)
        assert torch.equal(selection, original_selection)
        assert torch.equal(index_k[writes.long()], values[:, 0, 0])
        assert torch.equal(
            host[writes[:accepted_count].cpu().long()], values[:accepted_count].cpu()
        )
        assert torch.equal(
            host[writes[accepted_count:].cpu().long()], before[accepted_count:]
        )
