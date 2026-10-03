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

"""GPU byte parity for sparse residency, acceptance and slot recycling."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.offload import seed_locations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.offload import SparseKVOffload
from tokenspeed.runtime.layers.attention.kv_cache.offload_config import KVOffloadConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.storage import (
    OffloadFieldWorkspace,
)

register_cuda_ci(est_time=30, suite="runtime-1gpu")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_offload(arena, config):
    workspaces = []
    for name in config.field_ids:
        host = arena.field(name)
        workspaces.append(
            OffloadFieldWorkspace(
                name,
                tuple(host.shape[1:]),
                str(host.dtype).removeprefix("torch."),
                host[0].numel() * host.element_size(),
                ((config.device_rows + 63) // 64) * 64,
                tuple(config.metadata_counts().items()),
                config.temporary_bytes(),
            )
        )
    return SparseKVOffload(arena, config, workspaces=tuple(workspaces))


@pytest.mark.parametrize("queries", [1, 2, 4])
@pytest.mark.parametrize("overlap", [False, True])
def test_sparse_rows_and_accepted_prefix(queries, overlap):
    torch.manual_seed(7)
    names = ("layer.0.latent_kv", "layer.1.latent_kv")
    hosts = {
        name: torch.randint(0, 32767, (1024, 1, 32), dtype=torch.int16).pin_memory()
        for name in names
    }
    config = KVOffloadConfig(
        field_ids=names,
        hot_tokens=64,
        reserved_tokens=8,
        request_slots=4,
        topk=16,
        queries=queries,
        host_budget_bytes=1 << 20,
        overlap=overlap,
        cyclic_tokens=0,
        selection_consumers=(),
        device_rows=512,
        max_extend_tokens=64,
    )
    arena = SimpleNamespace(device="cuda", field=hosts.__getitem__)
    cache = make_offload(arena, config)
    rids = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    for round_id in range(3):
        full = torch.arange(
            700 + round_id * 2 * queries,
            700 + (round_id + 1) * 2 * queries,
            dtype=torch.int32,
            device="cuda",
        )
        positions = (
            torch.arange(queries, device="cuda", dtype=torch.int32).repeat(2)
            + 80
            + round_id * queries
        )
        # Shared prefix, duplicated selected entries, masked entry, two requests.
        selected = torch.randint(
            64 + round_id * 100,
            96 + round_id * 100,
            (2 * queries, 16),
            device="cuda",
            dtype=torch.int32,
        )
        selected[:, 0] = 64
        selected[:, 1] = -1
        selected[:, 2] = selected[:, 3]
        selected[:, -1] = full
        cache.begin(rids, num_extends=0, stream=torch.cuda.current_stream())
        cache.prefetch(names[1], selected, positions, full)
        accepted = torch.tensor([1, queries], dtype=torch.int32, device="cuda")
        before = {name: host.clone() for name, host in hosts.items()}
        for name in names:
            mapped, write_slots = cache.resolve(name, selected, positions, full)
            state = cache.fields[name]
            current = torch.randint(
                0, 32767, (2 * queries, 1, 32), dtype=torch.int16, device="cuda"
            )
            state.device[write_slots.long()] = current
            expected = hosts[name].to("cuda")
            expected[full.long()] = current
            mask = selected >= 0
            assert torch.equal(
                state.device[mapped[mask].long()].view(torch.uint8),
                expected[selected[mask].long()].view(torch.uint8),
            )
            assert (mapped[~mask] == -1).all()
        cache.commit(accepted)
        cache.synchronize()
        for name in names:
            state = cache.fields[name]
            for i, g in enumerate(full.cpu().tolist()):
                expected = (
                    state.device[state.current_hot[i]].cpu()
                    if i % queries < accepted[i // queries].item()
                    else before[name][g]
                )
                assert torch.equal(hosts[name][g], expected)
        if round_id == 1:
            cache.reset_requests(rids, stream=torch.cuda.current_stream())
            for host in hosts.values():
                host[64].fill_(12345)


def test_short_to_long_and_graph_replay():
    name = "layer.0.latent_kv"
    host = torch.arange(512 * 16, dtype=torch.int16).view(512, 1, 16).pin_memory()
    config = KVOffloadConfig(
        field_ids=(name,),
        hot_tokens=32,
        reserved_tokens=8,
        request_slots=3,
        topk=8,
        queries=1,
        host_budget_bytes=1 << 20,
        overlap=True,
        cyclic_tokens=0,
        selection_consumers=(),
        device_rows=512,
        max_extend_tokens=64,
    )
    cache = make_offload(SimpleNamespace(device="cuda", field=lambda _: host), config)
    rids = torch.tensor([1], dtype=torch.int32, device="cuda")
    full = torch.tensor([200], dtype=torch.int32, device="cuda")
    pos = torch.tensor([31], dtype=torch.int32, device="cuda")
    selected = torch.tensor(
        [[64, 65, 66, 67, 68, 69, -1, 200]], dtype=torch.int32, device="cuda"
    )
    accepted = torch.ones(1, dtype=torch.int32, device="cuda")
    value = torch.full((1, 1, 16), 123, dtype=torch.int16, device="cuda")

    def step():
        cache.begin(rids, num_extends=0, stream=torch.cuda.current_stream())
        cache.prefetch(name, selected, pos, full)
        mapped, slots = cache.resolve(name, selected, pos, full)
        cache.fields[name].device[slots.long()] = value
        gathered = cache.fields[name].device[mapped.clamp_min(0).long()]
        cache.commit(accepted)
        return gathered

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = step()
    for position in (31, 32, 33, 63, 64):
        pos.fill_(position)
        full.fill_(position + 200)
        selected[0, -1] = position + 200
        value.fill_(position)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(result[0, :6].cpu(), host[64:70])
        assert torch.equal(host[position + 200], value[0].cpu())


@pytest.mark.parametrize("length", [0, 1, 7, 8, 31, 32, 33])
def test_pd_short_preload(length):
    name = "layer.0.latent_kv"
    host = torch.randint(0, 32767, (256, 1, 16), dtype=torch.int16).pin_memory()
    config = KVOffloadConfig(
        field_ids=(name,),
        hot_tokens=32,
        reserved_tokens=8,
        request_slots=3,
        topk=8,
        queries=1,
        host_budget_bytes=1 << 20,
        overlap=False,
        cyclic_tokens=0,
        selection_consumers=(),
        device_rows=512,
        max_extend_tokens=64,
    )
    cache = make_offload(SimpleNamespace(device="cuda", field=lambda _: host), config)
    rids = torch.tensor([1], dtype=torch.int32, device="cuda")
    table = torch.tensor([[4, 8, 12, 16, 20]], dtype=torch.int32, device="cuda")
    positions = torch.tensor([length], dtype=torch.int32, device="cuda")
    cache.begin(rids, num_extends=0, stream=torch.cuda.current_stream())
    cache.seed(
        name,
        *seed_locations(
            table,
            positions,
            rids,
            page_size=8,
            hot=config.hot_tokens,
            stride=config.buffer_tokens,
            queries=config.queries,
            cyclic=config.cyclic_tokens,
        ),
    )
    torch.cuda.synchronize()
    state = cache.fields[name]
    if length < 32:
        indices = [int(table[0, i // 8]) * 8 + i % 8 for i in range(length)]
        assert torch.equal(state.device[40 : 40 + length].cpu(), host[indices])
        assert state.keys[40 + length : 72].eq(-1).all()
    else:
        assert state.keys.eq(-1).all()


@pytest.mark.parametrize("accepted_count", [0, 1, 2, 4])
@pytest.mark.parametrize("use_graph", [False, True])
def test_mtp_ring_wrap_and_rejected_kv(accepted_count, use_graph):
    name = "layer.0.latent_kv"
    host = torch.randint(0, 32767, (512, 1, 32), dtype=torch.int16).pin_memory()
    config = KVOffloadConfig(
        field_ids=(name,),
        hot_tokens=32,
        reserved_tokens=16,
        request_slots=3,
        topk=16,
        queries=4,
        host_budget_bytes=1 << 20,
        overlap=True,
        cyclic_tokens=12,
        selection_consumers=(),
        device_rows=512,
        max_extend_tokens=64,
    )
    cache = make_offload(SimpleNamespace(device="cuda", field=lambda _: host), config)
    rids = torch.tensor([1], dtype=torch.int32, device="cuda")
    table = torch.arange(8, 56, dtype=torch.int32, device="cuda").view(1, -1)
    accepted = torch.tensor([accepted_count], dtype=torch.int32, device="cuda")
    pos = torch.arange(40, 44, dtype=torch.int32, device="cuda")
    full = pos + 64
    selected = torch.full((4, 16), -1, dtype=torch.int32, device="cuda")
    current = torch.empty((4, 1, 32), device="cuda", dtype=torch.int16)
    state = cache.fields[name]

    def step():
        cache.begin(rids, num_extends=0, stream=torch.cuda.current_stream())
        cache.seed(
            name,
            *seed_locations(
                table,
                pos,
                rids,
                page_size=8,
                hot=config.hot_tokens,
                stride=config.buffer_tokens,
                queries=config.queries,
                cyclic=config.cyclic_tokens,
            ),
        )
        cache.prefetch(name, selected, pos, full)
        mapped, slots = cache.resolve(name, selected, pos, full)
        state.device[slots.long()] = current
        gathered = state.device[mapped.clamp_min(0).long()]
        cache.commit(accepted)
        return gathered

    graph = None
    if use_graph:
        original = host.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gathered = step()
        cache.reset_requests(rids, stream=torch.cuda.current_stream())
        host.copy_(original)
    start = 40
    for _ in range(4):
        pos.copy_(torch.arange(start, start + 4, dtype=torch.int32, device="cuda"))
        full.copy_(pos + 64)
        rows = []
        for position in range(start, start + 4):
            rows.append(
                [64, 65]
                + list(range(position - 7 + 64, position + 1 + 64))
                + [70, 71, 72, 73, 74, 75]
            )
        selected.copy_(torch.tensor(rows, dtype=torch.int32, device="cuda"))
        current.random_(0, 32767)
        before = host.clone()
        expected = host.to("cuda")
        expected[full.long()] = current
        if graph is None:
            gathered = step()
        else:
            graph.replay()
        cache.synchronize()
        assert torch.equal(gathered, expected[selected.long()])
        assert torch.equal(
            host[full[:accepted_count].cpu().long()], current[:accepted_count].cpu()
        )
        assert torch.equal(
            host[full[accepted_count:].cpu().long()],
            before[full[accepted_count:].cpu().long()],
        )
        start += accepted_count


@pytest.mark.parametrize("queries", [1, 4])
def test_attention_output_is_bitwise_after_relocation(queries):
    from tokenspeed_kernel.ops.attention.dsa import dsa_decode

    torch.manual_seed(127)
    name = "layer.1.latent_kv"
    host = torch.randn(16384, 1, 576, dtype=torch.bfloat16).pin_memory()
    config = KVOffloadConfig(
        field_ids=(name,),
        hot_tokens=8192,
        reserved_tokens=64,
        request_slots=3,
        topk=2048,
        queries=queries,
        host_budget_bytes=1 << 28,
        overlap=True,
        cyclic_tokens=0,
        selection_consumers=(),
        device_rows=3 * (8192 + 64),
        max_extend_tokens=64,
    )
    cache = make_offload(SimpleNamespace(device="cuda", field=lambda _: host), config)
    rids = torch.tensor([1], device="cuda", dtype=torch.int32)
    full = torch.arange(10000, 10000 + queries, device="cuda", dtype=torch.int32)
    positions = full.clone()
    selected = torch.randint(
        64, 9500, (queries, 2048), device="cuda", dtype=torch.int32
    )
    selected[:, -1] = full
    selected[:, -5:-1] = -1
    q = torch.randn(queries, 8, 576, device="cuda", dtype=torch.bfloat16)
    cache.begin(rids, num_extends=0, stream=torch.cuda.current_stream())
    cache.prefetch(name, selected, positions, full)
    mapped, slots = cache.resolve(name, selected, positions, full)
    state = cache.fields[name]
    baseline_pool = host.to("cuda")
    state.device[slots.long()] = baseline_pool[full.long()]

    def attend(pool, indices):
        return dsa_decode(
            q=q,
            kv_cache=pool,
            sparse_kv_cache=None,
            topk_slots=indices,
            topk_lens=None,
            max_seqlen_k=16384,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=192**-0.5,
            page_size=64,
            q_len_per_req=queries,
            solution="triton",
        )

    expected = attend(baseline_pool, selected)
    actual = attend(state.device, mapped)
    assert torch.isfinite(actual).all()
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))


def test_deterministic_topk_orders_logical_offsets_before_relocation():
    from tokenspeed_kernel.ops.attention.dsa.deep_gemm import _row_invariant_topk

    torch.manual_seed(15)
    scores = torch.randint(-8, 9, (3, 5000), device="cuda").float()
    lengths = torch.tensor([63, 2048, 4097], device="cuda", dtype=torch.int32)
    scores[:, :16] = float("inf")
    out = torch.empty(3, 2048, device="cuda", dtype=torch.int32)
    _row_invariant_topk(scores, lengths, out, 2048)
    expected = (
        scores.argsort(dim=-1, descending=True, stable=True)[:, :2048]
        .sort(dim=-1)
        .values
    )
    actual = torch.where(out < lengths[:, None], out, -1)
    expected = torch.where(expected < lengths[:, None], expected, -1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("overlap", [False, True])
def test_fixed_pool_masks_graph_padding_and_reuses_highest_slot(overlap):
    name = "layer.0.latent_kv"
    host = torch.randint(0, 32767, (512, 1, 16), dtype=torch.int16).pin_memory()
    config = KVOffloadConfig(
        field_ids=(name,),
        hot_tokens=32,
        reserved_tokens=8,
        request_slots=8,
        topk=8,
        queries=1,
        host_budget_bytes=1 << 20,
        overlap=overlap,
        cyclic_tokens=0,
        selection_consumers=(),
        device_rows=320,
        max_extend_tokens=64,
    )
    cache = make_offload(SimpleNamespace(device="cuda", field=lambda _: host), config)
    requests = torch.tensor([6, 7, 7, 0], device="cuda", dtype=torch.int32)
    selected = torch.tensor(
        [[64, 65, 66, 67, 68, 69, -1, 200]] * 4, device="cuda", dtype=torch.int32
    )
    full = torch.tensor([200, 201, 202, 203], device="cuda", dtype=torch.int32)
    positions = torch.full((4,), 80, device="cuda", dtype=torch.int32)
    accepted = torch.ones(4, device="cuda", dtype=torch.int32)
    value = torch.full((4, 1, 16), 123, device="cuda", dtype=torch.int16)

    def step():
        cache.begin(requests, num_extends=0, stream=torch.cuda.current_stream())
        cache.prefetch(name, selected, positions, full)
        mapped, writes = cache.resolve(name, selected, positions, full)
        cache.fields[name].device[writes.long()] = value
        result = cache.fields[name].device[mapped.clamp_min(0).long()]
        cache.commit(accepted)
        return result, mapped

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, mapped = step()
    for seed in (111, 222):
        cache.reset_requests(requests[:1], stream=torch.cuda.current_stream())
        host[64:70].fill_(seed)
        before = host[201:204].clone()
        graph.replay()
        cache.synchronize()
        assert torch.equal(result[0, :6].cpu(), host[64:70])
        assert mapped[1:].eq(-1).all()
        assert torch.equal(host[201:204], before)


def lru_reference(tags, order, selected, current_slots):
    """Stable batch LRU with deterministic first-occurrence miss enumeration."""
    protected = [
        s for s in order if (tags[s] > 0 and tags[s] in selected) or s in current_slots
    ]
    evictable = [s for s in order if s not in protected]
    misses = list(dict.fromkeys(g for g in selected if g > 0 and g not in tags))
    assert len(misses) <= len(evictable)
    for g, slot in zip(misses, evictable, strict=False):
        tags[slot] = g
    mapped = [tags.index(g) if g > 0 else -1 for g in selected]
    return (
        evictable[len(misses) :] + evictable[: len(misses)] + protected,
        mapped,
        misses,
    )


@pytest.mark.parametrize("queries,cyclic", [(1, 0), (4, 0), (4, 8)])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_lru_multiround_oracle_and_lifecycle(queries, cyclic, overlap, graph_mode):
    import random

    rng = random.Random(613)
    name, hot, reserved, slots = "layer.0.latent_kv", 32, 8, 4
    stride = hot + reserved
    host = torch.arange(2048 * 16, dtype=torch.int16).view(2048, 1, 16).pin_memory()
    config = KVOffloadConfig(
        field_ids=(name,),
        hot_tokens=hot,
        reserved_tokens=reserved,
        request_slots=slots,
        topk=8,
        queries=queries,
        host_budget_bytes=1 << 20,
        overlap=overlap,
        cyclic_tokens=cyclic,
        selection_consumers=(),
        device_rows=192,
        max_extend_tokens=64,
    )
    cache = make_offload(SimpleNamespace(device="cuda", field=lambda _: host), config)
    state = cache.fields[name]
    rids = torch.tensor([1, 2, 3, 3], device="cuda", dtype=torch.int32)
    positions = torch.full((4 * queries,), 80, device="cuda", dtype=torch.int32)
    full = torch.full_like(positions, 1000)
    selected = torch.full((4 * queries, 8), -1, device="cuda", dtype=torch.int32)

    def step():
        cache.begin(rids, num_extends=0, stream=torch.cuda.current_stream())
        cache.prefetch(name, selected, positions, full)
        cache.resolve(name, selected, positions, full)

    if graph_mode:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    cache.clear()
    tags = [[-1] * stride for _ in range(slots)]
    orders = [list(range(hot)) for _ in range(slots)]
    for iteration in range(40):
        # Real slots change batch position; duplicated padding must not mutate null.
        ids = [1, 2] if iteration % 2 else [2, 1]
        rids.copy_(torch.tensor(ids + [3, 3], device="cuda", dtype=torch.int32))
        ps, gs, selections = [], [], []
        expected_maps, expected_misses = [], []
        for b, rid in enumerate(ids):
            pos = hot - queries if iteration == 0 else 80 + iteration * queries
            current = []
            if not cyclic:
                tags[rid][hot:] = [-1] * reserved
            for qi in range(queries):
                g = 1000 + iteration * 2 * queries + b * queries + qi
                p = pos + qi
                slot = hot + p % cyclic if cyclic else (p if p < hot else hot + qi)
                tags[rid][slot] = g
                ps.append(p)
                gs.append(g)
                current.append(slot)
            history = [g for g in tags[rid][:hot] if g > 0]
            pool = history[:12] + list(
                range(64 + iteration % 4 * 12, 88 + iteration % 4 * 12)
            )
            row = [rng.choice(pool) for _ in range(queries * 8)]
            # All-hit and all-invalid rounds must still follow the batch rule.
            if iteration % 7 == 3:
                row = [rng.choice(history) if history else -1 for _ in row]
            if iteration % 7 == 4:
                row = [-1] * len(row)
            row[0] = -1
            row[1] = row[2]  # Duplicates survive attention mapping.
            if iteration % 7 != 4:
                row[-1] = gs[-1]
            orders[rid], mapped, misses = lru_reference(
                tags[rid], orders[rid], row, current
            )
            expected_maps.extend([rid * stride + s if s >= 0 else -1 for s in mapped])
            expected_misses.append(misses)
            selections.extend(row)
        positions.copy_(
            torch.tensor(ps + [90] * (2 * queries), device="cuda", dtype=torch.int32)
        )
        full.copy_(
            torch.tensor(gs + [1900] * (2 * queries), device="cuda", dtype=torch.int32)
        )
        selected.copy_(
            torch.tensor(
                selections + [99] * (2 * queries * 8), device="cuda", dtype=torch.int32
            ).view_as(selected)
        )
        if graph_mode:
            graph.replay()
        else:
            step()
        cache.synchronize()
        actual = state.indices[: selected.numel()].cpu().tolist()
        assert actual == expected_maps + [-1] * (2 * queries * 8)
        assert state.lru_slots.view(slots, hot).cpu().tolist() == orders
        assert state.keys.view(slots, stride).cpu().tolist() == tags
        for b in range(2):
            misses = state.miss_ids[b * queries * 8 : (b + 1) * queries * 8]
            assert misses[misses > 0].cpu().tolist() == expected_misses[b]
            assert state.miss_counts[b].item() == len(expected_misses[b])
            valid = misses > 0
            destinations = state.miss_dst[b * queries * 8 : (b + 1) * queries * 8][
                valid
            ]
            assert torch.equal(
                state.device[destinations.long()].cpu(),
                host[misses[valid].cpu().long()],
            )
        if iteration == 19:
            cache.reset_requests(rids[:1], stream=torch.cuda.current_stream())
            rid = ids[0]
            tags[rid], orders[rid] = [-1] * stride, list(range(hot))
    # Recovery invalidates all partitions because it reuses the shared payload.
    cache.begin(rids[:1], num_extends=1, stream=torch.cuda.current_stream())
    assert state.keys.eq(-1).all()
    assert state.lru_slots.view(slots, hot).cpu().tolist() == [list(range(hot))] * slots
    assert not state.seeded.any()
    cache.clear()
    assert state.device.eq(0).all()


def test_lru_full_union_and_oldest_victim_after_hit_only_round():
    name = "kv"
    host = torch.arange(512 * 16, dtype=torch.int16).view(512, 1, 16).pin_memory()
    config = KVOffloadConfig(
        field_ids=(name,),
        hot_tokens=8,
        reserved_tokens=1,
        request_slots=3,
        topk=8,
        queries=1,
        host_budget_bytes=1 << 20,
        overlap=False,
        cyclic_tokens=0,
        selection_consumers=(),
        device_rows=64,
        max_extend_tokens=16,
    )
    cache = make_offload(SimpleNamespace(device="cuda", field=lambda _: host), config)
    state = cache.fields[name]
    requests = torch.tensor([1], device="cuda", dtype=torch.int32)
    current = torch.tensor([400], device="cuda", dtype=torch.int32)
    position = torch.tensor([64], device="cuda", dtype=torch.int32)
    cases = [
        # Cold union occupies every ordinary slot.
        ([10, 11, 12, 13, 14, 15, 16, 17], list(range(8)), 8),
        # No evictables and no misses; selection order does not reorder hits.
        ([17, 16, 15, 14, 13, 12, 11, 10], list(range(8)), 0),
        # All-hit access promotes slots 0/1 without any host copy.
        ([11, 10, 11, -1, -1, -1, -1, -1], [2, 3, 4, 5, 6, 7, 0, 1], 0),
        # Oldest is slot 2, even though slot 0 is also unselected.
        ([20, -1, -1, -1, -1, -1, -1, -1], [3, 4, 5, 6, 7, 0, 1, 2], 1),
    ]
    for selected, expected_order, expected_count in cases:
        selection = torch.tensor([selected], device="cuda", dtype=torch.int32)
        cache.begin(requests, num_extends=0, stream=torch.cuda.current_stream())
        mapped, _ = cache.resolve(name, selection, position, current)
        assert state.lru_slots.view(3, 8)[1].cpu().tolist() == expected_order
        assert state.miss_counts[0].item() == expected_count
        valid = selection > 0
        assert torch.equal(
            state.device[mapped[valid].long()].cpu(),
            host[selection[valid].cpu().long()],
        )
    assert state.keys[9 + 2].item() == 20
    assert state.keys[9].item() == 10
    # A short request's current ordinary slot is protected even when unselected.
    cache.clear()
    position.zero_()
    selection = torch.tensor(
        [[20, -1, -1, -1, -1, -1, -1, -1]], device="cuda", dtype=torch.int32
    )
    cache.begin(requests, num_extends=0, stream=torch.cuda.current_stream())
    mapped, _ = cache.resolve(name, selection, position, current)
    assert mapped[0, 0].item() == 10
    assert state.keys[9].item() == 400
    assert state.lru_slots.view(3, 8)[1].cpu().tolist() == [2, 3, 4, 5, 6, 7, 1, 0]
