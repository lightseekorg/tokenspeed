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

"""Runtime row counts must not specialize offload kernels."""

from unittest.mock import patch

import pytest
import torch
from cutlass import cute
from tokenspeed_kernel.ops.kvcache import offload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rid_dtype", [torch.int32, torch.int64])
def test_dynamic_rows_do_not_compile(rid_dtype):
    host = torch.arange(512 * 16, dtype=torch.int16).reshape(512, 1, 16).pin_memory()
    device = torch.zeros((32 * 40, 1, 16), dtype=torch.int16, device="cuda")
    rids = torch.arange(32, dtype=rid_dtype, device="cuda")
    positions = torch.full((32,), 5, dtype=torch.int32, device="cuda")
    full = rids.to(torch.int32) + 200
    keys = torch.full((32 * 40,), -1, dtype=torch.int32, device="cuda")
    slots = torch.empty_like(rids)
    accepted = torch.ones_like(rids)
    accepted_full = torch.empty_like(rids)
    seeded = torch.zeros_like(rids)
    lru = torch.empty(32 * 32, device="cuda", dtype=torch.int32)
    order = torch.empty_like(lru)
    free_counts = torch.empty_like(rids)
    miss_counts = torch.empty_like(rids)
    miss_ids = torch.empty(32 * 8, device="cuda", dtype=torch.int32)
    miss_dst = torch.empty_like(miss_ids)
    output = torch.empty_like(miss_ids)
    entry_dest = torch.empty_like(miss_ids)
    hash_keys = torch.empty(0, device="cuda", dtype=torch.int32)
    hash_owners = torch.empty_like(hash_keys)

    def run(n):
        selected = rids[:n]
        offload.current_slots(
            selected,
            positions[:n],
            full[:n],
            keys,
            slots[:n],
            hot=32,
            stride=40,
            queries=1,
            cyclic=0,
        )
        offload.copy_rows(host, device, full[:n], slots[:n], writeback=False)
        offload.accepted_ids(full[:n], accepted[:n], accepted_full[:n], queries=1)
        seeded.zero_()
        history = (
            torch.arange(64, 96, dtype=torch.int32, device="cuda")
            .expand(n, -1)
            .contiguous()
        )
        hot_rows = selected.to(torch.int32)[:, None] * 40 + torch.arange(
            32, dtype=torch.int32, device="cuda"
        )
        offload.seed_rows(host, device, keys, seeded, selected, history, hot_rows)
        offload.reset_lru(lru, None, hot=32)
        offload.reset_lru(lru, selected, hot=32)
        offload.materialize(
            history[:, :8].contiguous(),
            selected,
            keys,
            slots[:n],
            miss_ids,
            miss_dst,
            output,
            host,
            device,
            lru,
            order,
            free_counts,
            miss_counts,
            entry_dest,
            hash_keys,
            hash_owners,
            hot=32,
            stride=40,
            queries=1,
        )

    for n in (1, 2, 16):
        run(n)
    with patch.object(cute, "compile", wraps=cute.compile) as compile_call:
        for n in (0, 3, 4, 17, 31):
            run(n)
    assert compile_call.call_count == 0
    torch.cuda.synchronize()
    assert torch.equal(accepted_full[:31], full[:31])


@pytest.mark.parametrize("queries", [1, 4, 8])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_hash_large_union_and_replay(queries, graph_mode):
    """Exercise real Q4 shared size, global tables, duplicate tags and padding."""
    h, stride, k, r = 4096, 5124, 2048, 4
    n = queries * k
    ints = lambda count, fill=0: torch.full(
        (count,), fill, dtype=torch.int32, device="cuda"
    )
    host = torch.arange(10000 * 17, dtype=torch.uint8).view(10000, 1, 17).pin_memory()
    device = torch.zeros((r * stride, 1, 17), dtype=torch.uint8, device="cuda")
    keys = ints(r * stride, -1)
    rids = torch.tensor([1, 2, 0], device="cuda", dtype=torch.int64)
    selected = (
        ((torch.arange(n, device="cuda", dtype=torch.int32) * 37) % 2048 + 1)
        .repeat(3)
        .view(3 * queries, k)
    )
    current = ints(3 * queries)
    miss, dst, output, dest = [ints(r * n) for _ in range(4)]
    lru = torch.arange(h, device="cuda", dtype=torch.int32).repeat(r)
    order, free, count = ints(r * h), ints(r), ints(r)
    table, shared = offload.hash_geometry(queries, k)
    hk, ho = ints(0 if shared else r * table), ints(0 if shared else r * table)

    def step():
        offload.materialize(
            selected,
            rids,
            keys,
            current,
            miss,
            dst,
            output,
            host,
            device,
            lru,
            order,
            free,
            count,
            dest,
            hk,
            ho,
            hot=h,
            stride=stride,
            queries=queries,
        )

    step()
    graph = torch.cuda.CUDAGraph()
    if graph_mode:
        with torch.cuda.graph(graph):
            step()
    for iteration in range(3):
        keys.fill_(-1)
        offload.reset_lru(lru, None, hot=h)
        # Two resident copies must map to the smaller physical slot, even
        # though LRU traversal encounters that copy last.
        keys[stride + 2] = 1
        keys[stride + 19] = 1
        lru.view(r, h)[1] = torch.arange(h - 1, -1, -1, device="cuda")
        device[stride + 2].copy_(host[1], non_blocking=True)
        device[stride + 19].copy_(host[1], non_blocking=True)
        if graph_mode:
            graph.replay()
        else:
            step()
        torch.cuda.synchronize()
        mapped = output[:n].long()
        assert mapped[0] == stride + 2
        assert output[2 * n : 3 * n].eq(-1).all()
        assert count[1].item() == 2048
        assert torch.equal(
            device[output[n : 2 * n].long()].cpu(),
            host[selected.flatten()[n : 2 * n].cpu().long()],
        )
        assert count[0].item() == 2047
        assert keys[:stride].eq(-1).all()
        assert torch.equal(
            device[mapped].cpu(), host[selected.flatten()[:n].cpu().long()]
        )
        canonical = list(dict.fromkeys(selected.flatten()[:n].cpu().tolist()))[1:]
        assert miss[:n][miss[:n] > 0].cpu().tolist() == canonical
        # An all-hit round must preserve both copies, mapping and bytes.
        step()
        assert count[0].item() == 0
        assert torch.equal(output[:n].long(), mapped)


def test_hash_collisions_and_large_positive_ids():
    """A full collision chain, invalid entries and duplicate high-valued tags."""

    def bucket(key):
        x = ((key ^ (key >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
        x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
        return (x ^ (x >> 16)) & 255

    collide = []
    value = 1
    while len(collide) < 64:
        if bucket(value) == 0:
            collide.append(value)
        value += 1
    h, stride, n, r = 128, 132, 128, 3
    ints = lambda count, fill=0: torch.full(
        (count,), fill, dtype=torch.int32, device="cuda"
    )
    host = torch.arange(value * 4, dtype=torch.int32).view(value, 1, 4).pin_memory()
    device = torch.zeros((r * stride, 1, 4), dtype=torch.int32, device="cuda")
    keys = ints(r * stride, -1)
    rids, current = ints(1, 1), ints(1)
    high = 2147483646
    keys[stride + 4] = high
    keys[stride + 17] = high
    selected = torch.tensor(
        (collide + [high, 0, -1] + collide[:61]), dtype=torch.int32, device="cuda"
    ).view(1, n)
    miss, dst, output, dest = [ints(n) for _ in range(4)]
    lru = torch.arange(h, device="cuda", dtype=torch.int32).repeat(r)
    order, free, count, empty = ints(r * h), ints(r), ints(r), ints(0)
    offload.materialize(
        selected,
        rids,
        keys,
        current,
        miss,
        dst,
        output,
        host,
        device,
        lru,
        order,
        free,
        count,
        dest,
        empty,
        empty,
        hot=h,
        stride=stride,
        queries=1,
    )
    torch.cuda.synchronize()
    assert count[0].item() == 64
    assert output[64].item() == stride + 4
    assert output[65:67].tolist() == [-1, -1]
    assert output[:61].tolist() == output[67:].tolist()
    assert torch.equal(device[output[:64].long()].cpu(), host[collide])
    assert miss[miss > 0].tolist() == collide
