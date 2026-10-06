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

"""TreeSpec hidden-row and position compaction against a plain per-request reference."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.tree_spec import TreeSpec, TreeSpecConfig
from tokenspeed.runtime.layers.attention.backends.paged.router import (
    CacheGroupRouter,
    RouterDecodeWriteLocations,
)
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeVerifyInputs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("nodes", [8, 64])
def test_compact_rows_packs_hidden_and_positions(nodes):
    gen = torch.Generator().manual_seed(nodes)
    bs, hidden_size = 3, 3000
    spec = TreeSpec(
        TreeSpecConfig(topk=4, num_steps=5, num_nodes=nodes),
        bs,
        torch.device("cuda"),
    )
    paths = [[0, 2, 5], [0, 1, 2, 3], [0]]  # jump, identity, root only
    path = torch.full((bs, nodes), -1, dtype=torch.int32)
    for b, p in enumerate(paths):
        path[b, : len(p)] = torch.tensor(p, dtype=torch.int32)
    spec.depth_buf[:bs] = torch.randint(0, 5, (bs, nodes), generator=gen).int().cuda()
    hidden = torch.randn(bs * nodes, hidden_size, generator=gen).bfloat16().cuda()
    positions = torch.randint(0, 1000, (bs * nodes,), generator=gen).cuda()

    want_hidden = hidden.clone()
    for b, p in enumerate(paths):
        rows = b * nodes + torch.tensor(p)
        want_hidden[b * nodes : b * nodes + len(p)] = hidden[rows]
    offsets = torch.arange(nodes, device="cuda").repeat(bs)
    want_positions = positions + offsets - spec.depth_buf[:bs].reshape(-1).long()

    spec.compact_rows(path.cuda(), hidden, positions)

    assert torch.equal(hidden, want_hidden)
    assert torch.equal(positions, want_positions)


@pytest.mark.parametrize("nodes", [8, 64])
def test_fresh_spec_is_the_chain(nodes):
    """Graph warmup runs compact_rows without a load_step; it must leave positions alone."""
    spec = TreeSpec(
        TreeSpecConfig(topk=1, num_steps=nodes - 1, num_nodes=nodes),
        3,
        torch.device("cuda"),
    )
    positions = torch.arange(3 * nodes, dtype=torch.int64, device="cuda") + 100
    before = positions.clone()
    hidden = torch.randn(3 * nodes, 64, device="cuda").bfloat16()
    root_only = torch.full((3, nodes), -1, dtype=torch.int32, device="cuda")
    root_only[:, 0] = 0
    spec.compact_rows(root_only, hidden, positions)
    assert torch.equal(positions, before)
    fresh = (spec.depth_buf.clone(), spec.mask_buf.clone())
    chain = torch.arange(-1, nodes - 1, dtype=torch.int32, device="cuda").repeat(8, 1)
    spec.load_step(3, torch.tensor([0, 1, 2], device="cuda"), chain)
    assert torch.equal(spec.depth_buf, fresh[0])
    assert torch.equal(spec.mask_buf, fresh[1])


def test_router_compacts_each_group_at_its_window_and_follows_the_pool():
    """Each cache group moves rows at its own window, also in a region another
    group shares (disjoint pages); layers aliasing one K region within a group
    move it once; a new pool gets its own tables."""
    nodes, bs, slots = 4, 2, 32
    router = CacheGroupRouter(
        None,
        is_draft=False,
        spec_num_tokens=nodes,
        device="cuda",
        consumed_group_ids=None,
    )
    router.bind_tree_verify(
        TreeVerifyInputs(
            torch.zeros(bs * nodes, dtype=torch.int64, device="cuda"),
            nodes,
            torch.zeros(bs, nodes, dtype=torch.int32, device="cuda"),
        )
    )
    locations = {
        "a": torch.tensor([3, 4, 5, 6, 20, 21, 22, 23], dtype=torch.int32).cuda(),
        "b": torch.tensor([8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int32).cuda(),
    }
    # The groups this router serves; compaction reads only their names.
    router.leaves = dict.fromkeys(locations)
    router.decode_write_locations = RouterDecodeWriteLocations(
        tokens_per_req=nodes, by_group=locations
    )
    path = torch.tensor([[0, 2, 3, -1], [0, 1, 3, -1]], dtype=torch.int32).cuda()

    def pool():
        k0, v0, v1 = (torch.randn(slots, 2, 8, device="cuda") for _ in range(3))
        buffers = {
            0: (k0, v0),
            1: (k0, v1),  # layer 1's K aliases layer 0's
            2: (k0, v1),  # group b shares group a's regions
        }
        return buffers, SimpleNamespace(
            history_group_by_layer=lambda: {0: "a", 1: "a", 2: "b"},
            get_kv_buffer=buffers.__getitem__,
        )

    def compacted(buf, windows):
        want = buf.clone()
        for window in windows:
            for b in range(bs):
                for depth, row in enumerate(path[b].tolist()):
                    if row >= 0:
                        want[window[b * nodes + depth]] = buf[window[b * nodes + row]]
        return want

    for _ in range(2):
        buffers, cache_pool = pool()
        router.cache_pool = cache_pool
        router._bind_tree_window_rows()
        regions = [
            (buffers[0][0], ("a", "b")),
            (buffers[0][1], ("a",)),
            (buffers[1][1], ("a", "b")),
        ]
        want = [
            compacted(buf, [locations[gid] for gid in groups])
            for buf, groups in regions
        ]
        router.compact_verify_window(path)
        for (got, _), expected in zip(regions, want):
            assert torch.equal(got, expected)
