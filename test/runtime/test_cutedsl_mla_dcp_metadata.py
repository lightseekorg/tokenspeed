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

"""CuTe leaf metadata lifecycle across full refresh and draft length updates."""

import pytest
import torch

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def make_leaf(monkeypatch):
    from tokenspeed.runtime.layers.attention.backends.paged import tokenspeed_mla

    # These tests execute real metadata kernels, but never prefill attention.
    monkeypatch.setattr(tokenspeed_mla, "warmup_compile_prefill", lambda **kwargs: None)

    def make(*, queries, draft, block, rank, degree, context=512):
        spec = MLAConfig(
            backend_name="tokenspeed_mla",
            num_attention_heads=128,
            num_kv_heads=1,
            head_dim=576,
            attn_tp_size=8,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            scaling=192**-0.5,
            kv_cache_dim=576,
        )
        config = AttnConfig(
            device="cuda",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_quant_method="",
            prefix_granularity=128,
            context_len=context,
            max_bs=8,
            speculative_num_draft_tokens=queries,
            is_draft=draft,
            draft_block_decode=block,
            kvp_size=degree,
            kvp_rank=rank,
            kvp_group=tuple(range(degree)),
            components=(spec,),
        )
        leaf = tokenspeed_mla.CuteDSLMLABackend(config, spec, kernel_page_size=64)
        # The metadata lifecycle needs a binding identity, not cache payload.
        leaf.set_cache_pool(object())
        leaf.configure_runtime(
            block_granularity=128, virtual_block_count=33, shard_count=degree
        )
        leaf.init_cuda_graph_state(8)
        return leaf

    return make


def _pages(blocks):
    return torch.tensor(
        [[2 * b + sub for b in row for sub in range(2)] for row in blocks],
        dtype=torch.int32,
        device="cuda",
    )


def _check(metadata, blocks, ends, *, rank, degree, queries, block):
    expected = []
    for row, length in zip(blocks, ends):
        owned = [b > 0 and (b - 1) % degree == rank for b in row for _ in range(128)]
        bounds = (
            [length] * queries
            if block
            else [max(0, length - queries + q + 1) for q in range(queries)]
        )
        expected.append([sum(owned[:end]) for end in bounds])
    assert metadata.dcp.local_visible_lens.tolist() == expected
    assert metadata.dcp.local_seq_lens.tolist() == [row[-1] for row in expected]
    for idx, row in enumerate(blocks):
        pages = [
            ((b - 1) // degree + 1) * 2 + sub
            for b in row
            if b > 0 and (b - 1) % degree == rank
            for sub in range(2)
        ]
        got = metadata.dcp.local_page_table[idx].tolist()
        assert got == pages + [0] * (len(got) - len(pages))


def _pointers(metadata):
    return tuple(
        t.data_ptr()
        for t in (
            metadata.page_table,
            metadata.seq_lens_k,
            metadata.dcp.virtual_page_table,
            metadata.dcp.local_page_table,
            metadata.dcp.page_prefix,
            metadata.dcp.local_visible_lens,
            metadata.dcp.local_seq_lens,
        )
    )


@pytest.mark.parametrize(
    "queries,draft,block,rank,degree",
    [
        pytest.param(1, False, False, 0, 2, id="decode"),
        pytest.param(4, False, False, 0, 2, id="verify"),
        pytest.param(4, True, False, 0, 2, id="mtp"),
        pytest.param(4, True, True, 0, 2, id="block-draft"),
        pytest.param(4, True, False, 1, 2, id="mtp-rank1"),
        pytest.param(4, True, False, 3, 4, id="mtp-degree4"),
    ],
)
def test_refresh_and_draft_updates_reuse_reserved_pages(
    make_leaf, queries, draft, block, rank, degree
):
    leaf = make_leaf(
        queries=queries, draft=draft, block=block, rank=rank, degree=degree
    )
    blocks = [[5, 2, 8, 1], [2, 5, 1, 8], [0, 0, 0, 0]]
    table = _pages(blocks)
    ends = torch.tensor([127, 255, 1], dtype=torch.int32, device="cuda")
    leaf.refresh_decode_metadata(3, 2, ends, table, num_extends=1)
    metadata = leaf.forward_decode_metadata
    pointers = _pointers(metadata)
    effective = [127, 255, max(queries if not draft or block else 1, 1)]
    _check(
        metadata,
        blocks,
        effective,
        rank=rank,
        degree=degree,
        queries=queries,
        block=block,
    )
    assert metadata.num_extends == 1
    table_before = metadata.dcp.local_page_table.clone()
    prefix_before = metadata.dcp.page_prefix.clone()

    def advance():
        if block:
            leaf.fill_block_decode_seq_lens(3, ends)
        else:
            leaf.advance_draft_forward_metadata(ends)

    advance()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        advance()
    for values in (
        [129, 257, 1],
        [65, 129, 1],
        [256, 512, 1],
        [1, 1, 1],
        [131, 259, 1],
    ):
        ends.copy_(torch.tensor(values, device="cuda"))
        graph.replay()
        effective = [max(v, queries) if block else v for v in values]
        _check(
            metadata,
            blocks,
            effective,
            rank=rank,
            degree=degree,
            queries=queries,
            block=block,
        )
        replay = metadata.dcp.local_visible_lens.clone()
        advance()
        torch.testing.assert_close(
            metadata.dcp.local_visible_lens, replay, atol=0, rtol=0
        )
        torch.testing.assert_close(metadata.dcp.local_page_table, table_before)
        torch.testing.assert_close(metadata.dcp.page_prefix, prefix_before)
        assert _pointers(metadata) == pointers
        if block:
            assert metadata.block_seq_lens.tolist() == effective
        else:
            assert leaf.decode_seq_lens_buffer[:3].tolist() == values


@pytest.mark.parametrize("block", [False, True])
def test_narrower_refresh_clears_old_reserved_pages(make_leaf, block):
    leaf = make_leaf(queries=4, draft=block, block=block, rank=0, degree=2)
    ends = torch.full((3,), 65, dtype=torch.int32, device="cuda")
    leaf.refresh_decode_metadata(
        3, 2, ends, _pages([[5, 2, 8, 1], [2, 5, 1, 8], [0, 0, 0, 0]])
    )
    metadata = leaf.forward_decode_metadata
    pointers = _pointers(metadata)
    short_blocks = [[2], [5], [0]]
    leaf.refresh_decode_metadata(3, 2, ends, _pages(short_blocks))
    assert leaf.forward_decode_metadata is metadata
    assert _pointers(metadata) == pointers
    _check(
        metadata,
        short_blocks,
        [65] * 3,
        rank=0,
        degree=2,
        queries=4,
        block=block,
    )


@pytest.mark.parametrize("block", [False, True])
def test_capture_seeding_keeps_null_rows_empty(make_leaf, block):
    leaf = make_leaf(queries=4, draft=block, block=block, rank=0, degree=2)
    ends = torch.ones(1, dtype=torch.int32, device="cuda")
    leaf.init_forward_metadata_capture_cuda_graph(1, ends, _pages([[0]]))
    metadata = leaf.forward_decode_metadata
    assert not metadata.dcp.local_visible_lens.any()
    assert not metadata.dcp.local_seq_lens.any()


@pytest.mark.parametrize("block", [False, True])
def test_batch_views_share_storage_across_refresh_and_replay(make_leaf, block):
    leaf = make_leaf(queries=4, draft=block, block=block, rank=1, degree=2)
    table = _pages([[2, 5, 8, 1]] * 8)
    ends = torch.full((8,), 129, dtype=torch.int32, device="cuda")
    leaf.refresh_decode_metadata(2, 2, ends[:2], table[:2])
    small = leaf.forward_decode_metadata
    # The consumer graph records addresses, just as an attention call would.
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        captured = small.dcp.local_visible_lens.clone()
    leaf.refresh_decode_metadata(8, 8, ends, table)
    large = leaf.forward_decode_metadata
    assert large.dcp.local_visible_lens.shape == (8, 4)
    assert (
        large.dcp.local_visible_lens.data_ptr()
        == small.dcp.local_visible_lens.data_ptr()
    )
    ends.fill_(257)
    leaf.refresh_decode_metadata(2, 2, ends[:2], table[:2], for_graph_replay=True)
    if block:
        leaf.fill_block_decode_seq_lens(2, ends[:2])
    graph.replay()
    torch.testing.assert_close(captured, small.dcp.local_visible_lens)


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("rebind", [False, True], ids=["reinitialize", "rebind"])
def test_buffer_reset_replaces_cached_views(make_leaf, block, rebind):
    leaf = make_leaf(queries=4, draft=block, block=block, rank=1, degree=2)
    blocks = [[2, 5, 8, 1]] * 2
    table = _pages(blocks)
    ends = torch.full((2,), 129, dtype=torch.int32, device="cuda")
    leaf.refresh_decode_metadata(2, 2, ends, table)
    old = leaf.forward_decode_metadata

    if rebind:
        leaf.set_cache_pool(object())
        leaf.configure_runtime(
            block_granularity=128, virtual_block_count=65, shard_count=2
        )
    leaf.init_cuda_graph_state(8)
    ends.fill_(257)
    leaf.refresh_decode_metadata(2, 2, ends, table)
    metadata = leaf.forward_decode_metadata
    assert metadata is not old
    assert metadata.dcp.virtual_block_count == (65 if rebind else 33)
    # Keep the old view alive so its allocation cannot be recycled. Check the
    # published view, not the identity of the backend's private state owner.
    assert metadata.dcp.page_prefix.data_ptr() != old.dcp.page_prefix.data_ptr()
    _check(
        metadata,
        blocks,
        [257] * 2,
        rank=1,
        degree=2,
        queries=4,
        block=block,
    )


def test_replicated_leaf_has_no_dcp_storage(make_leaf):
    leaf = make_leaf(queries=1, draft=False, block=False, rank=0, degree=1)
    table = _pages([[1, 2, 3, 4]])
    ends = torch.tensor([129], device="cuda", dtype=torch.int32)
    leaf.refresh_decode_metadata(1, 1, ends, table)
    assert leaf.forward_decode_metadata.dcp is None
    assert leaf._dcp is None
    leaf.advance_draft_forward_metadata(ends + 1)
    assert leaf.seq_lens_buf[0].item() == 130
