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


"""NPU DCP writes and Graph replay against physically sharded storage."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
    AscendDSABackend,
    _AscendDSAContextParallel,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend"
)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_single_rank_verify_has_per_candidate_causal_metadata(batch, width):
    from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels

    class CheckedKernels(AscendDSAKernels):
        def index(self, query, key, weights, q_ends, lengths, table, *policy):
            # Fail before a native launch if request rows reach per-query ops.
            assert q_ends.numel() == lengths.numel() == table.shape[0] == query.shape[0]
            return super().index(query, key, weights, q_ends, lengths, table, *policy)

    torch.npu.set_device(0)
    page, heads, prefix_rows = 128, 32, 1
    rows = batch * width
    host_pages = torch.tensor([[4, 2, 7], [6, 3, 5]][:batch], dtype=torch.int32)
    # Include an extend row so the decode slice cannot accidentally use row zero.
    table = torch.cat((torch.ones(1, 3, dtype=torch.int32), host_pages)).npu()
    endpoints = torch.tensor([129, 257][:batch], dtype=torch.int32)
    lengths = torch.cat((torch.tensor([20], dtype=torch.int32), endpoints)).npu()
    host_kv = torch.zeros(8, page, 1, 576, dtype=torch.bfloat16)
    for request, pages in enumerate(host_pages.tolist()):
        values = torch.arange(3 * page, dtype=torch.float32) + request * 400
        host_kv[pages, :, 0, :512] = values.bfloat16().view(3, page, 1)
    kv_cache = host_kv.npu()
    index_cache = torch.full((8, page, 1, 128), -99, dtype=torch.bfloat16, device="npu")
    query = torch.zeros(rows, heads, 576, dtype=torch.bfloat16, device="npu")
    index_query = torch.zeros(rows, 16, 128, dtype=torch.bfloat16, device="npu")
    weights = torch.zeros(rows, 16, dtype=torch.bfloat16, device="npu")
    index_keys = torch.arange(1, rows + 1, dtype=torch.bfloat16).view(rows, 1, 1)
    index_keys = index_keys.expand(rows, 1, 128).contiguous().npu()

    def physical_slots(ends, pages):
        positions = torch.stack([torch.arange(int(n) - width, int(n)) for n in ends])
        return (
            pages.gather(1, positions // page).long() * page + positions % page
        ).flatten()

    locations = physical_slots(endpoints, host_pages).npu()

    # The model, dense leaf and scheduler are absent; execute the real Ascend
    # forward, metadata kernel, Indexer and SFA against these owned cache buffers.
    backend = object.__new__(AscendDSABackend)
    backend.kernel_page_size = page
    backend.spec_num_tokens = width
    backend.step_counter = None
    backend.kvp_group = (0,)
    backend._ascend_placement = None
    backend._fused_group = None
    backend._indexer_spec = SimpleNamespace(
        index_head_dim=128,
        index_topk=2048,
        kv_cache_dim=576,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    backend._indexer_kernels = CheckedKernels()
    backend._dcp = _AscendDSAContextParallel(
        degree=1, rank=0, ranks=(0,), virtual_block_count=None
    )
    backend._dcp.allocate_decode_buffers(rows, 3, "npu")
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=prefix_rows, seq_lens=lengths, page_table=table
        )
    )
    pool = SimpleNamespace(
        get_component=lambda *args: index_cache,
        get_key_buffer=lambda *args: kv_cache,
    )
    layer = SimpleNamespace(layer_id=0, scaling=576**-0.5, logit_cap=0)

    def run():
        backend._clear_forward_scratch()
        return backend.forward_decode(
            query,
            None,
            None,
            layer,
            locations,
            pool,
            batch,
            index_key=index_keys,
            index_query=index_query,
            index_weights=weights,
            index_init_tokens=0,
            index_local_tokens=0,
            index_cache_written=False,
        )

    def check(actual, ends, pages):
        causal = torch.stack(
            [torch.arange(int(n) - width + 1, int(n) + 1) for n in ends]
        )
        assert backend._dcp.seq_lens.cpu().tolist() == causal.flatten().tolist()
        assert backend._dcp.query_ends.cpu().tolist() == list(range(1, rows + 1))
        slots = physical_slots(ends, pages)
        assert torch.equal(backend._dcp.write_slots.cpu(), slots)
        written = index_cache.view(-1, 128)[slots.npu()].cpu()
        torch.testing.assert_close(
            written, index_keys.cpu().view(rows, 128), rtol=0, atol=0
        )
        assert (index_cache[0].cpu() == -99).all()
        actual = actual.cpu().view(rows, heads, 512)
        for row, end in enumerate(causal.flatten().tolist()):
            request = row // width
            history = host_kv[pages[request].long()].reshape(-1, 576)[:end, :512]
            expected = history.double().mean(0).float().expand(heads, -1)
            torch.testing.assert_close(
                actual[row].float(), expected, rtol=4e-3, atol=1e-3
            )

    for _ in range(2):
        eager = run()
    torch.npu.synchronize()
    check(eager, endpoints, host_pages)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = run()
    pointers = (backend._dcp.page_table.data_ptr(), backend._dcp.seq_lens.data_ptr())
    for changed in (False, True, False):
        ends = endpoints + 5 if changed else endpoints
        pages = host_pages.flip(0) if changed else host_pages
        lengths[prefix_rows:].copy_(ends)
        table[prefix_rows:].copy_(pages)
        locations.copy_(physical_slots(ends, pages))
        graph.replay()
        torch.npu.synchronize()
        check(replayed, ends, pages)
        assert (
            backend._dcp.page_table.data_ptr(),
            backend._dcp.seq_lens.data_ptr(),
        ) == pointers
    graph.reset()


@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("block_size", [128, 256])
def test_dcp_local_storage_replay(rank, block_size):
    from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels

    torch.npu.set_device(0)
    degree, virtual_count, page = 8, 17, 128
    subpages = block_size // page
    slots_cpu = torch.arange(virtual_count * block_size, dtype=torch.int64)
    slots = slots_cpu.to("npu")
    keys = (
        (slots_cpu % 31)
        .to(torch.bfloat16)
        .view(-1, 1, 1)
        .expand(-1, 1, 128)
        .contiguous()
        .to("npu")
    )
    cache = torch.full(
        (3 * block_size, 1, 128), -99, dtype=torch.bfloat16, device="npu"
    )
    placement = CachePlacement(block_size, virtual_count, tuple(range(degree)), rank)
    table = torch.zeros((8, 4 * subpages), dtype=torch.int32, device="npu")
    lengths = torch.zeros(8, dtype=torch.int32, device="npu")
    dcp = _AscendDSAContextParallel(
        degree=degree,
        rank=rank,
        ranks=tuple(range(degree)),
        virtual_block_count=virtual_count,
    )
    dcp.allocate_decode_buffers(8, table.shape[1], "npu")
    # Scatter itself has no dependency on the optional indexer extension.
    kernels = object.__new__(AscendDSAKernels)

    def run():
        dcp.clear_forward_metadata()
        local, owned = resolve_cache_slots(slots, placement)
        local = torch.where(owned, local, -1)
        kernels.scatter(keys, cache, local)
        for _ in range(6):
            dcp.prepare_metadata(
                seq_lens=lengths[:],
                page_table=table[:],
                page_size=page,
                block_granularity=block_size,
                init_tokens=131,
                local_tokens=129,
                query_width=1,
                query_group_size=1,
                slots=None,
            )
        return local

    def update(blocks, end):
        pages = [b * subpages + s for b in blocks for s in range(subpages)]
        table.copy_(torch.tensor([pages] * 8, dtype=torch.int32))
        lengths.copy_(torch.arange(end - 7, end + 1, dtype=torch.int32))
        return pages

    update([16, 1, 10, 7], 4 * block_size)
    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        mapped = run()
    pointers = (dcp.page_table.data_ptr(), dcp.seq_lens.data_ptr())
    for blocks, end in [
        ([16, 1, 10, 7], 4 * block_size),
        ([2, 9, 8, 15], 2 * block_size + 3),
        ([9, 16, 2, 1], block_size + 2),
    ]:
        pages = update(blocks, end)
        graph.replay()
        torch.npu.synchronize()
        assert pointers == (dcp.page_table.data_ptr(), dcp.seq_lens.data_ptr())
        mapped_cpu = mapped.cpu()
        owner = (slots_cpu // block_size > 0) & (
            ((slots_cpu // block_size - 1) % degree) == rank
        )
        assert (mapped_cpu[~owner] == -1).all()
        expected = torch.full(cache.shape, -99, dtype=torch.bfloat16)
        expected[mapped_cpu[owner]] = keys.cpu()[owner]
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
        meta = dcp.metadata(start=0, end=8)
        local_table = meta.page_table.cpu()
        local_lens = meta.seq_lens.cpu().tolist()
        for row, n in enumerate(lengths.cpu().tolist()):
            positions = [
                p
                for p in range(n)
                if (pages[p // page] // subpages - 1) % degree == rank
            ]
            assert local_lens[row] == len(positions)
            assert int(meta.init_counts[row]) == sum(p < 131 for p in positions)
            assert int(meta.local_counts[row]) == sum(
                p >= max(n - 129, 0) for p in positions
            )
            offsets = torch.arange(len(positions))
            physical = local_table[row, offsets // page].long() * page + offsets % page
            virtual = torch.tensor(
                [pages[p // page] * page + p % page for p in positions],
                dtype=torch.long,
            )
            torch.testing.assert_close(
                expected[physical], keys.cpu()[virtual], rtol=0, atol=0
            )


@pytest.mark.parametrize("endpoint", [64, 6144])
def test_dcp_metadata_feeds_indexer_and_sparse_attention(endpoint):
    """Feed actual Indexer/SFA from eight local metadata shards on one NPU.

    CPU reference selection and merging supply collective contents here. The
    production path uses fused distributed TopK/LSE; no legacy CP kernels are
    required by this metadata regression.
    """
    from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels

    torch.npu.set_device(0)
    torch.manual_seed(981016)
    kernels = AscendDSAKernels()
    degree, block_size, page, width, topk = 8, 256, 128, 8, 2048
    blocks = list(range(24, 0, -1))
    pages = [b * 2 + s for b in blocks for s in range(2)]
    positions = torch.arange(6144)
    slots_cpu = torch.tensor(pages)[positions // page] * page + positions % page
    slots = slots_cpu.to("npu")
    # The indexer exports BF16 candidate scores. Use distinct BF16 values
    # across binades, avoiding score ties introduced by integer rounding.
    keys_cpu = torch.zeros(6144, 1, 128, dtype=torch.bfloat16)
    keys_cpu[:, 0, 0] = torch.ldexp(
        1 + (positions % 128).float() / 128, positions // 128 - 24
    )
    keys = keys_cpu.to("npu")
    kv_cpu = (torch.randn(6144, 1, 576) * 0.2).bfloat16()
    kv = kv_cpu.to("npu")
    index_query = torch.zeros(width, 16, 128, dtype=torch.bfloat16, device="npu")
    index_query[:, 0, 0] = 1
    weights = torch.zeros(width, 16, dtype=torch.bfloat16, device="npu")
    weights[:, 0] = 1
    query_cpu = (torch.randn(width, 64, 576) * 0.2).bfloat16()
    query = query_cpu.to("npu")
    packed = query.reshape(width, degree, 8, 576).permute(1, 0, 2, 3).contiguous()
    lengths = torch.arange(
        endpoint - width + 1, endpoint + 1, dtype=torch.int32, device="npu"
    )
    table = torch.tensor([pages] * width, dtype=torch.int32, device="npu")
    q_ends = torch.arange(1, width + 1, dtype=torch.int32, device="npu")
    indices, values, metadata, caches = [], [], [], []
    for rank in range(degree):
        placement = CachePlacement(block_size, 25, tuple(range(degree)), rank)
        local, owned = resolve_cache_slots(slots, placement)
        local = torch.where(owned, local, -1)
        index_cache = torch.full(
            (8, page, 1, 128), -99, dtype=torch.bfloat16, device="npu"
        )
        kv_cache = torch.full(
            (8, page, 1, 576), -99, dtype=torch.bfloat16, device="npu"
        )
        kernels.scatter(keys, index_cache, local)
        kernels.scatter(kv, kv_cache, local)
        dcp = _AscendDSAContextParallel(
            degree=degree,
            rank=rank,
            ranks=tuple(range(degree)),
            virtual_block_count=25,
        )
        dcp.refresh_metadata(
            seq_lens=lengths,
            page_table=table,
            page_size=page,
            block_granularity=block_size,
            init_tokens=0,
            local_tokens=0,
            query_width=1,
            query_group_size=1,
            slots=None,
        )
        meta = dcp.metadata(start=0, end=width)
        ix, score = kernels.index_partial(
            index_query,
            index_cache,
            weights,
            q_ends,
            meta.seq_lens,
            meta.page_table,
            topk,
            meta.init_counts,
            meta.local_counts,
            sparse_mode=3,
        )
        indices.append(ix)
        values.append(score)
        metadata.append(meta)
        caches.append(kv_cache)
    global_positions = (
        torch.cat(values, dim=-1).squeeze(1).topk(topk, dim=-1).indices.int()
    )
    outputs, lses = [], []
    selected_positions = [[] for _ in range(width)]
    scale = 576**-0.5
    for rank in range(degree):
        meta = metadata[rank]
        # Reference the fused TopK result's rank-major candidate offsets. This
        # is test input construction, not another production execution path.
        positions_cpu = global_positions.cpu().long() - rank * topk
        owned = (positions_cpu >= 0) & (positions_cpu < topk)
        chosen = (
            indices[rank].cpu().squeeze(1).gather(1, positions_cpu.clamp(0, topk - 1))
        )
        selected_cpu = torch.full((width, 1, topk), -1, dtype=torch.int32)
        chunk_counts = []
        for row in range(width):
            valid = chosen[row][owned[row] & (chosen[row] >= 0)]
            selected_cpu[row, 0, : valid.numel()] = valid
            chunk_counts.append((valid.numel() + 127) // 128)
        selected = selected_cpu.npu()
        chunks = torch.tensor(chunk_counts, dtype=torch.int32, device="npu")
        for row, n in enumerate(lengths.cpu().tolist()):
            logical = [
                p for p in range(n) if (pages[p // page] // 2 - 1) % degree == rank
            ]
            chosen = selected[row, 0].cpu()
            chosen = chosen[chosen >= 0].long()
            assert (chosen < len(logical)).all()
            selected_positions[row].extend(
                torch.tensor(logical, dtype=torch.long)[chosen].tolist()
            )
        out, maximum, total = kernels.attention_partial(
            packed,
            caches[rank],
            selected,
            chunks,
            q_ends,
            meta.seq_lens,
            meta.page_table,
            scale,
            sparse_mode=3,
        )
        outputs.append(out.reshape(64 * width, 512).cpu().float())
        lses.append(
            (maximum + total.log()).squeeze(0).transpose(0, 1).reshape(-1).cpu().float()
        )
    stacked_lse = torch.stack(lses)
    weights = torch.softmax(stacked_lse, dim=0)
    actual = (torch.stack(outputs) * weights[:, :, None]).sum(0)
    actual = actual.view(64, width, 512).transpose(0, 1)
    for row, n in enumerate(lengths.cpu().tolist()):
        selected = list(range(max(0, n - topk), n))
        assert sorted(selected_positions[row]) == selected
        logits = query_cpu[row].float() @ kv_cpu[selected, 0].float().T * scale
        expected = logits.softmax(-1) @ kv_cpu[selected, 0, :512].float()
        torch.testing.assert_close(actual[row], expected, rtol=4e-3, atol=1e-3)
