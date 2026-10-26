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

"""Token-local MLA preserves request boundaries and bounded page reads."""

from dataclasses import dataclass

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
    _AscendDSAPrefillPlan,
)
from tokenspeed.runtime.layers.attention.backends.paged.ascend_mla import (
    AscendMLABackend,
    mla_prefill_spans,
)
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement
from tokenspeed.runtime.layers.attention.mla.ascend import AscendMLAAttention


@pytest.mark.parametrize("degree", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "extends,prefixes",
    [
        ([16384], [0]),
        ([16384], [114688]),
        ([3, 0, 9], [127, 55, 1025]),
        ([0, 1, 0], [0, 13, 900]),
        ([0, 0], [12, 0]),
    ],
)
def test_local_query_spans_preserve_causal_positions(degree, extends, prefixes):
    expected = [
        (req, prefix + pos)
        for req, (extent, prefix) in enumerate(zip(extends, prefixes, strict=True))
        for pos in range(extent)
    ]
    total = sum(extends)
    counts = [total // degree + (rank < total % degree) for rank in range(degree)]
    first = 0
    for rows in counts:
        spans = mla_prefill_spans(extends, prefixes, first=first, rows=rows)
        actual = []
        active_requests = []
        offset = 0
        for req, extent in enumerate(extends):
            if max(first, offset) < min(first + rows, offset + extent):
                active_requests.append(req)
            offset += extent
        for req, (begin, end, past) in zip(active_requests, spans, strict=True):
            actual.extend((req, past + pos) for pos in range(end - begin))
        assert actual == expected[first : first + rows]
        assert sum(end - begin for begin, end, _ in spans) == rows
        first += rows


@pytest.mark.parametrize("first,rows", [(-1, 1), (0, -1), (3, 2)])
def test_invalid_query_partition_fails(first, rows):
    with pytest.raises(ValueError):
        mla_prefill_spans([4], [0], first=first, rows=rows)


@dataclass
class ReceivePlane:
    kv: torch.Tensor


@pytest.mark.parametrize("begin,end", [(0, 1), (5, 129), (127, 257), (0, 384)])
def test_bounded_reads_follow_the_mapped_table_without_changing_storage(begin, end):
    raw = torch.arange(7 * 128 * 576, dtype=torch.int32).reshape(-1, 576)
    table = torch.tensor([[4, 1, 6], [6, 4, 1]])
    backend = object.__new__(AscendMLABackend)
    backend.kernel_page_size = 128
    backend._prefill_pages = table
    backend._prefill_runtime = ReceivePlane(raw)
    before = raw.clone()
    for sequence in range(2):
        expected = torch.cat([raw[p * 128 : (p + 1) * 128] for p in table[sequence]])
        actual = backend.read_prefill_rows(sequence, begin, end)
        torch.testing.assert_close(actual, expected[begin:end], rtol=0, atol=0)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


@dataclass
class RequestMetadata:
    extend_seq_lens_cpu: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    page_table: torch.Tensor

    @property
    def extend_prefix_lens(self):
        return self.extend_prefix_lens_cpu

    @property
    def cum_extend_seq_lens(self):
        return torch.cat(
            (torch.zeros(1, dtype=torch.int64), self.extend_seq_lens_cpu.cumsum(0))
        )


@pytest.mark.parametrize("degree", [1, 2, 8, 16])
@pytest.mark.parametrize("page_size", [64, 128])
def test_head_shards_share_full_query_metadata_and_receive_page_addresses(
    degree, page_size
):
    metadata = RequestMetadata(
        torch.tensor([129, 0, 17]),
        torch.tensor([127, 500, 31]),
        torch.tensor([[2, 3, 4, 5], [0, 0, 0, 0], [6, 0, 0, 0]], dtype=torch.int32),
    )
    placement = CachePlacement(128, 9, tuple(range(degree)), 0)
    plans = [
        _AscendDSAPrefillPlan.build(
            metadata,
            placement,
            degree=degree,
            rank=rank,
            page_size=page_size,
            topk=1,
            local_cache_pages=32,
            query_partition=(0, 146),
        )
        for rank in range(degree)
    ]
    for plan in plans:
        assert (plan.row_start, plan.row_end) == (0, 146)
        assert sum(plan.row_counts) == 146
        torch.testing.assert_close(plan.query_cu, torch.tensor([0, 129, 146]))
        torch.testing.assert_close(plan.key_cu, torch.tensor([0, 256, 304]))
        torch.testing.assert_close(plan.page_table, plans[0].page_table)
        assert plan.query_seq.tolist() == [0] * 129 + [1] * 17


@dataclass
class CPUOwnerCommit:
    """Stand in only for the native NPU byte-copy kernel."""

    current: torch.Tensor
    cache: torch.Tensor

    def patch(self, layer_id, slots, direct, *, width, block_rows, virtual_blocks):
        live = (slots >= block_rows) & (slots < virtual_blocks * block_rows)
        self.cache.index_copy_(0, slots[live].long(), self.current[live])


@pytest.mark.parametrize("prefixes,extends", [([0], [129]), ([127, 255], [3, 2])])
def test_prefill_commits_router_slots_across_request_and_page_boundaries(
    prefixes, extends
):
    from tokenspeed.runtime.layers.attention.backends.paged.write_locations import (
        extend_write_locations,
    )

    page = 128
    table = torch.tensor([[2, 5, 3], [6, 4, 1]], dtype=torch.int32)[: len(extends)]
    total = sum(extends)
    slots = torch.empty((1, total), dtype=torch.int32)
    extend_write_locations(
        table[None],
        torch.tensor([page]),
        torch.tensor(prefixes),
        torch.tensor(extends),
        total,
        slots,
        page_sizes_cpu=(page,),
    )
    current = torch.arange(total * 576, dtype=torch.int32).reshape(total, 576)
    cache = torch.full((7 * page, 576), -1, dtype=torch.int32)
    backend = object.__new__(AscendMLABackend)
    backend._prefill_runtime = CPUOwnerCommit(current, cache)
    backend._placement = CachePlacement(page, 7, (0,), 0)
    backend._prefill_plan = _AscendDSAPrefillPlan.build(
        RequestMetadata(torch.tensor(extends), torch.tensor(prefixes), table),
        backend._placement,
        degree=1,
        rank=0,
        page_size=page,
        topk=1,
        local_cache_pages=7,
        query_partition=(0, total),
    )
    backend.consume_prefill_kv(7, slots[0])
    expected = torch.full_like(cache, -1)
    token = 0
    for req, (prefix, count) in enumerate(zip(prefixes, extends, strict=True)):
        for pos in range(prefix, prefix + count):
            expected[int(table[req, pos // page]) * page + pos % page] = current[token]
            token += 1
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize(
    "prefix,query_rows,block",
    [
        (0, 17, 64),
        (47, 17, 64),
        (64, 17, 64),
        (65, 7, 64),
        (145, 17, 64),
        (513, 17, 64),
        (65537, 17, 65536),
    ],
)
def test_mla_prefill_bounds_history_and_preserves_causal_attention(
    monkeypatch, device, prefix, query_rows, block
):
    """Compare real block projection/merge with full CPU FP64 attention."""
    from tokenspeed.runtime.layers.attention.mla import ascend as mla_host

    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("requires an NPU")
        torch.npu.set_device(0)

    class Projection(torch.nn.Linear):
        def forward(self, rows):
            return super().forward(rows), None

    def reference_attention(q, k, v, *, causal, scale):
        scores = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
        if causal:
            keep = torch.arange(k.shape[0])[None, :] <= (
                torch.arange(q.shape[0])[:, None] + k.shape[0] - q.shape[0]
            )
            scores.masked_fill_(~keep, -torch.inf)
        output = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.double())
        return output, scores.logsumexp(-1).T.float()

    if device == "cpu":
        # Substitute only the unavailable NPU attention primitive with real math.
        def cpu_attention(**kwargs):
            out, lse = reference_attention(
                kwargs["q"],
                kwargs["k"],
                kwargs["v"],
                causal=kwargs["is_causal"],
                scale=kwargs["softmax_scale"],
            )
            return out.to(kwargs["q"].dtype), lse

        monkeypatch.setattr(mla_host, "mla_prefill", cpu_attention)

    torch.manual_seed(20261023)
    layer = AscendMLAAttention.__new__(AscendMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.num_local_heads = 2
    layer.kv_lora_rank = 512
    layer.qk_nope_head_dim = layer.v_head_dim = 128
    layer.attn_mha = torch.nn.Identity()
    layer.attn_mha.scaling = 192**-0.5
    layer.kv_b_proj = Projection(512, 512, bias=False, dtype=torch.bfloat16)
    rows = (torch.randn(prefix + query_rows, 576) * 0.3).bfloat16()
    query = (torch.randn(query_rows, 2, 192) * 0.3).bfloat16()
    expanded = (
        torch.nn.functional.linear(
            rows[:, :512].double(), layer.kv_b_proj.weight.double()
        )
        .bfloat16()
        .double()
        .view(-1, 2, 256)
    )
    key = torch.cat((expanded[..., :128], rows[:, None, 512:].expand(-1, 2, -1)), -1)
    expected, _ = reference_attention(
        query, key, expanded[..., 128:], causal=True, scale=layer.attn_mha.scaling
    )
    layer.to(device)
    rows, query = rows.to(device), query.to(device)
    reads = []

    def read_rows(begin, end):
        reads.append((begin, end))
        return rows[begin:end]

    with torch.inference_mode():
        actual = (
            layer._prefill_sequence(
                query, read_rows, prefix=prefix, kv_block_tokens=block
            )
            .cpu()
            .double()
        )
    assert torch.isfinite(actual).all()
    assert max(end - begin for begin, end in reads) <= block
    assert sum(end - begin for begin, end in reads) == rows.shape[0]
    torch.testing.assert_close(actual, expected, atol=2**-6, rtol=2**-6)


@pytest.mark.parametrize("batch", [24, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mla_graph_updates_lengths_and_releases_scratch(batch, dtype):
    """Real packed FIA: caller buffers, changed-input replay and graph lifetime."""
    import gc
    import weakref

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    pytest.importorskip("flash_ops")
    from tokenspeed_kernel_npu.ops.mla_dcp import LocalLengths

    torch.npu.set_device(0)
    torch.manual_seed(20261023)
    heads, page, scale = 64, 64, 192**-0.5
    q = torch.zeros((batch, 1, heads * 512), dtype=torch.bfloat16, device="npu")
    qr = torch.zeros((batch, 1, heads * 64), dtype=torch.bfloat16, device="npu")
    kv = torch.zeros((4, page, 576), dtype=torch.bfloat16, device="npu")
    table = torch.zeros((batch, 2), dtype=torch.int32, device="npu")
    lengths = LocalLengths(torch.full((batch,), 128, device="npu"))
    outputs = [torch.empty_like(q, dtype=dtype) for _ in range(6)]
    lses = [
        torch.empty((batch, heads, 1), dtype=torch.float32, device="npu")
        for _ in outputs
    ]
    expected, expected_lse = torch.empty_like(outputs[0]), torch.empty_like(lses[0])

    def invoke(out, lse):
        return torch.ops.tokenspeed_npu.dcp_mla.out(
            q,
            qr,
            kv,
            table,
            lengths.key,
            lengths.values,
            num_heads=heads,
            scale=scale,
            out=out,
            lse=lse,
        )

    with torch.inference_mode():
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for out, lse in zip(outputs, lses):
                invoke(out, lse)
        stream.synchronize()
        before = torch.npu.memory_allocated()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            for out, lse in zip(outputs, lses):
                actual, actual_lse = invoke(out, lse)
                assert actual.data_ptr() == out.data_ptr()
                assert actual_lse.data_ptr() == lse.data_ptr()
        allocated = []
        for step in range(12):
            # Nonuniform local endpoints must win over TS's global-length input.
            current_lengths = [
                (0, 1, 63, 64, 65, 127, 128)[(i + step) % 7] for i in range(batch)
            ]
            keys = (torch.randn(4, page, 576) * 0.1).bfloat16()
            pages = torch.randint(0, 4, (batch, 2), dtype=torch.int32)
            kv.copy_(keys)
            table.copy_(pages)
            if step:
                q.copy_(torch.randn(q.shape, dtype=torch.bfloat16) * 0.1)
                qr.copy_(torch.randn(qr.shape, dtype=torch.bfloat16) * 0.1)
            lengths.refresh(torch.tensor(current_lengths, device="npu"))
            invoke(expected, expected_lse)
            graph.update(cpu_update_input=[{"actual_seq_lengths_kv": [1] * batch}])
            graph.replay()
            torch.npu.synchronize()
            for out, lse in zip(outputs, lses):
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
            if step == 0:
                # Zero Q gives independent CPU mean(KV) and log(length) goldens.
                for row, length in enumerate(current_lengths):
                    if length == 0:
                        assert torch.count_nonzero(expected[row]).item() == 0
                        assert torch.isposinf(expected_lse[row]).all().item()
                        continue
                    logical = keys[pages[row].long()].reshape(-1, 576)[:length]
                    golden = logical[:, :512].double().mean(0).expand(heads, -1)
                    tolerance = 2**-6 if dtype == torch.bfloat16 else 2**-16
                    torch.testing.assert_close(
                        expected[row].cpu().double().reshape(heads, 512),
                        golden,
                        atol=tolerance,
                        rtol=2**-6 if dtype == torch.bfloat16 else 2**-10,
                    )
                    torch.testing.assert_close(
                        expected_lse[row].cpu().double(),
                        torch.full(
                            (heads, 1), float(length), dtype=torch.float64
                        ).log(),
                        atol=2**-16,
                        rtol=2**-10,
                    )
            allocated.append(torch.npu.memory_allocated())
        assert max(allocated) - min(allocated) <= 1 << 20
        reference = weakref.ref(graph)
        del graph
        gc.collect()
        torch.npu.synchronize()
        assert reference() is None
        assert torch.npu.memory_allocated() <= before + (1 << 20)


def test_mla_projection_panels_preserve_coefficients():
    import torch
    from tokenspeed_kernel.ops.attention.mla.ascend import pack_mla_panels

    for n, k, width in ((2112, 4096, 96), (1536, 1536, 128)):
        generator = torch.Generator().manual_seed(92001 + width)
        weight = torch.randn((n, k), generator=generator).bfloat16()
        original = weight.clone()
        carrier = pack_mla_panels(weight, width=width)
        raw = carrier.reshape(16, -1, 16).permute(1, 0, 2).contiguous()
        restored = raw.reshape(n // width, k // 512, 32, width // 16, 16, 16)
        restored = restored.permute(0, 3, 4, 1, 2, 5).contiguous().reshape(n, k)
        assert torch.equal(restored, original)
        assert torch.equal(weight, original)


def test_mla_projection_panels_reject_incomplete_tiles():
    import pytest
    import torch
    from tokenspeed_kernel.ops.attention.mla.ascend import pack_mla_panels

    for shape, width, dtype in (
        ((96, 512), 0, torch.bfloat16),
        ((100, 512), 96, torch.bfloat16),
        ((96, 768), 96, torch.bfloat16),
        ((96, 512), 96, torch.float32),
    ):
        with pytest.raises(ValueError):
            pack_mla_panels(torch.empty(shape, dtype=dtype), width=width)
