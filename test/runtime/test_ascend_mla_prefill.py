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

from tokenspeed.runtime.layers.attention.backends.paged.ascend_mla import (
    AscendMLABackend,
    mla_prefill_spans,
)
from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillPlan
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
        AscendPrefillPlan.build(
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

    def patch(
        self,
        layer_id,
        slots,
        direct,
        *,
        slots_workspace,
        mapped_slots,
        width,
        block_rows,
        virtual_blocks,
    ):
        committed = slots_workspace[: slots.numel()]
        committed.copy_(slots)
        slots = committed
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
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillOptions

    backend._prefill_workspace, buffers = AscendPrefillOptions(
        total, -(-total // page)
    ).allocate_workspace(local_rows=7 * page, degree=1, index_width=0, device="cpu")
    backend._prefill_slots = buffers[3]
    backend._placement = CachePlacement(page, 7, (0,), 0)
    backend._prefill_plan = AscendPrefillPlan.build(
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

    for n, k, width in (
        (2112, 4096, 96),
        (1536, 1536, 128),
        (1344, 2048, 96),
        (768, 768, 128),
    ):
        generator = torch.Generator().manual_seed(92001 + width)
        weight = torch.randn((n, k), generator=generator).bfloat16()
        original = weight.clone()
        carrier = pack_mla_panels(weight, width=width)
        raw = carrier.reshape(16, -1, 16).permute(1, 0, 2).contiguous()
        panel = 256 if k == 768 else 512
        restored = raw.reshape(n // width, k // panel, panel // 16, width // 16, 16, 16)
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
        ((96, 784), 96, torch.bfloat16),
        ((96, 512), 96, torch.float32),
    ):
        with pytest.raises(ValueError):
            pack_mla_panels(torch.empty(shape, dtype=dtype), width=width)


def _prefill_cache_and_backend(monkeypatch, *, options):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillCache
    from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheFieldSpec,
        pack,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "numerics", "fast")
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 128)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    group = CacheGroupSpec(
        "full_attention",
        "full_history",
        rows_per_page=128,
        entry_stride_tokens=1,
        replayable=False,
    )
    field = CacheFieldSpec("layer.0.latent_kv", "latent", (128, 1, 576), "bfloat16")
    layout = pack(
        ((group, (field,)),),
        prefix_granularity=128,
        cache_blocks_per_lcm_block={"full_attention": 1},
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(3), "npu", cache_group_specs=(group,))
    kwargs = {
        "arena": arena,
        "dtype": torch.bfloat16,
        "model_dtype": torch.bfloat16,
        "quant_method": "none",
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "layer_num": 1,
        "rank": 0,
    }
    if options is None:
        pool = MLATokenToKVPool(**kwargs)
    else:

        class NativePool(AscendPrefillCache, MLATokenToKVPool):
            pass

        pool = NativePool(prefill_options=options, **kwargs)
    spec = MLAConfig(
        backend_name=None,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=576,
        attn_tp_size=1,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
    )
    config = AttnConfig(
        device="npu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=512,
        max_bs=4,
        components=(spec,),
    )
    return pool, AscendMLABackend(config, spec, kernel_page_size=128)


@pytest.mark.parametrize("engine", ["null", "prefill", "decode"])
def test_ordinary_mla_cache_keeps_prefix_history_without_native_library(
    monkeypatch, engine
):
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", engine)
    monkeypatch.setitem(global_server_args_dict, "max_prefill_tokens", None)
    monkeypatch.delenv("TOKENSPEED_DSA_CP_LIBRARY", raising=False)
    pool, backend = _prefill_cache_and_backend(monkeypatch, options=None)
    backend.set_cache_pool(pool)
    assert backend.prepare_prefill_workspace() == 0
    assert backend._prefill_workspace is None
    assert backend._prefill_slots is None
    rows = pool.get_key_buffer(0).view(-1, 576)
    rows[128:255].fill_(7)
    backend._init_prefill_metadata(
        seq_lens=torch.tensor([130], device="npu", dtype=torch.int32),
        extend_prefix_lens=torch.tensor([127], device="npu", dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor([127], dtype=torch.int32),
        extend_seq_lens=torch.tensor([3], device="npu", dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
        page_table=torch.tensor([[1, 2, 3, 0]], device="npu", dtype=torch.int32),
    )
    meta = backend.chunked_prefill_metadata
    assert meta.chunked_loop_num > 0
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

    layer = PagedAttention(2, 576, 192**-0.5, 1, 0, rotary_emb=None, qk_norm=None)
    chunks = list(
        backend.read_prefix_history(
            pool, layer, meta.chunk_kv_indices_list, dst_dtype=torch.bfloat16
        )
    )
    assert sum(k.shape[0] for k, _ in chunks) == 127
    for latent, rotary in chunks:
        torch.testing.assert_close(latent.cpu(), torch.full_like(latent.cpu(), 7))
        torch.testing.assert_close(rotary.cpu(), torch.full_like(rotary.cpu(), 7))


def test_native_cache_binding_needs_no_transport_library_or_workspace(monkeypatch):
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillOptions

    monkeypatch.delenv("TOKENSPEED_DSA_CP_LIBRARY", raising=False)
    pool, backend = _prefill_cache_and_backend(
        monkeypatch, options=AscendPrefillOptions(129, 4)
    )
    backend.set_cache_pool(pool)
    assert backend.native_prefill_enabled
    assert backend._prefill_runtime is None
    assert backend._prefill_workspace is None
    assert backend._prefill_slots is None


def test_native_mla_backend_owns_commit_slots_until_transport_closes(
    monkeypatch, tmp_path
):
    import os

    import torch.distributed as dist

    import tokenspeed.runtime.distributed.process_group_manager as groups
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillOptions

    options = AscendPrefillOptions(129, 4, 128)
    pool, backend = _prefill_cache_and_backend(monkeypatch, options=options)
    if not os.environ.get("TOKENSPEED_DSA_CP_LIBRARY"):
        pytest.skip("requires the native Prefill transport library")
    manager = groups.ProcessGroupManager()
    monkeypatch.setattr(groups, "process_group_manager", manager)
    manager.init_emulated_rank_zero(
        f"file://{tmp_path}/bootstrap", "gloo", timeout=60, device_id=None
    )
    manager.init_process_group((0,), "gloo")
    try:
        backend.set_cache_pool(pool)
        assert backend._prefill_runtime is None
        assert backend._prefill_workspace is None
        backend.configure_runtime(
            block_granularity=128,
            virtual_block_count=pool.arena.runtime_contract.virtual_block_counts[
                "full_attention"
            ],
            shard_count=1,
        )
        backend.init_forward_metadata(
            bs=1,
            num_extends=1,
            forward_mode=ForwardMode.EXTEND,
            seq_lens=torch.tensor([1], device="npu", dtype=torch.int32),
            extend_prefix_lens=torch.tensor([0], device="npu", dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([1], device="npu", dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
            page_table=torch.tensor([[1, 0, 0, 0]], device="npu", dtype=torch.int32),
            page_table_cpu=torch.tensor([[1, 0, 0, 0]], dtype=torch.int32),
            extend_with_prefix=False,
            query_shard=None,
        )
        assert backend._prefill_runtime is None
        assert backend._prefill_workspace is None
        expected = options.required_workspace(
            local_rows=pool.get_key_buffer(0).numel() // 576, degree=1, index_width=0
        )
        backend.prefetch_prefill_layer(0)
        owner = backend._prefill_workspace
        slots = backend._prefill_slots
        assert owner.frozen and owner.capacity_bytes == expected
        assert slots.numel() >= options.max_current_rows
        for value in (1, 2):
            if value > 1:
                backend.prefetch_prefill_layer(0)
            current = torch.full((1, 576), value, dtype=torch.bfloat16, device="npu")
            backend.start_prefill_kv(0, current)
            backend.consume_prefill_kv(
                0, torch.tensor([128], dtype=torch.int32, device="npu")
            )
            backend.finish_prefill_layer(0)
            torch.testing.assert_close(
                pool.get_key_buffer(0).view(-1, 576)[128].cpu(),
                current[0].cpu(),
                rtol=0,
                atol=0,
            )
            assert backend._prefill_workspace is owner
            assert backend._prefill_slots is slots
    finally:
        backend.close_prefill()
        dist.destroy_process_group()
    assert backend._prefill_workspace is None
    assert backend._prefill_slots is None


@pytest.mark.parametrize("degree", [1, 8, 16])
@pytest.mark.parametrize("index_width", [0, 128])
def test_native_workspace_size_includes_history_current_and_send(degree, index_width):
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillOptions

    # The backend's pool includes aligned int64 commit slots beside transport.
    size = (4 * 128 + (degree + 1) * 128) * (576 + index_width) * 2 + 256
    exact = AscendPrefillOptions(1, 1)
    assert (
        exact.required_workspace(local_rows=128, degree=degree, index_width=index_width)
        == size
    )


@pytest.mark.parametrize("pages,rows", [(0, 1), (-1, 1), (1, 129)])
def test_native_workspace_rejects_unbounded_or_oversized_current_rows(pages, rows):
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillOptions

    with pytest.raises(ValueError, match="page limit"):
        AscendPrefillOptions(rows, pages, 128)


@pytest.mark.parametrize("degree", [1, 8])
@pytest.mark.parametrize("page_size", [64, 128])
def test_bounded_prefill_counts_all_requests_and_compacts_owner_pages(
    degree, page_size
):
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillOptions

    table = torch.tensor([[2, 3, 4, 5], [2, 3, 6, 7]], dtype=torch.int32)
    metadata = RequestMetadata(torch.tensor([3, 17]), torch.tensor([127, 63]), table)
    # Each request rounds to a 128-token scheduler page separately: 2 + 1.
    options = AscendPrefillOptions(20, max_kv_page_num=3, kv_page_size=128)
    placement = CachePlacement(128, 33, tuple(range(degree)), 0)
    plans = [
        AscendPrefillPlan.build(
            metadata,
            placement,
            degree=degree,
            rank=rank,
            page_size=page_size,
            topk=1,
            local_cache_pages=64,
            query_partition=(0, 20),
            options=options,
        )
        for rank in range(degree)
    ]
    for plan in plans:
        assert plan.gather is not None
        assert sum(plan.gather.row_counts) <= 3 * 128
        assert plan.gather.table[0, 0] == plan.gather.table[1, 0]
        expected = []
        for req, (prefix, count) in enumerate([(127, 3), (63, 17)]):
            expected.extend(
                int(plan.gather.table[req, pos // page_size]) * page_size
                + pos % page_size
                for pos in range(prefix, prefix + count)
            )
        assert plan.patch_slots.tolist() == expected
    with pytest.raises(ValueError, match="needs 3 KV pages, limit is 2"):
        AscendPrefillPlan.build(
            metadata,
            placement,
            degree=degree,
            rank=0,
            page_size=page_size,
            topk=1,
            local_cache_pages=64,
            options=AscendPrefillOptions(20, 2, 128),
        )


def test_bounded_workspace_size_does_not_follow_cache_capacity():
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillOptions

    options = AscendPrefillOptions(16384, 1024, 128)
    small = options.required_workspace(local_rows=128, degree=8, index_width=128)
    large = options.required_workspace(local_rows=1 << 20, degree=8, index_width=128)
    assert small == large == 2 * 704 * (2 * 1025 * 128 + 9 * 2048) + 16384 * 8


@pytest.mark.parametrize("degree", [1, 2, 8])
@pytest.mark.parametrize("index_width", [0, 128])
@pytest.mark.parametrize("max_pages", [2, 32])
def test_prefill_workspace_owns_buffers_apart_from_main_stream_scratch(
    degree, index_width, max_pages
):
    from tokenspeed.runtime.execution.workspace import WorkspacePool
    from tokenspeed.runtime.layers.attention.dcp.ascend import AscendPrefillOptions

    options = AscendPrefillOptions(129, max_kv_page_num=max_pages, kv_page_size=128)
    pool, buffers = options.allocate_workspace(
        local_rows=2048, degree=degree, index_width=index_width, device="cpu"
    )
    required = options.required_workspace(
        local_rows=2048, degree=degree, index_width=index_width
    )
    assert pool.frozen
    assert (
        pool.capacity_bytes
        == sum(x.numel() * x.element_size() for x in buffers)
        == required
    )
    addresses = [x.data_ptr() for x in buffers if x.numel()]
    assert len(set(addresses)) == len(addresses)
    for index, value in enumerate(buffers):
        value.fill_(index + 1)
    scratch = WorkspacePool("cpu", initial_nbytes=0)
    scratch.allocate(((129,), torch.int64))[0].fill_(-1)
    for index, value in enumerate(buffers):
        assert torch.all(value == index + 1)
    # The owner outlives the local views even if the allocator reuses scratch.
    del buffers
    with pytest.raises(RuntimeError, match="frozen"):
        pool.allocate(((required + 256,), torch.uint8))
