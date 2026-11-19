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

"""Ascend DSA gives each verify candidate its own causal DCP metadata."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.attention.ascend_qcp_prefill_workspace import (
    AscendQCPPrefillWorkspaceLayout,
)
from tokenspeed_kernel.ops.attention.dsa.paged import (
    PagedDSAAttentionKernels,
    PagedDSAIndexerKernels,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry

from tokenspeed.runtime.layers.attention import ascend_qcp_prefill, backends
from tokenspeed.runtime.layers.attention.backends.paged import (
    ascend_dsa,
)
from tokenspeed.runtime.layers.attention.backends.paged import (
    ascend_dsa_local_decode as ascend_local_decode,
)
from tokenspeed.runtime.layers.attention.backends.paged import (
    ascend_dsa_tp_kvp_decode as ascend_kvp_decode,
)
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.dcp import ascend as ascend_dcp
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.registry import _BACKEND_REGISTRY


def _unavailable(*args, **kwargs):
    raise AssertionError("This device computation was not supplied by the CPU fixture")


def _cpu_indexer(provider):
    return PagedDSAIndexerKernels(
        **{
            target: getattr(provider, source, _unavailable)
            for target, source in (
                ("store_index_keys", "scatter"),
                ("select_indices", "index"),
                ("select_candidates", "index_partial"),
                ("select_local_indices", "select_local"),
                ("merge_topk", "merge_topk_fused"),
            )
        }
    )


def _cpu_attention(provider):
    return PagedDSAAttentionKernels(
        **{
            name: getattr(provider, name, _unavailable)
            for name in (
                "attention",
                "attention_partial",
                "merge_partials",
                "merge_lse_direct",
            )
        }
    )


@pytest.fixture(autouse=True)
def cpu_computations(monkeypatch):
    for module in (ascend_local_decode, ascend_kvp_decode):
        monkeypatch.setattr(
            module,
            "select_paged_dsa_indexer",
            lambda: _cpu_indexer(ascend_dsa.ascend_dsa_kernels()),
        )
        monkeypatch.setattr(
            module,
            "select_paged_dsa_attention",
            lambda: _cpu_attention(ascend_dsa.ascend_dsa_kernels()),
        )
    monkeypatch.setattr(
        ascend_kvp_decode,
        "validate_dsa_context_parallel",
        lambda **kw: ascend_dsa.ascend_dsa_kernels().require_fused_context_parallel(
            **kw
        ),
    )


def _selection_for_test(
    backend, query, layer, slots, pool, lengths, table, inputs, *, width, fused
):
    backend.spec_num_tokens = width
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=0,
            seq_lens=lengths,
            page_table=table,
        )
    )
    layer = SimpleNamespace(layer_id=layer.layer_id, scaling=1.0, logit_cap=0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            backend._decode,
            "_run_indexed_sparse_attention",
            lambda selection, *a, **kw: selection,
        )
        return backend.forward_decode(
            query,
            None,
            None,
            layer,
            slots,
            pool,
            query.shape[0] // width,
            fused=fused,
            **inputs,
        )


@pytest.fixture(autouse=True)
def cpu_metadata_operator(monkeypatch):
    # This suite exercises backend lifecycle without NPU buffers. Actual native
    # integer math and changing-input Graph replay are tested on hardware.
    def reference(table, lengths, **k):
        from tokenspeed.runtime.layers.attention.dcp.metadata import (
            CompactDCPLayout,
            refresh_dcp_page_table_metadata,
        )

        width = k["query_width"]
        if width > 1:
            offsets = torch.arange(1 - width, 1, dtype=lengths.dtype)
            lengths = (lengths[:, None] + offsets).clamp_min(1).reshape(-1)
            table = table.repeat_interleave(width, 0)
        visible = torch.stack(
            (
                lengths.clamp_max(k["initial_tokens"]),
                (lengths - k["window_tokens"]).clamp_min(0),
                lengths,
            ),
            1,
        )
        meta = refresh_dcp_page_table_metadata(
            page_table=table,
            virtual_block_count=k["virtual_block_count"],
            degree=k["degree"],
            rank=k["rank"],
            layout=CompactDCPLayout(
                seq_lens=lengths,
                page_size=k["page_size"],
                block_granularity=k["block_granularity"],
                visible_lens=visible,
            ),
            previous=None,
        )
        k["out"].copy_(meta.local_page_table)
        k["local_lengths"].copy_(meta.local_seq_lens)
        k["initial_counts"].copy_(meta.local_visible_lens[:, 0])
        k["window_counts"].copy_(
            meta.local_visible_lens[:, 2] - meta.local_visible_lens[:, 1]
        )
        k["query_ends"].copy_(torch.arange(1, lengths.numel() + 1, dtype=torch.int32))
        last = slice(k["query_group_size"] - 1, None, k["query_group_size"])
        for output, source in (
            ("index_table", "out"),
            ("index_lengths", "local_lengths"),
            ("index_initial_counts", "initial_counts"),
            ("index_query_ends", "query_ends"),
        ):
            k[output].copy_(k[source][last])
        if k["slots"] is not None:
            placement = CachePlacement(
                k["block_granularity"],
                k["virtual_block_count"],
                tuple(range(k["degree"])),
                k["rank"],
            )
            local, owned = resolve_cache_slots(k["slots"], placement)
            k["local_slots"].copy_(torch.where(owned, local, -1))

    monkeypatch.setattr(ascend_dcp, "ascend_dsa_cp_metadata", reference)


def test_ascend_backend_reuses_the_common_dsa_base():
    assert issubclass(ascend_dsa.AscendDSABackend, DSABackend)


def test_both_dsa_modules_are_imported():
    assert backends.ascend_dsa is ascend_dsa
    assert backends.dsa.DSABackend is DSABackend
    expected = ascend_dsa.AscendDSABackend if current_platform().is_npu else DSABackend
    assert _BACKEND_REGISTRY["dsa"][1] is expected


@pytest.mark.skipif(not current_platform().is_npu, reason="Ascend RMSNorm kernel")
@pytest.mark.parametrize("epsilon", [1e-5, 3e-4])
def test_index_key_normalization_uses_checkpoint_epsilon(epsilon):
    from transformers import PretrainedConfig

    from tokenspeed.runtime.layers.attention.dsa.ascend import AscendDSAIndexer

    config = PretrainedConfig(
        index_n_heads=8,
        index_head_dim=128,
        index_use_rope=False,
        rms_norm_eps=epsilon,
    )
    indexer = AscendDSAIndexer(config, hidden_size=64, q_lora_rank=32)
    norm = indexer.k_norm.to(device="npu", dtype=torch.bfloat16)
    # Small keys make epsilon material, so a hardcoded default cannot pass.
    key = torch.linspace(-0.003, 0.003, 4 * 128).reshape(4, 128).bfloat16()
    expected = torch.nn.functional.rms_norm(
        key.float(), (128,), torch.ones(128), eps=epsilon
    ).bfloat16()
    actual = norm(key.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=2**-7, atol=2**-14)


def test_ascend_mla_prologue_supports_native_prefill_and_decode():
    spec = KernelRegistry.get().get_by_name("ascend_composite_mla_prologue")
    assert spec is not None
    assert spec.capability.vendors == frozenset({"ascend"})
    assert spec.traits["expanded"] == frozenset({False, True})
    assert spec.traits["kv_format"] == frozenset({"native"})
    assert spec.traits["store"] == frozenset({False, True})


@pytest.mark.parametrize("degree,rank", [(1, 0), (2, 0), (4, 0), (8, 3), (16, 0)])
def test_ascend_dsa_consumes_host_kvp_topology(monkeypatch, degree, rank):
    # Mock device initialization, but keep the NPU constructor and its topology
    # checks. The config deliberately has no legacy dcp_* attributes.
    def initialize_base(self, config, spec, *, kernel_page_size):
        self.is_draft = config.is_draft
        self.spec_num_tokens = 8
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size

    monkeypatch.setattr(DSABackend, "__init__", initialize_base)

    monkeypatch.setattr(ascend_dsa, "ascend_dsa_kernels", lambda: SimpleNamespace())
    config = SimpleNamespace(
        max_bs=64,
        speculative_num_draft_tokens=8,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method=None,
        is_draft=False,
        kvp_size=degree,
        kvp_rank=rank,
        kvp_group=tuple(range(degree)),
    )
    spec = SimpleNamespace(
        is_dsa=True,
        index_k_format="bf16",
        attn_tp_size=degree,
        num_attention_heads=32,
    )
    # Prefill construction must not validate unused fused Decode operators.
    ascend_dsa.AscendDSABackend(config, spec, kernel_page_size=128)
    if degree > 1:
        spec.attn_tp_size = 1
        with pytest.raises(ValueError, match="KV sharding"):
            ascend_dsa.AscendDSABackend(config, spec, kernel_page_size=128)
    else:
        spec.attn_tp_size = 4
        prefill = ascend_dsa.AscendDSABackend(config, spec, kernel_page_size=128)
        with pytest.raises(ValueError, match="Local DSA Decode requires TP1/KVP1"):
            prefill.forward_decode(
                torch.empty(1, 32, 576),
                None,
                None,
                SimpleNamespace(logit_cap=0),
                None,
                None,
                1,
                fused=False,
            )


def test_dsa_slots_share_the_metadata_refresh():
    dcp = ascend_dcp.AscendDCPMetadata(
        degree=8,
        rank=1,
        ranks=tuple(range(8)),
        virtual_block_count=17,
    )
    slots = torch.tensor([128, 256, 0, 257], dtype=torch.int64)
    lengths = torch.tensor([260], dtype=torch.int32)
    table = torch.tensor([[1, 2, 3]], dtype=torch.int32)

    def prepare():
        dcp.prepare_metadata(
            seq_lens=lengths,
            page_table=table,
            page_size=128,
            block_granularity=128,
            init_tokens=0,
            local_tokens=0,
            query_width=1,
            query_group_size=1,
            slots=slots,
        )
        return dcp.write_slots

    first = prepare()
    assert first.tolist() == [-1, 128, -1, 129]
    assert prepare().data_ptr() == first.data_ptr()
    dcp.clear_forward_metadata()
    slots[0] = 258
    assert prepare().tolist() == [130, 128, -1, 129]


def test_each_candidate_keeps_its_own_local_page_prefix():
    table = torch.tensor([[1, 2, 3]] * 4, dtype=torch.int32)
    dcp = ascend_dcp.AscendDCPMetadata(
        virtual_block_count=8,
        degree=8,
        rank=1,
        ranks=tuple(range(8)),
    )
    dcp.refresh_metadata(
        page_table=table,
        seq_lens=torch.tensor([254, 255, 256, 257], dtype=torch.int32),
        page_size=128,
        block_granularity=128,
        init_tokens=0,
        local_tokens=0,
        query_width=1,
        query_group_size=1,
        slots=None,
    )
    compact = dcp.metadata(start=0, end=4)
    assert compact.seq_lens.tolist() == [126, 127, 128, 128]
    assert compact.page_table[:, 0].tolist() == [1, 1, 1, 1]


def test_dcp_metadata_is_shared_only_within_matching_forward(monkeypatch):
    dcp = ascend_dcp.AscendDCPMetadata(
        virtual_block_count=17,
        degree=8,
        rank=1,
        ranks=tuple(range(8)),
    )
    table = torch.tensor([[1, 2, 3]] * 4, dtype=torch.int32)
    lengths = torch.tensor([254, 255, 256, 257], dtype=torch.int32)
    calls = []
    refresh = dcp.refresh_metadata

    def counted(**kwargs):
        calls.append(kwargs)
        refresh(**kwargs)

    monkeypatch.setattr(dcp, "refresh_metadata", counted)

    def prepare(local=0, rows=4):
        dcp.prepare_metadata(
            page_table=table[:rows],
            seq_lens=lengths[:rows],
            page_size=128,
            block_granularity=128,
            init_tokens=0,
            local_tokens=local,
            query_width=1,
            query_group_size=1,
            slots=None,
        )

    for _ in range(6):
        prepare()
    assert len(calls) == 1
    assert dcp.seq_lens.tolist() == [126, 127, 128, 128]
    # The next round can update the same device addresses in place.
    backend = SimpleNamespace(_ascend_placement=None, _dcp=dcp)
    backend._dcp.clear_forward_metadata()
    lengths.fill_(130)
    table[:, 1] = 10
    prepare()
    assert len(calls) == 2
    assert dcp.seq_lens.tolist() == [2] * 4
    assert dcp.page_table[:, 0].tolist() == [2] * 4
    prepare(local=1)
    assert len(calls) == 3
    assert dcp.local_counts.tolist() == [1] * 4
    prepare(local=1, rows=2)
    assert len(calls) == 4
    assert dcp.seq_lens.tolist() == [2] * 2
    # Rebinding, even to the same capacity, invalidates the forward result.
    dcp.bind_virtual_block_count(17)
    prepare(local=1, rows=2)
    assert len(calls) == 5


def test_decode_metadata_storage_is_stable_across_batch_sizes():
    dcp = ascend_dcp.AscendDCPMetadata(
        degree=1,
        rank=0,
        ranks=(0,),
        virtual_block_count=8,
    )
    dcp.allocate_decode_buffers(rows=8, columns=3, device="cpu")
    large = dcp._buffer("page_table", (8, 3), torch.device("cpu"))
    small = dcp._buffer("page_table", (2, 3), torch.device("cpu")).zero_()
    assert large.data_ptr() == small.data_ptr()
    assert small.tolist() == [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(RuntimeError, match="preallocated decode geometry"):
        dcp._buffer("page_table", (9, 3), torch.device("cpu"))


@pytest.mark.parametrize("prefill_rows,prefill_columns", [(17, 3), (4, 4)])
def test_large_prefill_does_not_replace_decode_metadata_buffers(
    prefill_rows, prefill_columns
):
    dcp = ascend_dcp.AscendDCPMetadata(
        degree=8,
        rank=0,
        ranks=tuple(range(8)),
        virtual_block_count=17,
    )
    dcp.allocate_decode_buffers(8, 3, "cpu")

    def refresh(rows, columns):
        dcp.refresh_metadata(
            seq_lens=torch.full((rows,), 260, dtype=torch.int32),
            page_table=torch.tensor(
                [[1, 2, 9] + [0] * (columns - 3)] * rows, dtype=torch.int32
            ),
            page_size=128,
            block_granularity=128,
            init_tokens=0,
            local_tokens=0,
            query_width=1,
            query_group_size=1,
            slots=None,
        )
        meta = dcp.metadata(start=0, end=rows)
        assert meta.seq_lens.tolist() == [132] * rows
        assert meta.page_table.tolist() == [[1, 2] + [0] * (columns - 2)] * rows
        return meta

    small = refresh(8, 3)
    pointers = (small.page_table.data_ptr(), small.seq_lens.data_ptr())
    refresh(prefill_rows, prefill_columns)
    restored = refresh(8, 3)
    assert (restored.page_table.data_ptr(), restored.seq_lens.data_ptr()) == pointers


@pytest.mark.parametrize("degree", [2, 8])
@pytest.mark.parametrize("block_size", [128, 256])
def test_sharded_index_write_and_read_use_local_storage(
    monkeypatch, degree, block_size
):
    """Virtual IDs exceed each rank's capacity; replicated addressing fails."""

    class ReachedIndexer(Exception):
        pass

    subpages = block_size // 128
    blocks = [degree * 2, 1, degree + 1, 2]
    # No repeated block within a request, including degree=1.
    blocks = list(dict.fromkeys(blocks))
    pages = [b * subpages + s for b in blocks for s in range(subpages)]
    table = torch.tensor([pages] * 8, dtype=torch.int32)
    lengths = torch.arange(
        len(pages) * 128 - 7, len(pages) * 128 + 1, dtype=torch.int32
    )
    virtual_count = degree * 2 + 1
    slots = torch.arange(virtual_count * block_size, dtype=torch.int64)
    keys = slots.float().view(-1, 1, 1).expand(-1, 1, 8).contiguous()
    for rank in range(degree):
        placement = CachePlacement(
            block_size, virtual_count, tuple(range(degree)), rank
        )
        cache = torch.full((3 * block_size, 1, 8), -99.0)
        dcp = ascend_dcp.AscendDCPMetadata(
            degree=degree,
            rank=rank,
            ranks=tuple(range(degree)),
            virtual_block_count=virtual_count,
        )
        # Stop before the merge kernel: exercise real writes and metadata
        # independently for every owner.

        class Kernels:
            def scatter(self, rows, destination, locations):
                assert int(locations.max()) < cache.shape[0]
                virtual_blocks = slots // block_size
                owners = (virtual_blocks > 0) & ((virtual_blocks - 1) % degree == rank)
                assert (locations[~owners] == -1).all()
                valid = locations >= 128
                destination.view_as(cache)[locations[valid]] = rows[valid]

            def index_partial(self, *args, **kwargs):
                raise ReachedIndexer

            def index(self, *args):
                raise ReachedIndexer

        backend = _fused_backend(monkeypatch, 8, degree=degree, heads=64, rank=rank)
        backend._indexer_spec.index_head_dim = 8
        backend._indexer_spec.index_topk = 512
        backend._decode._decode_indexer = _cpu_indexer(Kernels())
        backend._decode._dcp = dcp
        backend._decode.placement = placement
        monkeypatch.setattr(backend, "cache_placement", lambda layer: placement)
        with pytest.raises(ReachedIndexer):
            _selection_for_test(
                backend,
                torch.zeros(8, 1, 8),
                SimpleNamespace(layer_id=0),
                slots,
                SimpleNamespace(get_component=lambda *args: cache),
                lengths,
                table,
                dict(
                    index_key=keys,
                    index_query=None,
                    index_weights=None,
                    index_init_tokens=131,
                    index_local_tokens=129,
                ),
                width=1,
                fused=False,
            )
        assert (cache[:block_size] == -99).all()
        for local_block in (1, 2):
            v = (local_block - 1) * degree + rank + 1
            torch.testing.assert_close(
                cache[local_block * block_size : (local_block + 1) * block_size],
                keys[v * block_size : (v + 1) * block_size],
            )
        # Check the same ABI also for degree=1 (no collective branch).
        dcp.refresh_metadata(
            seq_lens=lengths,
            page_table=table,
            page_size=128,
            block_granularity=block_size,
            init_tokens=131,
            local_tokens=129,
            query_width=1,
            query_group_size=1,
            slots=None,
        )
        meta = dcp.metadata(start=0, end=8)
        for row, end in enumerate(lengths.tolist()):
            expected = []
            positions = []
            for pos in range(end):
                page = pages[pos // 128]
                block = page // subpages
                if (block - 1) % degree == rank:
                    expected.append(page * 128 + pos % 128)
                    positions.append(pos)
            n = int(meta.seq_lens[row])
            assert n == len(expected)
            assert int(meta.init_counts[row]) == sum(p < 131 for p in positions)
            assert int(meta.local_counts[row]) == sum(
                p >= max(0, end - 129) for p in positions
            )
            indices = torch.arange(n)
            addresses = (
                meta.page_table[row, indices // 128].long() * 128 + indices % 128
            )
            torch.testing.assert_close(
                cache[addresses, 0, 0], torch.tensor(expected, dtype=torch.float32)
            )


class _CpuFusedDsaKernels:
    """CPU stand-ins for native SFA and IPC, using real producer buffers."""

    def require_fused_context_parallel(self, *, rank_size, num_heads):
        """The fixture substitutes the installed device and IPC operators."""

    def __init__(self, rows, degree):
        self.rows = rows
        self.peer_outputs = torch.arange(degree, dtype=torch.float32)[
            :, None, None, None
        ]
        self.peer_maximum = torch.arange(degree, dtype=torch.float32)[:, None, None] / 4
        self.peer_sum = torch.arange(1, degree + 1, dtype=torch.float32)[:, None, None]

    def index_partial(self, *args, **kwargs):
        values = torch.arange(2048, dtype=torch.float32).expand(self.rows, 1, -1)
        indices = torch.arange(2048, dtype=torch.int32).expand(self.rows, 1, -1)
        return indices.contiguous(), values.contiguous()

    def merge_topk_fused(self, indices, values, *, group, rank_size):
        # Only the registered native communicator exists in this environment.
        rank = {"fused-dsa": 0}[group]
        return indices[rank].expand(self.rows, -1).contiguous(), torch.full(
            (self.rows,), 16, dtype=torch.int32
        )

    def attention_partial(self, *args, out, **kwargs):
        output, maximum, total = out
        output.fill_(0)
        maximum.fill_(0)
        total.fill_(1)
        return out

    def merge_lse_direct(self, output, maximum, total, *, group, rank_size):
        rank = {"fused-dsa": 0}[group]
        # Resolve the communicator peers, then include the original producer.
        heads = output.shape[0]
        peers = self.peer_outputs[:rank_size].expand(-1, heads, self.rows, 512).clone()
        maxima = self.peer_maximum[:rank_size].expand(-1, self.rows, heads).clone()
        sums = self.peer_sum[:rank_size].expand_as(maxima).clone()
        peers[rank].copy_(output)
        maxima[rank].copy_(maximum[0])
        sums[rank].copy_(total[0])
        global_max = maxima.max(0).values
        weights = (maxima - global_max).exp() * sums
        weighted = peers * weights.transpose(1, 2)[..., None]
        merged = weighted.sum(0) / weights.sum(0).T[..., None]
        local_heads = output.shape[0] // rank_size
        return merged[rank * local_heads : (rank + 1) * local_heads].bfloat16()


def _fused_backend(
    monkeypatch,
    rows,
    *,
    degree,
    heads,
    rank=0,
    backend_class=ascend_dsa.AscendDSABackend,
):
    def initialize_base(self, config, spec, *, kernel_page_size):
        self.is_draft = False
        self.spec_num_tokens = 8
        self.num_local_heads = heads // degree
        self.step_counter = None
        self.kernel_page_size = kernel_page_size

    kernels = _CpuFusedDsaKernels(rows, degree)
    monkeypatch.setattr(DSABackend, "__init__", initialize_base)
    monkeypatch.setattr(ascend_dsa, "ascend_dsa_kernels", lambda: kernels)
    config = SimpleNamespace(
        max_bs=64,
        speculative_num_draft_tokens=8,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method=None,
        kvp_size=degree,
        kvp_rank=rank,
        kvp_group=tuple(range(degree)),
    )
    spec = SimpleNamespace(
        is_dsa=True,
        index_k_format="bf16",
        attn_tp_size=degree,
        num_attention_heads=heads,
        index_head_dim=128,
        index_topk=2048,
        kv_cache_dim=576,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    backend = backend_class(config, spec, kernel_page_size=128)
    backend._dense_backend = SimpleNamespace(forward_decode_metadata=None)
    placement = CachePlacement(128, 17, tuple(range(degree)), rank)
    if degree > 1:
        backend._decode.bind_cache(placement)
    monkeypatch.setattr(backend, "cache_placement", lambda layer: placement)
    return backend


class _CpuCollectiveDsaKernels(_CpuFusedDsaKernels):
    """Substitute device math while real Gloo exchanges rank-specific tensors."""

    def __init__(self, rows, degree, rank):
        super().__init__(rows, degree)
        self.degree = degree
        self.rank = rank

    def index_partial(self, *args, **kwargs):
        indices = torch.arange(2048, dtype=torch.int32).expand(self.rows, 1, -1)
        scores = indices.float() * self.degree + self.rank
        return indices.contiguous(), scores.contiguous()

    def select_local(self, indices, positions, rank):
        owned = positions.div(indices.shape[-1], rounding_mode="floor") == rank
        selected = indices.squeeze(1).gather(1, positions.long() % indices.shape[-1])
        output = torch.full_like(selected, -1)
        for row in range(self.rows):
            values = selected[row, owned[row]]
            output[row, : values.numel()] = values
        return (
            output[:, None, :],
            (owned.sum(1) + 127).div(128, rounding_mode="floor").int(),
        )

    def attention_partial(self, *args, **kwargs):
        head = torch.arange(64, dtype=torch.float32)[:, None, None] / 8
        row = torch.arange(self.rows, dtype=torch.float32)[None, :, None] / 4
        feature = (torch.arange(512, dtype=torch.float32) % 4)[None, None, :] / 16
        output = self.rank + head + row + feature
        maximum = torch.full((1, self.rows, 64), self.rank / 4)
        total = torch.full_like(maximum, self.rank + 1)
        if self.rank == self.degree - 1:
            output.fill_(float("nan"))
            maximum.fill_(-3e38)
            total.zero_()
        return output, maximum, total

    def merge_partials(self, output, lse):
        weights = torch.softmax(lse, dim=0)
        return (output * weights[..., None]).sum(0).bfloat16()


def _collective_dsa_worker(rank, rendezvous):
    from tokenspeed.runtime.distributed import process_group_manager as pg_module
    from tokenspeed.runtime.distributed.mapping import MappingBase
    from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager

    torch.set_num_threads(1)
    degree = 8
    manager = ProcessGroupManager()
    manager.init_distributed(
        MappingBase(rank=rank, world_size=degree),
        rendezvous,
        backend="gloo",
        timeout=45,
        device_id=None,
    )
    try:
        manager.init_process_group(tuple(range(degree)), backend="gloo")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pg_module, "process_group_manager", manager)
            for rows in (1, 3, 16):
                backend = _fused_backend(
                    patch, rows, degree=degree, heads=64, rank=rank
                )
                cpu = _CpuCollectiveDsaKernels(rows, degree, rank)
                backend._decode._decode_indexer = _cpu_indexer(cpu)
                backend._decode._decode_attention = _cpu_attention(cpu)
                # Small shapes must retain the general path even after the
                # model has configured native buffers for larger shapes.
                backend._decode.configure_fused_decode(
                    group="fused-dsa",
                    ranks=tuple(backend._decode._dcp.ranks),
                    attention_outputs={
                        64: (
                            torch.empty(64, 64, 512),
                            torch.empty(1, 64, 64),
                            torch.empty(1, 64, 64),
                        )
                    },
                )
                query = torch.full((rows, 8, 576), rank, dtype=torch.bfloat16)
                layer = SimpleNamespace(layer_id=0, scaling=1.0)
                index_cache = torch.zeros(8, 128, 1, 128, dtype=torch.bfloat16)
                kv_cache = torch.zeros(8, 128, 1, 576, dtype=torch.bfloat16)
                pool = SimpleNamespace(
                    get_component=lambda *args: index_cache,
                    get_key_buffer=lambda *args: kv_cache,
                )
                selection = _selection_for_test(
                    backend,
                    query,
                    layer,
                    torch.arange(rows),
                    pool,
                    torch.full((rows,), 128, dtype=torch.int32),
                    torch.ones(rows, 1, dtype=torch.int32),
                    dict(
                        index_query=None,
                        index_weights=None,
                        index_init_tokens=0,
                        index_local_tokens=0,
                        index_cache_written=True,
                    ),
                    width=1,
                    fused=False,
                )
                expected_query = torch.arange(degree, dtype=torch.bfloat16)[
                    :, None, None, None
                ].expand(degree, rows, 8, 576)
                torch.testing.assert_close(selection.query, expected_query)
                # Across all eight shards the largest 2048 scores contain
                # exactly the last 256 local indices from every rank.
                expected_indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32)
                expected_indices[:, :, :256] = torch.arange(2047, 1791, -1)
                torch.testing.assert_close(selection.indices, expected_indices)
                assert selection.valid_chunks.tolist() == [2] * rows
                ranks = torch.arange(degree - 1, dtype=torch.float64)
                offset = (ranks * torch.softmax(ranks / 4 + (ranks + 1).log(), 0)).sum()
                head = torch.arange(rank * 8, (rank + 1) * 8)[:, None, None] / 8
                row = torch.arange(rows)[None, :, None] / 4
                feature = (torch.arange(512) % 4)[None, None, :] / 16
                expected = (offset + head.double() + row + feature).bfloat16()
                for head_major in (False, True):
                    actual = backend._decode._run_indexed_sparse_attention(
                        selection,
                        layer,
                        pool,
                        head_major_output=head_major,
                    )
                    reference = (
                        expected
                        if head_major
                        else expected.transpose(0, 1).contiguous().flatten(1)
                    )
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_collective_dsa_small_rows_merge_real_eight_rank_exchanges(tmp_path):
    """Exercise padding, rank/head order and empty-shard LSE over real Gloo."""
    if torch.distributed.is_initialized() or torch.cuda.is_initialized():
        pytest.skip("requires an isolated CPU process group")
    if not torch.distributed.is_gloo_available():
        pytest.skip("requires the Gloo CPU backend")
    torch.multiprocessing.start_processes(
        _collective_dsa_worker,
        args=(f"file://{tmp_path / 'dsa-rendezvous'}",),
        nprocs=8,
        join=True,
        start_method="fork",
    )


def test_dsa_metadata_needs_no_collective_process_group(monkeypatch):
    # The global runtime may exist while this backend has no HCCL groups.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    backend = _fused_backend(monkeypatch, 64, degree=8, heads=64)
    backend._decode._dcp.refresh_metadata(
        page_table=torch.tensor([[1, 2]], dtype=torch.int32),
        seq_lens=torch.tensor([256], dtype=torch.int32),
        page_size=128,
        block_granularity=128,
        init_tokens=0,
        local_tokens=0,
        query_width=1,
        query_group_size=1,
        slots=None,
    )
    metadata = backend._decode._dcp.metadata(start=0, end=1)
    assert metadata.seq_lens.tolist() == [128]
    assert metadata.page_table.tolist() == [[1, 0]]


@pytest.mark.parametrize("missing", ["npu_dcp_merge_topk", "npu_dsa_merge_lse_direct"])
def test_fused_dsa_binding_requires_context_parallel_operators(monkeypatch, missing):
    backend = _fused_backend(monkeypatch, 1, degree=8, heads=32)

    def reject(*, rank_size, num_heads):
        assert (rank_size, num_heads) == (8, 32)
        raise RuntimeError(f"Missing operator {missing}")

    monkeypatch.setattr(
        ascend_dsa.ascend_dsa_kernels(), "require_fused_context_parallel", reject
    )
    with pytest.raises(RuntimeError, match=missing):
        backend._decode.configure_fused_decode(
            group="test",
            ranks=tuple(range(8)),
            attention_outputs={
                1: (
                    torch.empty(32, 1, 512),
                    torch.empty(1, 1, 32),
                    torch.empty(1, 1, 32),
                )
            },
        )
    assert backend._decode._fused_decode is None


@pytest.mark.parametrize(
    "heads,degree,rows",
    [
        (64, 2, 64),
        (64, 4, 64),
        (64, 8, 64),
        (64, 16, 64),
        (64, 8, 128),
        (64, 8, 256),
        (64, 8, 512),
        (32, 8, 1),
        (32, 8, 32),
        (32, 8, 512),
    ],
)
@pytest.mark.parametrize("head_major", [False, True])
def test_fused_dsa_uses_registered_outputs_and_merges_lse(
    monkeypatch, heads, degree, rows, head_major
):
    backend = _fused_backend(monkeypatch, rows, degree=degree, heads=heads)
    outputs = (
        torch.empty(heads, rows, 512),
        torch.empty(1, rows, heads),
        torch.empty(1, rows, heads),
    )
    backend._decode.configure_fused_decode(
        group="fused-dsa",
        ranks=tuple(backend._decode._dcp.ranks),
        attention_outputs={rows: outputs},
    )
    local_heads = heads // degree
    packed_query = torch.zeros(degree, rows, local_heads, 576, dtype=torch.bfloat16)
    layer = SimpleNamespace(layer_id=0, scaling=1.0)
    index_cache = torch.zeros(2, 128, 1, 128, dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 128, 1, 576, dtype=torch.bfloat16)
    pool = SimpleNamespace(
        get_component=lambda *args: index_cache,
        get_key_buffer=lambda *args: kv_cache,
    )
    selection = _selection_for_test(
        backend,
        packed_query[0],
        layer,
        torch.arange(rows),
        pool,
        torch.full((rows,), 128, dtype=torch.int32),
        torch.ones(rows, 1, dtype=torch.int32),
        dict(
            index_query=None,
            index_weights=None,
            index_init_tokens=0,
            index_local_tokens=0,
            index_cache_written=True,
            packed_query=packed_query,
        ),
        width=1,
        fused=True,
    )
    assert selection.query is packed_query
    assert selection.indices.shape == (rows, 2048)
    actual = backend._decode._run_indexed_sparse_attention(
        selection,
        layer,
        pool,
        head_major_output=head_major,
    )
    # Independent log-softmax reference over the configured context shards.
    ranks = torch.arange(degree, dtype=torch.float64)
    probabilities = torch.softmax(ranks / 4 + (ranks + 1).log(), dim=0)
    expected = (ranks * probabilities).sum().bfloat16()
    shape = (local_heads, rows, 512) if head_major else (rows, local_heads * 512)
    torch.testing.assert_close(
        actual, torch.full(shape, expected, dtype=torch.bfloat16)
    )


def test_dsa_rejects_resources_for_a_different_cp_group(monkeypatch):
    backend = _fused_backend(monkeypatch, 64, degree=4, heads=64)
    outputs = (torch.empty(64, 64, 512), torch.empty(1, 64, 64), torch.empty(1, 64, 64))
    with pytest.raises(ValueError, match="cover the CP group"):
        backend._decode.configure_fused_decode(
            group="fused-dsa",
            ranks=tuple(range(8)),
            attention_outputs={64: outputs},
        )
    assert backend._decode._fused_decode is None


def test_dsa_missing_fused_resources_fail_before_native_launch(monkeypatch):
    backend = _fused_backend(monkeypatch, 64, degree=8, heads=64)
    selection = ascend_kvp_decode._AscendSparseSelection(
        torch.empty(8, 64, 8, 576),
        None,
        None,
        None,
        None,
        None,
        3,
        True,
    )
    pool = SimpleNamespace(get_key_buffer=lambda layer: torch.empty(128, 576))
    with pytest.raises(RuntimeError, match="Initialize fused DSA resources"):
        backend._decode._run_indexed_sparse_attention(
            selection,
            SimpleNamespace(layer_id=0, scaling=1.0),
            pool,
            head_major_output=True,
        )
    outputs = (torch.empty(64, 64, 512), torch.empty(1, 64, 64), torch.empty(1, 64, 64))
    backend._decode.configure_fused_decode(
        group="fused-dsa",
        ranks=tuple(backend._decode._dcp.ranks),
        attention_outputs={64: outputs},
    )
    selection = dataclasses.replace(selection, query=torch.empty(8, 65, 8, 576))
    with pytest.raises(ValueError, match="prepared"):
        backend._decode._run_indexed_sparse_attention(
            selection,
            SimpleNamespace(layer_id=0, scaling=1.0),
            pool,
            head_major_output=True,
        )


class _CacheStepPool:
    def __init__(self, layers):
        self.index = {layer: torch.zeros(33, 128, 1, 2) for layer in range(layers)}
        self.kv = {layer: torch.zeros(33, 128, 1, 3) for layer in range(layers)}

    def get_component(self, layer_id, name):
        assert name == "dsa_index_k"
        return self.index[layer_id]

    def get_key_buffer(self, layer_id):
        return self.kv[layer_id]


class _CacheStepConsumer:
    """Read the real host completion word and snapshot the announced cache."""

    def __init__(self, counter, pool):
        self.counter = counter
        self.pool = pool
        self.layer_id = 0
        self.last_step = 0
        self.notifications = []

    def poll(self):
        step = int(self.counter.h_ready_cache_step)
        if step != self.last_step:
            self.notifications.append(
                (step, self.layer_id, self.pool.index[self.layer_id].clone())
            )
            self.last_step = step


class _CacheStepKernels:
    """CPU cache writes and attention in place of unavailable device kernels."""

    def __init__(self, consumer):
        self.consumer = consumer
        self.fail_on_short_chunk = False

    def require_fused_context_parallel(self, *, rank_size, num_heads):
        pass

    def scatter(self, rows, cache, locations):
        self.consumer.poll()
        if self.fail_on_short_chunk and rows.shape[0] < 1024:
            raise RuntimeError("device index-cache write failed")
        owned = locations >= 0
        cache.view(-1, rows.shape[-1])[locations[owned].long()] = rows[owned]

    def index(self, query, cache, weights, *metadata):
        self.consumer.poll()
        return (
            torch.zeros(query.shape[0], 1, 2, dtype=torch.int32),
            torch.ones(query.shape[0], dtype=torch.int32),
        )

    def index_partial(self, query, cache, weights, *metadata, **kwargs):
        self.consumer.poll()
        indices, _ = self.index(query, cache, weights)
        return indices, torch.zeros_like(indices, dtype=torch.float32)

    def select_local(self, indices, positions, rank):
        return indices, torch.ones(indices.shape[0], dtype=torch.int32)

    def prefill_index(self, query, *args):
        self.consumer.poll()
        return torch.zeros(query.shape[0], 2, dtype=torch.int32)

    def prefill_attention(self, query, *args):
        self.consumer.poll()
        return query[..., :2].transpose(0, 1).contiguous()

    def attention(self, query, *args):
        self.consumer.poll()
        return query.clone()

    def attention_partial(self, query, *args, **kwargs):
        self.consumer.poll()
        degree, rows, heads, width = query.shape
        output = query.permute(0, 2, 1, 3).reshape(degree * heads, rows, width)[..., :2]
        maximum = torch.zeros(1, rows, degree * heads)
        return output, maximum, torch.ones_like(maximum)

    def merge_partials(self, output, lse):
        return output[0]


class _CacheStepCollectives:
    """Identical CPU peers replace the unavailable device process group."""

    def __init__(self, degree):
        self.degree = degree
        self.ready = True

    def all_to_all(self, output, values, *, auxiliary):
        output.copy_(values)

    def all_gather(self, values, *, auxiliary):
        return torch.cat([values] * self.degree)

    def combine(self, output, lse, *, merge):
        return output[:, : output.shape[1] // self.degree]


class _CacheStepBackend(ascend_dsa.AscendDSABackend):
    def __init__(self, config, spec, *, kernel_page_size):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        self.prefill_metadata = None
        self.decode_metadata = None

    @property
    def forward_prefill_metadata(self):
        return self.prefill_metadata

    @property
    def forward_decode_metadata(self):
        return self.decode_metadata


def _cache_step_backend(monkeypatch, *, degree, layers):
    from tokenspeed.runtime.pd.utils import StepCounter as DeviceStepCounter

    # Only device allocation is substituted: record_cache and its modulo
    # arithmetic/published host Tensor remain the production implementation.
    tensor = torch.tensor
    with monkeypatch.context() as device:
        device.setattr(torch.Tensor, "cuda", lambda self, gpu_id: self)
        device.setattr(
            torch,
            "tensor",
            lambda *a, **k: tensor(
                *a, **{n: v for n, v in k.items() if n != "pin_memory"}
            ),
        )
        counter = DeviceStepCounter(device="cpu", gpu_id=0)
    pool = _CacheStepPool(layers)
    consumer = _CacheStepConsumer(counter, pool)
    kernels = _CacheStepKernels(consumer)

    def initialize_base(self, config, spec, *, kernel_page_size):
        # The dense leaf constructor needs model/device kernels; none of the
        # Ascend selection, chunking or cache completion methods is replaced.
        self.is_draft = False
        self.spec_num_tokens = 8
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size
        self.step_counter = None
        self.kernel_page_size = kernel_page_size

    monkeypatch.setattr(DSABackend, "__init__", initialize_base)
    monkeypatch.setattr(ascend_dsa, "ascend_dsa_kernels", lambda: kernels)
    # Compact model weights keep this device-free lifecycle test inexpensive.
    config = SimpleNamespace(
        max_bs=64,
        speculative_num_draft_tokens=8,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method=None,
        kvp_size=degree,
        kvp_rank=0,
        kvp_group=tuple(range(degree)),
    )
    spec = SimpleNamespace(
        is_dsa=True,
        index_k_format="bf16",
        attn_tp_size=degree,
        num_attention_heads=degree,
        index_head_dim=2,
        index_topk=2,
        kv_cache_dim=3,
        kv_lora_rank=2,
        qk_rope_head_dim=1,
    )
    backend = _CacheStepBackend(config, spec, kernel_page_size=128)
    backend.step_counter = counter
    placement = CachePlacement(128, 33, tuple(range(degree)), 0)
    if degree > 1:
        backend._decode.bind_cache(placement)
        backend._decode._collective_context = _CacheStepCollectives(degree)
    monkeypatch.setattr(
        backend, "cache_placement", lambda layer: placement if degree > 1 else None
    )
    return backend, pool, consumer


def _cache_step_inputs(backend, pool, consumer, *, rows, layer_id, batch):
    from tokenspeed.runtime.layers.attention.backends.paged.mla import (
        MLAPrefillMetadata,
    )

    consumer.layer_id = layer_id
    extends = torch.tensor([rows], dtype=torch.int32)
    zero = torch.zeros(1, dtype=torch.int32)
    table = torch.arange(1, 33, dtype=torch.int32)[None]
    backend.prefill_metadata = MLAPrefillMetadata(
        seq_lens=extends,
        extend_prefix_lens=zero,
        extend_seq_lens=extends,
        cum_extend_seq_lens=torch.tensor([0, rows], dtype=torch.int32),
        cum_seq_lens_kv=None,
        page_table=table,
        extend_prefix_lens_cpu=zero,
        extend_seq_lens_cpu=extends,
        max_extend_seq_len=rows,
        max_extend_prefix_len=0,
        use_absorbed_cached_extend=False,
        chunked_loop_num=0,
        chunk_kv_indices_list=[],
        chunked_seq_len=torch.empty(0),
        cu_chunked_seq_len=torch.empty(0),
        max_chunk_len_per_loop=[],
    )
    positions = torch.arange(rows, dtype=torch.float32)
    query = positions[:, None, None].expand(-1, 1, 3).contiguous()
    index_key = torch.stack(
        (positions + batch * 10000 + layer_id * 5000 + 1, positions + 1), dim=1
    )
    slots = torch.arange(128, 128 + rows, dtype=torch.int64)
    placement = backend.cache_placement(None)
    if placement is None:
        local, owned = slots, torch.ones(rows, dtype=torch.bool)
    else:
        local, owned = resolve_cache_slots(slots, placement)
    expected = pool.index[layer_id].clone()
    expected.view(-1, 2)[local[owned].long()] = index_key[owned]
    # The real MLA prologue owns KV. Its CPU stand-in lands before attention,
    # while the real Ascend backend writes the index cache a chunk at a time.
    pool.kv[layer_id].fill_(batch + layer_id + 1)
    inputs = dict(
        index_query=torch.zeros(rows, 1, 2),
        index_key=index_key,
        index_weights=torch.ones(rows, 1),
        index_init_tokens=0,
        index_local_tokens=0,
    )
    return query, slots, inputs, expected


class _CpuPrefillTransport:
    """Only device byte-copy kernels are replaced; the real Prefill owns order."""

    def __init__(self, backend):
        self.backend = backend
        self.degree, self.rank = backend._kvp_degree, 0
        self.local_rows = 33 * 128

    def gather_history(self, *args, **kwargs):
        return torch.empty(128, 3), torch.empty(128, 2)

    def gather_current(self, kv, index, *, row_counts):
        return kv, index

    def store_current(
        self, packet, kv, index, kv_cache, index_cache, slots, direct, **kwargs
    ):
        placement = self.backend.cache_placement(None)
        if placement is None:
            local, owned = slots, slots > 0
        else:
            local, owned = resolve_cache_slots(slots, placement)
        locations = torch.where(owned, local, -1)
        self.backend._indexer_kernels.scatter(packet[1], index_cache, locations)
        kv_cache.view(-1, 3)[local[owned].long()] = packet[0][owned]


def _prepare_cache_step_prefill(backend, pool, query, inputs):
    rows, degree = len(query), backend._kvp_degree
    backend.cache_pool = pool
    backend._prefill_layout = AscendQCPPrefillWorkspaceLayout(
        current_capacity=rows,
        null_rows=128,
        history_capacity=4096,
        degree=degree,
        has_index=True,
    )
    prefill = ascend_qcp_prefill.AscendQCPPrefill(
        layout=backend._prefill_layout,
        page_size=128,
        topk=2,
        runtime=_CpuPrefillTransport(backend),
    )
    prefill.row_counts = [rows] + [0] * (degree - 1)
    prefill.row_start, prefill.row_end = 0, rows
    prefill.placement = backend.cache_placement(None)
    prefill.page_table = torch.zeros(1, 1, dtype=torch.int32)
    prefill.query_cu = prefill.key_cu = torch.tensor([0, rows], dtype=torch.int32)
    prefill.query_seq = torch.arange(1, rows + 1, dtype=torch.int32)
    prefill.scan_count = torch.ones(rows, dtype=torch.int32)
    prefill.current_direct = torch.arange(rows)
    prefill.patch_slots = torch.full((rows,), -1, dtype=torch.int64)
    prefill.history_slots = torch.empty(0, dtype=torch.int64)
    prefill.history_counts = (0,) * degree
    prefill.history_rows = 128
    prefill.current_rank_stride = ((rows + 127) // 128) * 128
    prefill.key_row_count = rows
    backend._prefill = prefill
    inputs["prefill_latent"] = torch.ones(rows, 3)
    return query.expand(-1, degree, -1).contiguous()


@pytest.mark.parametrize("degree,rows", [(2, 1), (2, 127), (2, 129), (1, 129)])
def test_prefill_cache_counter_announces_completed_layers_across_batches(
    monkeypatch, degree, rows
):
    backend, pool, consumer = _cache_step_backend(monkeypatch, degree=degree, layers=3)
    completed = 0
    for batch in range(3):
        for layer_id in range(3):
            query, slots, inputs, expected = _cache_step_inputs(
                backend, pool, consumer, rows=rows, layer_id=layer_id, batch=batch
            )
            query = _prepare_cache_step_prefill(backend, pool, query, inputs)
            layer = SimpleNamespace(layer_id=layer_id, scaling=1.0, logit_cap=0)
            backend.forward_extend(
                query, None, None, layer, slots, pool, 1, fused=False, **inputs
            )
            consumer.poll()
            completed += 1
            assert int(backend.step_counter.h_ready_cache_step) == completed
            assert int(backend.step_counter.d_ready_cache_step) == completed
            assert len(consumer.notifications) == completed
            step, announced_layer, cache = consumer.notifications[-1]
            assert (step, announced_layer) == (completed, layer_id)
            torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize("degree", [1, 2])
def test_decode_cache_counter_still_announces_one_completed_layer(monkeypatch, degree):
    from tokenspeed.runtime.layers.attention.backends.paged.mla import MLADecodeMetadata

    backend, pool, consumer = _cache_step_backend(monkeypatch, degree=degree, layers=2)
    for batch in range(3):
        for layer_id in range(2):
            query, slots, inputs, expected = _cache_step_inputs(
                backend, pool, consumer, rows=1, layer_id=layer_id, batch=batch
            )
            backend.decode_metadata = MLADecodeMetadata(
                num_extends=0,
                page_table=torch.arange(1, 33, dtype=torch.int32)[None],
                seq_lens=torch.tensor([129], dtype=torch.int32),
            )
            backend._decode.refresh(backend.decode_metadata)
            layer = SimpleNamespace(layer_id=layer_id, scaling=1.0, logit_cap=0)
            backend.forward_decode(
                query, None, None, layer, slots, pool, 1, fused=False, **inputs
            )
            consumer.poll()
            expected_step = batch * 2 + layer_id + 1
            assert int(backend.step_counter.h_ready_cache_step) == expected_step
            assert len(consumer.notifications) == expected_step
            torch.testing.assert_close(
                consumer.notifications[-1][2], expected, rtol=0, atol=0
            )


def test_incomplete_prefill_cache_write_does_not_publish_layer_completion(monkeypatch):
    backend, pool, consumer = _cache_step_backend(monkeypatch, degree=2, layers=1)
    query, slots, inputs, _ = _cache_step_inputs(
        backend, pool, consumer, rows=129, layer_id=0, batch=0
    )
    query = _prepare_cache_step_prefill(backend, pool, query, inputs)
    backend._indexer_kernels.fail_on_short_chunk = True
    layer = SimpleNamespace(layer_id=0, scaling=1.0, logit_cap=0)
    with pytest.raises(RuntimeError, match="index-cache write failed"):
        backend.forward_extend(
            query, None, None, layer, slots, pool, 1, fused=False, **inputs
        )
    consumer.poll()
    assert int(backend.step_counter.h_ready_cache_step) == 0
    assert not consumer.notifications


@pytest.mark.parametrize("width", [1, 8, 9])
@pytest.mark.parametrize("page,block", [(64, 128), (128, 128), (128, 256)])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_mla_uses_shared_metadata_and_router_slots_for_every_layer(
    width, page, block, slot_dtype
):
    owner = ascend_dcp.AscendDCPMetadata(
        degree=8, rank=1, ranks=tuple(range(8)), virtual_block_count=19
    )
    owner.allocate_decode_buffers(3 * width, 4, "cpu")
    table = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [0, 0, 0, 0]], dtype=torch.int32)
    lengths = torch.tensor([page + 3, 3 * page - 1, width], dtype=torch.int32)
    placement = CachePlacement(block, 19, tuple(range(8)), 1)
    # Authoritative router slots include null, foreign, owned and invalid IDs.
    locations = torch.tensor(
        [0, block, 2 * block + 3, 19 * block, -1] * width, dtype=slot_dtype
    )

    def prepare():
        owner.prepare_metadata(
            seq_lens=lengths,
            page_table=table,
            page_size=page,
            block_granularity=block,
            init_tokens=0,
            local_tokens=0,
            query_width=width,
            query_group_size=width,
            slots=None,
        )

    prepare()
    reference, live = resolve_cache_slots(locations, placement)
    expected = torch.where(live, reference, -1).to(torch.int64)
    local = owner.metadata(start=0, end=3 * width)
    address = local.page_table.data_ptr()
    assert torch.count_nonzero(local.seq_lens[-width:]) == 0
    slots = owner.prepare_write_slots(locations, placement)
    torch.testing.assert_close(slots, expected, atol=0, rtol=0)
    for _ in range(4):
        assert owner.prepare_write_slots(locations.view(-1), placement) is slots
        assert owner.metadata(start=0, end=3 * width).page_table.data_ptr() == address
    # New storage with the same geometry must replace the cached slot mapping.
    replacement = torch.full_like(locations, 2 * block + 5)
    replaced_slots = owner.prepare_write_slots(replacement, placement)
    assert replaced_slots is not slots
    torch.testing.assert_close(
        replaced_slots,
        torch.full_like(locations, block + 5, dtype=torch.int64),
        atol=0,
        rtol=0,
    )
    replacement.set_(torch.full_like(replacement, 2 * block + 9))
    rebound_slots = owner.prepare_write_slots(replacement, placement)
    assert rebound_slots is not replaced_slots
    torch.testing.assert_close(
        rebound_slots,
        torch.full_like(locations, block + 9, dtype=torch.int64),
        atol=0,
        rtol=0,
    )
    # Next step overwrites the same router/input storage; no stale metadata or
    # ownership mask can escape the step invalidation.
    owner.clear_forward_metadata()
    table.zero_()
    locations.fill_(2 * block + 7)
    lengths.fill_(width)
    prepare()
    assert owner.page_table.data_ptr() == address
    assert torch.count_nonzero(owner.seq_lens) == 0
    slots = owner.prepare_write_slots(locations, placement)
    torch.testing.assert_close(
        slots,
        torch.full_like(locations, block + 7, dtype=torch.int64),
        atol=0,
        rtol=0,
    )


def test_shared_metadata_reuses_alias_views_and_refreshes_replacement_storage():
    owner = ascend_dcp.AscendDCPMetadata(
        degree=8, rank=1, ranks=tuple(range(8)), virtual_block_count=19
    )
    owner.allocate_decode_buffers(2, 4, "cpu")
    lengths = torch.tensor([256, 384], dtype=torch.int32)
    table = torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]], dtype=torch.int32)

    def prepare(seq_lens, page_table):
        owner.prepare_metadata(
            seq_lens=seq_lens,
            page_table=page_table,
            page_size=128,
            block_granularity=128,
            init_tokens=0,
            local_tokens=0,
            query_width=1,
            query_group_size=1,
            slots=None,
        )

    prepare(lengths, table)
    expected = owner.seq_lens.clone()
    # The prepared results remain layer-shared until forward invalidation.
    lengths.zero_()
    prepare(lengths.view_as(lengths), table.view_as(table))
    torch.testing.assert_close(owner.seq_lens, expected, atol=0, rtol=0)
    prepare(lengths.clone(), table.clone())
    assert torch.count_nonzero(owner.seq_lens) == 0
    # Rebinding the same Tensor object must also invalidate its old view.
    prepare(lengths, table)
    lengths.set_(torch.tensor([256, 384], dtype=torch.int32))
    prepare(lengths, table)
    torch.testing.assert_close(owner.seq_lens, expected, atol=0, rtol=0)


@pytest.mark.parametrize("initial,window", [(-1, 0), (0, -1), (2048, 1)])
def test_decode_policy_rejects_invalid_budget_before_cache_write(
    monkeypatch, initial, window
):
    backend = _fused_backend(monkeypatch, 1, degree=1, heads=64)
    with pytest.raises(ValueError, match="token"):
        backend.forward_decode(
            torch.zeros(1, 1, 576),
            None,
            None,
            SimpleNamespace(logit_cap=0),
            None,
            None,
            1,
            fused=False,
            index_init_tokens=initial,
            index_local_tokens=window,
        )


@pytest.mark.parametrize(
    "fused,packed,message",
    [
        (True, False, "requires the fused Prolog query"),
        (False, True, "does not accept a fused Prolog query"),
        (True, True, "Initialize fused DSA resources"),
    ],
)
def test_decode_rejects_mismatched_prolog_before_cache_write(
    monkeypatch, fused, packed, message
):
    backend = _fused_backend(monkeypatch, 8, degree=8, heads=64)
    with pytest.raises((ValueError, RuntimeError), match=message):
        backend.forward_decode(
            torch.zeros(8, 8, 576),
            None,
            None,
            SimpleNamespace(logit_cap=0),
            None,
            None,
            1,
            fused=fused,
            index_init_tokens=0,
            index_local_tokens=0,
            packed_query=torch.zeros(8, 8, 8, 576) if packed else None,
        )


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_decode_grouped_indexer_preserves_policy_and_each_causal_end(
    monkeypatch, width
):
    class ReachedIndexer(Exception):
        pass

    initial, window = 130, 129
    records = []

    class Indexer:
        def index_partial(
            self,
            query,
            cache,
            weights,
            q_ends,
            lengths,
            table,
            topk,
            initial_counts,
            local_counts,
            *,
            sparse_mode,
            visible_lengths,
        ):
            records.append(
                (
                    q_ends.clone(),
                    lengths.clone(),
                    initial_counts.clone(),
                    local_counts.clone(),
                    (
                        lengths.clone()
                        if visible_lengths is None
                        else visible_lengths.clone()
                    ),
                )
            )
            raise ReachedIndexer

    backend = _fused_backend(monkeypatch, width, degree=2, rank=1, heads=64)
    backend._decode._decode_indexer = _cpu_indexer(Indexer())
    table = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    cache = torch.zeros(9, 128, 1, 128)
    with pytest.raises(ReachedIndexer):
        _selection_for_test(
            backend,
            torch.zeros(width, 32, 576),
            SimpleNamespace(layer_id=0),
            torch.arange(width),
            SimpleNamespace(get_component=lambda *args: cache),
            torch.tensor([260], dtype=torch.int32),
            table,
            dict(
                index_query=None,
                index_weights=None,
                index_cache_written=True,
                index_init_tokens=initial,
                index_local_tokens=window,
            ),
            width=width,
            fused=False,
        )
    ends, lengths, prefix, recent, visible = records[0]
    causal = list(range(261 - width, 261))
    owned = lambda n: sum(128 <= p < 256 for p in range(n))
    assert ends.tolist() == [width]
    assert lengths.tolist() == [owned(causal[-1])]
    assert prefix.tolist() == [owned(min(initial, causal[-1]))]
    assert visible.tolist() == [owned(n) for n in causal]
    assert recent.tolist() == [owned(n) - owned(max(0, n - window)) for n in causal]


def test_mixed_decode_uses_only_decode_requests(monkeypatch):
    backend = _fused_backend(monkeypatch, 1, degree=1, heads=64)
    metadata = SimpleNamespace(
        num_extends=1,
        q_len_per_req=1,
        seq_lens=torch.tensor([12, 24], dtype=torch.int32),
        page_table=torch.tensor([[1], [2]], dtype=torch.int32),
    )
    backend._decode.refresh(metadata)
    assert backend._decode.query_ends.tolist() == [1]
    assert backend._decode.seq_lens.tolist() == [24]
    assert backend._decode.page_table.tolist() == [[2]]


def test_failed_decode_index_write_does_not_publish_completion(monkeypatch):
    from tokenspeed.runtime.layers.attention.backends.paged.mla import MLADecodeMetadata

    backend, pool, consumer = _cache_step_backend(monkeypatch, degree=1, layers=1)
    query, slots, inputs, _ = _cache_step_inputs(
        backend, pool, consumer, rows=1, layer_id=0, batch=0
    )
    backend._indexer_kernels.fail_on_short_chunk = True
    backend.decode_metadata = MLADecodeMetadata(
        num_extends=0,
        page_table=torch.tensor([[1, 2]], dtype=torch.int32),
        seq_lens=torch.tensor([129], dtype=torch.int32),
    )
    backend._decode.refresh(backend.decode_metadata)
    before = pool.get_key_buffer(0).clone()
    with pytest.raises(RuntimeError, match="index-cache write failed"):
        backend.forward_decode(
            query,
            None,
            None,
            SimpleNamespace(layer_id=0, logit_cap=0),
            slots,
            pool,
            1,
            fused=False,
            **inputs,
        )
    consumer.poll()
    assert int(backend.step_counter.h_ready_cache_step) == 0
    assert not consumer.notifications
    torch.testing.assert_close(pool.get_key_buffer(0), before, rtol=0, atol=0)


def test_fused_buffer_binding_validates_then_releases_borrowed_views(monkeypatch):
    backend = _fused_backend(monkeypatch, 8, degree=8, heads=64)
    outputs = (torch.empty(64, 8, 512), torch.empty(1, 8, 64), torch.empty(1, 8, 64))
    with pytest.raises(ValueError, match="FP32 SFA producer"):
        backend._decode.configure_fused_decode(
            group="fused-dsa", ranks=tuple(range(8)), attention_outputs={8: outputs[:1]}
        )
    assert backend._decode._fused_decode is None
    backend._decode.configure_fused_decode(
        group="fused-dsa", ranks=tuple(range(8)), attention_outputs={8: outputs}
    )
    backend._decode.clear_fused_decode()
    outputs[0].fill_(17)
    assert float(outputs[0][0, 0, 0]) == 17
    with pytest.raises(RuntimeError, match="Initialize fused DSA resources"):
        backend.forward_decode(
            torch.zeros(8, 8, 576),
            None,
            None,
            SimpleNamespace(logit_cap=0),
            None,
            None,
            1,
            fused=True,
            index_init_tokens=0,
            index_local_tokens=0,
            packed_query=torch.zeros(8, 8, 8, 576),
        )
