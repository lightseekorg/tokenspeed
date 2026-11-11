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
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry

from tokenspeed.runtime.layers.attention import backends
from tokenspeed.runtime.layers.attention.backends.paged import ascend_dsa
from tokenspeed.runtime.layers.attention.dcp import ascend as ascend_dcp
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.registry import _BACKEND_REGISTRY


def _backend_config(
    *, degree, rank, heads, index_dim=128, topk=2048, latent=512, rope=64
):
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig

    # Use the portable MLA metadata contract on this CPU host. Device math is
    # supplied independently through the registry; the real constructors run.
    spec = DSAConfig(
        backend_name="tokenspeed_mla",
        num_attention_heads=heads,
        num_kv_heads=1,
        head_dim=latent + rope,
        attn_tp_size=degree,
        kv_lora_rank=latent,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope,
        v_head_dim=128,
        scaling=1.0,
        kv_cache_dim=latent + rope,
        index_topk=topk,
        index_head_dim=index_dim,
        index_n_heads=heads,
        index_k_format="bf16",
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=8192,
        max_bs=512,
        speculative_num_draft_tokens=8,
        kvp_size=degree,
        kvp_rank=rank,
        kvp_group=tuple(range(degree)),
        components=(spec,),
    )
    return config, spec


def _install_device_operations(monkeypatch, operations):
    """Use real registry dispatch with CPU stand-ins only for absent NPU math."""
    from tokenspeed_kernel.platform import CapabilityRequirement
    from tokenspeed_kernel.registry import register_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    previous = KernelRegistry.get()
    registry = KernelRegistry()
    for name, spec in previous._by_name.items():
        registry.register(spec, previous.get_impl(name))
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    methods = {
        "scatter": "store_index_keys",
        "index": "dsa_local_topk",
        "prefill_index": "dsa_gathered_prefill_topk",
        "prefill_attention": "dsa_gathered_prefill_attention",
        "index_partial": "dsa_local_topk_candidates",
        "select_local": "dsa_select_local_topk",
        "attention": "dsa_sparse_decode",
        "attention_partial": "dsa_sparse_decode_partial",
        "merge_topk_fused": "merge_dsa_topk",
        "merge_lse_direct": "merge_dsa_lse",
        "merge_partials": "merge_normalized_partials",
    }
    for method, mode in methods.items():
        function = getattr(operations, method, None)
        if function is not None:
            register_kernel(
                "attention",
                mode,
                name=f"cpu_test_{mode}",
                solution="cpu_test",
                capability=CapabilityRequirement(vendors=frozenset({"cpu"})),
                signatures=[
                    format_signature(input=dense_tensor_format(dtype))
                    for dtype in (
                        torch.bfloat16,
                        torch.float32,
                        torch.int32,
                        torch.int64,
                    )
                ],
            )(function)
    validate = getattr(operations, "require_fused_context_parallel", None)
    if validate is not None:
        monkeypatch.setattr(ascend_dsa, "validate_dsa_fused_decode", validate)


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

    monkeypatch.setattr(ascend_dcp, "dsa_cp_metadata", reference)


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


@pytest.mark.parametrize(
    "degree,rank,missing",
    [
        (1, 0, None),
        (1, 0, "npu_dcp_merge_topk"),
        (2, 0, None),
        (4, 0, None),
        (8, 3, None),
        (8, 3, "npu_dcp_merge_topk"),
        (8, 3, "npu_dsa_merge_lse_direct"),
        (16, 0, None),
    ],
)
def test_ascend_dsa_consumes_host_kvp_topology(monkeypatch, degree, rank, missing):
    def require_fused(*, rank_size, num_heads):
        if missing is not None:
            raise RuntimeError(f"Ascend DSA CP requires fused operators: {missing}")

    monkeypatch.setattr(ascend_dsa, "validate_dsa_fused_decode", require_fused)
    config, spec = _backend_config(degree=degree, rank=rank, heads=2 * degree)
    backend = ascend_dsa.AscendDSABackend(config, spec, kernel_page_size=128)
    assert (backend._dcp.degree, backend._dcp.rank, backend._dcp.ranks) == (
        degree,
        rank,
        config.kvp_group,
    )
    if degree > 1:
        if missing is not None:
            with pytest.raises(RuntimeError, match=missing):
                backend.configure_fused_decode(
                    group="test", ranks=config.kvp_group, attention_outputs={64: ()}
                )
        spec.attn_tp_size = 1
        with pytest.raises(ValueError, match="complete head-TP group"):
            ascend_dsa.AscendDSABackend(config, spec, kernel_page_size=128)


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


def test_indexer_writes_only_its_own_cache_after_mla_prologue(monkeypatch):
    index_cache = torch.full((2, 128, 1, 32), -7.0)

    class Kernels:
        def scatter(self, rows, cache, locations):
            cache.view(-1, 32).index_copy_(0, locations.long(), rows.view(-1, 32))

        def index(self, *args):
            return torch.zeros((1, 1), dtype=torch.int32), torch.ones(
                (1,), dtype=torch.int32
            )

    class Pool:
        def get_key_buffer(self, layer_id):
            raise AssertionError("MLA prologue already wrote the KV cache")

        def get_component(self, layer_id, name):
            assert name == "dsa_index_k"
            return index_cache

    _install_device_operations(monkeypatch, Kernels())
    backend = SimpleNamespace(
        _indexer_spec=SimpleNamespace(index_head_dim=32, index_topk=1),
        kernel_page_size=128,
        step_counter=None,
        _dcp=SimpleNamespace(degree=1),
        cache_placement=lambda layer: None,
    )
    key = torch.ones((1, 1, 32))
    locations = torch.tensor([131], dtype=torch.int32)
    ascend_dsa.AscendDSABackend._select_indexed(
        backend,
        torch.ones((1, 1, 8)),
        None,
        SimpleNamespace(layer_id=0),
        locations,
        Pool(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        {
            "index_key": key,
            "index_query": torch.ones((1, 1, 32)),
            "index_weights": torch.ones((1, 1)),
            "index_init_tokens": 0,
            "index_local_tokens": 0,
        },
        query_group_size=1,
        query_width=1,
        context_parallel=False,
        fused=False,
    )
    torch.testing.assert_close(index_cache.view(-1, 32)[131], key.view(-1))
    torch.testing.assert_close(index_cache[0], torch.full_like(index_cache[0], -7.0))


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


def test_dcp_metadata_is_shared_only_within_matching_forward():
    dcp = ascend_dcp.AscendDCPMetadata(
        virtual_block_count=17,
        degree=8,
        rank=1,
        ranks=tuple(range(8)),
    )
    table = torch.tensor([[1, 2, 3]] * 4, dtype=torch.int32)
    lengths = torch.tensor([254, 255, 256, 257], dtype=torch.int32)

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

    prepare()
    address = dcp.page_table.data_ptr()
    for _ in range(2):
        prepare()
        assert dcp.page_table.data_ptr() == address
    assert dcp.seq_lens.tolist() == [126, 127, 128, 128]
    # The next round can update the same device addresses in place.
    backend = SimpleNamespace(_ascend_placement=None, _dcp=dcp)
    ascend_dsa.AscendDSABackend._clear_forward_scratch(backend)
    lengths.fill_(130)
    table[:, 1] = 10
    prepare()
    assert dcp.seq_lens.tolist() == [2] * 4
    assert dcp.page_table[:, 0].tolist() == [2] * 4
    prepare(local=1)
    assert dcp.local_counts.tolist() == [1] * 4
    prepare(local=1, rows=2)
    assert dcp.seq_lens.tolist() == [2] * 2
    # Rebinding, even to the same capacity, invalidates the forward result.
    lengths.fill_(140)
    table[:, 1] = 2
    dcp.bind_virtual_block_count(17)
    prepare(local=1, rows=2)
    assert dcp.seq_lens.tolist() == [12, 12]


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


@pytest.mark.parametrize("degree", [1, 2, 8])
@pytest.mark.parametrize("block_size", [128, 256])
def test_sharded_index_write_and_read_use_local_storage(
    monkeypatch, degree, block_size
):
    """Virtual IDs exceed each rank's capacity; replicated addressing fails."""

    class ReachedIndexer(Exception):
        pass

    rows = 8
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

        _install_device_operations(monkeypatch, Kernels())
        backend = SimpleNamespace(
            _indexer_spec=SimpleNamespace(index_head_dim=8, index_topk=512),
            kernel_page_size=128,
            step_counter=None,
            _dcp=dcp,
            _fused_group="test-dsa",
            cache_placement=lambda layer: placement,
        )
        with pytest.raises(ReachedIndexer):
            ascend_dsa.AscendDSABackend._select_indexed(
                backend,
                torch.zeros(8, 1, 8),
                None,
                SimpleNamespace(layer_id=0),
                slots,
                SimpleNamespace(get_component=lambda *args: cache),
                torch.arange(1, 9, dtype=torch.int32),
                lengths,
                table,
                dict(
                    index_key=keys,
                    index_query=torch.zeros(rows, 64, 128, dtype=torch.bfloat16),
                    index_weights=torch.zeros(rows, 64),
                    index_init_tokens=131,
                    index_local_tokens=129,
                ),
                context_parallel=True,
                query_group_size=1,
                query_width=1,
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


@pytest.mark.parametrize("initial,local", [(-1, 0), (0, -1), (3, 2)])
def test_index_policy_is_validated_at_execution(initial, local):
    backend = SimpleNamespace(_indexer_spec=SimpleNamespace(index_topk=4))
    with pytest.raises(ValueError):
        ascend_dsa.AscendDSABackend._select_indexed(
            backend,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            {"index_init_tokens": initial, "index_local_tokens": local},
            query_group_size=1,
            query_width=1,
            context_parallel=False,
            fused=False,
        )


@pytest.mark.parametrize("width", [2, 4, 8])
def test_dcp_indexer_receives_model_policy_for_each_candidate(monkeypatch, width):
    class ReachedIndexer(Exception):
        pass

    lengths = torch.arange(253, 253 + width, dtype=torch.int32)
    table = torch.tensor([[1, 2, 3]] * width, dtype=torch.int32)
    initial, local = 130, 129
    records = []

    class Dcp(ascend_dcp.AscendDCPMetadata):
        def refresh_metadata(
            self,
            *,
            seq_lens,
            page_table,
            page_size,
            block_granularity,
            init_tokens,
            local_tokens,
            **metadata_args,
        ):
            assert init_tokens == initial and local_tokens == local
            super().refresh_metadata(
                page_table=page_table,
                seq_lens=seq_lens,
                page_size=page_size,
                block_granularity=block_granularity,
                init_tokens=init_tokens,
                local_tokens=local_tokens,
                **metadata_args,
            )

    class Kernels:
        def scatter(self, *args):
            pass

        def index_partial(self, *args, **kwargs):
            records.append(args)
            raise ReachedIndexer

    _install_device_operations(monkeypatch, Kernels())
    backend = SimpleNamespace(
        _indexer_spec=SimpleNamespace(index_head_dim=8, index_topk=512),
        kernel_page_size=128,
        step_counter=None,
        _fused_group="test-dsa",
        _dcp=Dcp(
            degree=2,
            rank=1,
            ranks=(0, 1),
            virtual_block_count=8,
        ),
        cache_placement=lambda layer: CachePlacement(128, 8, (0, 1), 1),
    )
    with pytest.raises(ReachedIndexer):
        ascend_dsa.AscendDSABackend._select_indexed(
            backend,
            torch.ones(width, 1, 8),
            None,
            SimpleNamespace(layer_id=0),
            torch.arange(width),
            SimpleNamespace(get_component=lambda *args: torch.zeros(3, 128, 1, 8)),
            torch.arange(1, width + 1, dtype=torch.int32),
            lengths,
            table,
            {
                "index_key": torch.ones(width, 1, 8),
                "index_query": torch.ones(width, 1, 8),
                "index_weights": torch.ones(width, 1),
                "index_init_tokens": initial,
                "index_local_tokens": local,
            },
            context_parallel=True,
            query_group_size=1,
            query_width=1,
            fused=False,
        )
    args = records[0]
    expected_initial = [
        sum(128 <= p < 256 for p in range(min(initial, n))) for n in lengths.tolist()
    ]
    expected_local = [
        sum(128 <= p < 256 for p in range(max(n - local, 0), n))
        for n in lengths.tolist()
    ]
    assert args[7].tolist() == expected_initial
    assert args[8].tolist() == expected_local
    assert args[4].tolist() == [min(max(n - 128, 0), 128) for n in lengths.tolist()]


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
        peers = self.peer_outputs[:rank_size].expand(-1, 64, self.rows, 512).clone()
        maxima = self.peer_maximum[:rank_size].expand(-1, self.rows, 64).clone()
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
    monkeypatch, rows, *, degree, rank=0, backend_class=ascend_dsa.AscendDSABackend
):
    kernels = _CpuFusedDsaKernels(rows, degree)
    _install_device_operations(monkeypatch, kernels)
    config, spec = _backend_config(degree=degree, rank=rank, heads=64)
    backend = backend_class(config, spec, kernel_page_size=128)
    backend._dcp.bind_virtual_block_count(17)
    backend.configure_runtime(
        block_granularity=128, virtual_block_count=17, shard_count=degree
    )
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
                backend = _fused_backend(patch, rows, degree=degree, rank=rank)
                _install_device_operations(
                    patch, _CpuCollectiveDsaKernels(rows, degree, rank)
                )
                # Small shapes must retain the general path even after the
                # model has configured native buffers for larger shapes.
                backend.configure_fused_decode(
                    group="fused-dsa",
                    ranks=tuple(backend._dcp.ranks),
                    attention_outputs={64: (torch.empty(64, 64, 512),)},
                )
                query = torch.full((rows, 8, 576), rank, dtype=torch.bfloat16)
                layer = SimpleNamespace(layer_id=0, scaling=1.0)
                index_cache = torch.zeros(8, 128, 1, 128, dtype=torch.bfloat16)
                kv_cache = torch.zeros(8, 128, 1, 576, dtype=torch.bfloat16)
                pool = SimpleNamespace(
                    get_component=lambda *args: index_cache,
                    get_key_buffer=lambda *args: kv_cache,
                )
                selection = backend._select_indexed(
                    query,
                    None,
                    layer,
                    torch.arange(rows),
                    pool,
                    torch.arange(1, rows + 1, dtype=torch.int32),
                    torch.full((rows,), 128, dtype=torch.int32),
                    torch.ones(rows, 1, dtype=torch.int32),
                    dict(
                        index_query=torch.zeros(rows, 64, 128, dtype=torch.bfloat16),
                        index_weights=torch.zeros(rows, 64),
                        index_init_tokens=0,
                        index_local_tokens=0,
                        index_cache_written=True,
                    ),
                    query_group_size=1,
                    context_parallel=True,
                    query_width=1,
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
                    for is_prefill in (False, True):
                        actual = backend._run_indexed_sparse_attention(
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


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_single_rank_verify_preserves_each_candidates_causal_context(
    monkeypatch, width
):
    class SingleRankKernels:
        def index(self, query, key, weights, *metadata):
            return torch.zeros(query.shape[0], 1, 1, dtype=torch.int32), torch.ones(
                query.shape[0], dtype=torch.int32
            )

    rows = 2 * width
    backend = _fused_backend(monkeypatch, rows, degree=1)
    _install_device_operations(monkeypatch, SingleRankKernels())
    monkeypatch.setattr(backend, "cache_placement", lambda layer: None)
    table = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    lengths = torch.tensor([129, 257], dtype=torch.int32)
    cache = torch.zeros(8, 128, 1, 128, dtype=torch.bfloat16)
    selection = backend._select_indexed(
        torch.zeros(rows, 1, 576),
        None,
        SimpleNamespace(layer_id=0),
        torch.zeros(rows, dtype=torch.int64),
        SimpleNamespace(get_component=lambda *args: cache),
        None,
        lengths,
        table,
        dict(
            index_query=torch.zeros(rows, 16, 128),
            index_weights=torch.zeros(rows, 64),
            index_init_tokens=0,
            index_local_tokens=0,
            index_cache_written=True,
        ),
        query_group_size=width,
        query_width=width,
        context_parallel=False,
        fused=False,
    )
    assert selection.q_ends.tolist() == list(range(1, rows + 1))
    assert selection.kv_lengths.tolist() == list(range(130 - width, 130)) + list(
        range(258 - width, 258)
    )
    assert selection.table.tolist() == [[1, 2, 3]] * width + [[4, 5, 6]] * width
    assert selection.context_parallel is False
    assert lengths.tolist() == [129, 257]


def test_dsa_metadata_needs_no_collective_process_group(monkeypatch):
    # The global runtime may exist while this backend has no HCCL groups.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    backend = _fused_backend(monkeypatch, 64, degree=8)
    backend._dcp.refresh_metadata(
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
    metadata = backend._dcp.metadata(start=0, end=1)
    assert metadata.seq_lens.tolist() == [128]
    assert metadata.page_table.tolist() == [[1, 0]]


@pytest.mark.parametrize("degree,rows", [(8, 64), (8, 512)])
@pytest.mark.parametrize("head_major", [False, True])
def test_fused_dsa_uses_prepared_outputs_and_merges_lse(
    monkeypatch, degree, rows, head_major
):
    backend = _fused_backend(monkeypatch, rows, degree=degree)
    outputs = (
        torch.empty(64, rows, 512),
        torch.empty(1, rows, 64),
        torch.empty(1, rows, 64),
    )
    backend.configure_fused_decode(
        group="fused-dsa",
        ranks=tuple(backend._dcp.ranks),
        attention_outputs={rows: outputs},
    )
    local_heads = 64 // degree
    packed_query = torch.zeros(degree, rows, local_heads, 576, dtype=torch.bfloat16)
    layer = SimpleNamespace(layer_id=0, scaling=1.0)
    index_cache = torch.zeros(2, 128, 1, 128, dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 128, 1, 576, dtype=torch.bfloat16)
    pool = SimpleNamespace(
        get_component=lambda *args: index_cache,
        get_key_buffer=lambda *args: kv_cache,
    )
    selection = backend._select_indexed(
        packed_query[0],
        None,
        layer,
        torch.arange(rows),
        pool,
        torch.arange(1, rows + 1, dtype=torch.int32),
        torch.full((rows,), 128, dtype=torch.int32),
        torch.ones(rows, 1, dtype=torch.int32),
        dict(
            index_query=torch.zeros(rows, 64, 128, dtype=torch.bfloat16),
            index_weights=torch.zeros(rows, 64),
            index_init_tokens=0,
            index_local_tokens=0,
            index_cache_written=True,
            packed_query=packed_query,
        ),
        query_group_size=8,
        context_parallel=True,
        query_width=1,
        fused=True,
    )
    assert selection.query is packed_query
    assert selection.indices.shape == (rows, 2048)
    actual = backend._run_indexed_sparse_attention(
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
    backend = _fused_backend(monkeypatch, 64, degree=4)
    outputs = (torch.empty(64, 64, 512), torch.empty(1, 64, 64), torch.empty(1, 64, 64))
    with pytest.raises(ValueError, match="cover the CP group"):
        backend.configure_fused_decode(
            group="fused-dsa",
            ranks=tuple(range(8)),
            attention_outputs={64: outputs},
        )
    assert backend._fused_group is None


def test_dsa_missing_fused_resources_fail_before_native_launch(monkeypatch):
    backend = _fused_backend(monkeypatch, 64, degree=8)
    selection = ascend_dsa._AscendSparseSelection(
        torch.empty(8, 64, 8, 576),
        None,
        None,
        None,
        None,
        None,
        3,
        True,
        True,
    )
    pool = SimpleNamespace(get_key_buffer=lambda layer: torch.empty(128, 576))
    with pytest.raises(RuntimeError, match="Initialize fused DSA resources"):
        backend._run_indexed_sparse_attention(
            selection,
            SimpleNamespace(layer_id=0),
            pool,
            head_major_output=True,
        )
    outputs = (torch.empty(64, 64, 512), torch.empty(1, 64, 64), torch.empty(1, 64, 64))
    backend.configure_fused_decode(
        group="fused-dsa",
        ranks=tuple(backend._dcp.ranks),
        attention_outputs={64: outputs},
    )
    selection = dataclasses.replace(selection, query=torch.empty(8, 65, 8, 576))
    with pytest.raises(ValueError, match="prepared"):
        backend._run_indexed_sparse_attention(
            selection,
            SimpleNamespace(layer_id=0),
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
        self.fail_index_store = False
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

    def prefill_index(self, query, *args):
        self.consumer.poll()
        return torch.zeros(query.shape[0], 2, dtype=torch.int32)

    def prefill_attention(self, query, *args):
        self.consumer.poll()
        return query[..., :2].transpose(0, 1).contiguous()

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
        self._fixture_prefill_metadata = None
        self.decode_metadata = None

    @property
    def forward_prefill_metadata(self):
        return self._fixture_prefill_metadata

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

    _install_device_operations(monkeypatch, kernels)
    config, spec = _backend_config(
        degree=degree,
        rank=0,
        heads=degree,
        index_dim=2,
        topk=2,
        latent=2,
        rope=1,
    )
    backend = _CacheStepBackend(config, spec, kernel_page_size=128)
    backend.step_counter = counter
    backend._dcp.bind_virtual_block_count(33)
    backend._collective_context = _CacheStepCollectives(degree)
    placement = CachePlacement(128, 33, tuple(range(degree)), 0)
    backend.configure_runtime(
        block_granularity=128, virtual_block_count=33, shard_count=degree
    )

    def store_current_index(
        workspace, direct, mapped_slots, *, block_rows, virtual_blocks
    ):
        consumer.poll()
        if consumer.fail_index_store:
            raise RuntimeError("device index-cache write failed")
        current = torch.cat(
            [
                workspace.index[
                    workspace.history_rows
                    + owner * workspace.send_rows : workspace.history_rows
                    + owner * workspace.send_rows
                    + count
                ]
                for owner, count in enumerate(
                    backend.prefill_cache_metadata.projection_row_counts
                )
            ]
        )
        slots = workspace.write_slots
        if degree > 1:
            locations, owned = resolve_cache_slots(slots, placement)
        else:
            locations, owned = slots, slots >= block_rows
        workspace.source_index.view(-1, 2)[locations[owned].long()] = current[owned]
        update = (direct == 0) & (mapped_slots >= 0)
        workspace.index[mapped_slots[update].long()] = current[update]

    monkeypatch.setattr(ascend_dsa, "store_gathered_index_keys", store_current_index)
    monkeypatch.setattr(ascend_dsa, "finish_prefill_workspace", lambda workspace: None)
    return backend, pool, consumer


def _cache_step_inputs(backend, pool, consumer, *, rows, layer_id, batch, prefill):
    from tokenspeed.runtime.layers.attention.backends.paged.mla import (
        MLAPrefillMetadata,
    )

    consumer.layer_id = layer_id
    extends = torch.tensor([rows], dtype=torch.int32)
    zero = torch.zeros(1, dtype=torch.int32)
    table = torch.arange(1, 33, dtype=torch.int32)[None]
    backend._fixture_prefill_metadata = MLAPrefillMetadata(
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
    if prefill:
        from tokenspeed_kernel.ops.attention.ascend_prefill import PrefillWorkspace

        from tokenspeed.runtime.layers.attention.dcp.ascend import (
            AscendPrefillOptions,
            DSAPrefillQueryMetadata,
            PrefillCacheMetadata,
        )

        layout = PrefillCacheMetadata.build(
            backend._fixture_prefill_metadata,
            placement,
            degree=backend._dcp.degree,
            rank=0,
            page_size=128,
            local_cache_pages=33,
            options=AscendPrefillOptions(rows, 128),
        )
        backend._prefill_cache_metadata = layout
        backend._prefill_query_metadata = DSAPrefillQueryMetadata.build(
            backend._fixture_prefill_metadata,
            layout,
            topk=2,
        )
        history_rows = 128 + sum(layout.history_gather.row_counts)
        send_rows = -(-rows // (backend._dcp.degree * 128)) * 128
        index = torch.zeros(history_rows + backend._dcp.degree * send_rows, 2)
        for owner, count in enumerate(layout.projection_row_counts):
            source = sum(layout.projection_row_counts[:owner])
            dest = history_rows + owner * send_rows
            index[dest : dest + count] = index_key[source : source + count]
        backend._prefill_workspace = PrefillWorkspace(
            layer_id,
            layout,
            pool.kv[layer_id],
            pool.index[layer_id],
            torch.zeros(index.shape[0], 3),
            index,
            history_rows,
            send_rows,
            rows,
            tuple(backend._dcp.ranks),
            0,
            None,
            slots.clone(),
            0,
        )
        query = (
            query[layout.query_start : layout.query_end]
            .expand(-1, backend.num_local_heads * backend._dcp.degree, -1)
            .contiguous()
        )
    return query, slots, inputs, expected


@pytest.mark.parametrize("degree,rows", [(1, 7), (8, 17), (8, 129)])
def test_prefill_cache_counter_announces_completed_layers_across_batches(
    monkeypatch, degree, rows
):
    backend, pool, consumer = _cache_step_backend(monkeypatch, degree=degree, layers=3)
    completed = 0
    for batch in range(3):
        for layer_id in range(3):
            query, slots, inputs, expected = _cache_step_inputs(
                backend,
                pool,
                consumer,
                rows=rows,
                layer_id=layer_id,
                batch=batch,
                prefill=True,
            )
            layer = SimpleNamespace(layer_id=layer_id, scaling=1.0, logit_cap=0)
            indices = backend.select_prefill_indices(
                inputs["index_query"][: query.shape[0]],
                inputs["index_weights"][: query.shape[0]],
                layer,
                initial=0,
                window=0,
            )
            backend.forward_extend(
                query,
                None,
                None,
                layer,
                slots,
                pool,
                1,
                fused=False,
                prefill_indices=indices,
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
                backend,
                pool,
                consumer,
                rows=1,
                layer_id=layer_id,
                batch=batch,
                prefill=False,
            )
            backend.decode_metadata = MLADecodeMetadata(
                num_extends=0,
                page_table=torch.arange(1, 33, dtype=torch.int32)[None],
                seq_lens=torch.tensor([129], dtype=torch.int32),
            )
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
        backend, pool, consumer, rows=7, layer_id=0, batch=0, prefill=True
    )
    consumer.fail_index_store = True
    layer = SimpleNamespace(layer_id=0, scaling=1.0, logit_cap=0)
    with pytest.raises(RuntimeError, match="index-cache write failed"):
        backend.select_prefill_indices(
            inputs["index_query"][: query.shape[0]],
            inputs["index_weights"][: query.shape[0]],
            layer,
            initial=0,
            window=0,
        )
    consumer.poll()
    assert int(backend.step_counter.h_ready_cache_step) == 0
    assert not consumer.notifications


def test_fused_selection_rejects_missing_prolog_query(monkeypatch):
    backend = _fused_backend(monkeypatch, 64, degree=8)
    with pytest.raises(ValueError, match="requires the fused Prolog query"):
        backend._select_indexed(
            torch.empty(64, 8, 576),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            {},
            fused=True,
            query_group_size=8,
            context_parallel=True,
            query_width=8,
        )


def test_ordinary_selection_rejects_fused_prolog_query(monkeypatch):
    backend = _fused_backend(monkeypatch, 64, degree=8)
    with pytest.raises(ValueError, match="does not accept a fused Prolog query"):
        backend._select_indexed(
            torch.empty(64, 8, 576),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            {"packed_query": torch.empty(8, 64, 8, 576)},
            fused=False,
            query_group_size=8,
            context_parallel=True,
            query_width=8,
        )


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
