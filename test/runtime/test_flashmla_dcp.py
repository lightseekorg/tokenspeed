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

import os
import sys
from dataclasses import replace

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    compact_kvp_pages,
    virtual_slots_to_local,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="The FlashMLA and GPU DSA leaves need an NVIDIA GPU.",
)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_query_gather_preserves_bytes(monkeypatch, dtype):
    from tokenspeed.runtime.layers.attention.dcp import comm

    query = torch.arange(256, dtype=torch.uint8).view(dtype).reshape(2, 8, 16)
    query = query.transpose(1, 2)

    def gather(payload, group, dim):
        assert payload.dtype == torch.uint8 and payload.is_contiguous()
        assert group == (0, 1) and dim == -1
        return torch.cat((payload, payload), dim=dim)

    monkeypatch.setattr(comm, "all_gather", gather)
    result = comm.gather_query_heads(query, (0, 1))
    assert result.dtype == dtype
    torch.testing.assert_close(
        result.view(torch.uint8),
        torch.cat((query.view(torch.uint8), query.view(torch.uint8)), dim=1),
        rtol=0,
        atol=0,
    )
    assert comm.gather_query_heads(query, (0,)) is query


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_noncontiguous_pages_and_partial_tail(degree, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Scheduler blocks 5, 2, 8 expand to two 64-token kernel pages each.
    table = torch.tensor(
        [[10, 11, 4, 5, 16, 17], [0, 0, 0, 0, 0, 0]], dtype=torch.int32, device=device
    )
    lengths = torch.tensor([273, 1], dtype=torch.int32, device=device)
    total = 0
    for rank in range(degree):
        out = torch.empty_like(table)
        local = torch.empty_like(lengths)
        compact_kvp_pages(
            table,
            lengths,
            page_size=64,
            block_granularity=128,
            virtual_block_count=1024,
            degree=degree,
            rank=rank,
            out=out,
            local_lengths=local,
        )
        expected = []
        n = 0
        for i, block in enumerate([5, 2, 8]):
            if (block - 1) % degree == rank:
                remaining = min(128, max(273 - i * 128, 0))
                for sub in range((remaining + 63) // 64):
                    expected.append(((block - 1) // degree + 1) * 2 + sub)
                n += remaining
        assert out[0, : len(expected)].tolist() == expected
        assert local.tolist() == [n, 0]
        assert not out[1].any()
        total += n
    assert total == 273


@pytest.mark.parametrize(
    "override, message",
    [
        ({"device": "cpu"}, "requires CUDA"),
    ],
)
def test_flashmla_dcp_rejects_unsupported_execution(override, message):
    from test.runtime.conftest import kimi_recipe

    config = kimi_recipe(tp_size=8).attn_config
    config = replace(
        config,
        device="cuda",
        components=(
            replace(config.components[0], backend_name="flashmla"),
            *config.components[1:],
        ),
        pd_disaggregation_enabled=False,
        kvp_size=4,
        kvp_rank=0,
        kvp_group=(0, 1, 2, 3),
    )
    for draft in (False, True):
        replace(
            config,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            is_draft=draft,
            pd_disaggregation_enabled=True,
        )
    with pytest.raises(ValueError, match=message):
        replace(config, **override)


def test_flashmla_absorbed_extend_refuses_sharded_cache():
    # The prefill wrapper plans on the virtual page table, so attending the
    # local shard through it would silently mix wrong pages; the backend must
    # refuse instead of relying on every model to route around it.
    flashmla = pytest.importorskip(
        "tokenspeed.runtime.layers.attention.backends.paged.flashmla"
    )
    leaf = flashmla.FlashMLABackend.__new__(flashmla.FlashMLABackend)
    leaf.kvp_group = (0, 1)
    with pytest.raises(RuntimeError, match="absorbed extend"):
        leaf._forward_absorbed_extend(torch.empty(0), layer=None, token_to_kv_pool=None)


def test_kimi_capacity_shards_only_mla():
    from test.runtime.conftest import kimi_recipe

    recipes = []
    for degree in [1, 4]:
        recipe = kimi_recipe(tp_size=8, max_bs=1)
        recipe.server_args.max_total_tokens = None
        recipe.cache_budget_bytes = 1 << 30
        components = recipe.attn_config.components
        recipe.attn_config = replace(
            recipe.attn_config,
            device="cuda",
            components=(
                replace(
                    components[0],
                    backend_name="flashmla",
                ),
                *components[1:],
            ),
            pd_disaggregation_enabled=False,
            kvp_size=degree,
            kvp_rank=0,
            kvp_group=tuple(range(degree)),
        )
        recipes.append(recipe)
    base, sharded = [r.setup().spec for r in recipes]
    assert {s.group_id: s.shard_count for s in sharded.cache_group_specs} == {
        s.group_id: (4 if s.group_id == "full_attention" else 1)
        for s in base.cache_group_specs
    }
    assert sharded.memory_plan.arena_bytes == base.memory_plan.arena_bytes
    assert sharded.token_capacity > base.token_capacity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("query_width,causal", [(1, True), (6, True), (6, False)])
def test_flashmla_shards_merge_to_full_attention(degree, query_width, causal):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashMLA dense decode requires SM90")
    from tokenspeed_kernel.ops.attention.mla.cuda import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    torch.manual_seed(17)
    device = "cuda"
    width, heads, length = 576, 32, 258
    source = torch.randn(12 * 128, 1, width, device=device, dtype=torch.bfloat16) * 0.1
    table = torch.tensor([[10, 11, 4, 5, 16, 17]], dtype=torch.int32, device=device)
    lengths = torch.tensor([length], dtype=torch.int32, device=device)
    q = torch.randn(1, query_width, heads, width, device=device, dtype=torch.bfloat16)
    q += 0.25
    source[5 * 128 : 6 * 128] += 0.4
    source[2 * 128 : 3 * 128] -= 0.3
    source[8 * 128 : 9 * 128] += 0.1

    def attention(query, cache, pages, lens):
        return flash_mla_with_kvcache(
            query,
            cache.view(-1, 64, 1, width),
            pages,
            lens,
            512,
            get_mla_metadata()[0],
            softmax_scale=width**-0.5,
            causal=causal,
        )

    ref, _ = attention(q, source, table, lengths)
    outputs, lses = [], []
    loc = torch.arange(source.shape[0], device=device)
    for rank in range(degree):
        local_cache = torch.zeros(
            ((12 - 1 + degree - 1) // degree + 1) * 128,
            1,
            width,
            device=device,
            dtype=source.dtype,
        )
        slots, owned = virtual_slots_to_local(
            loc, rows_per_page=128, virtual_block_count=12, degree=degree, rank=rank
        )
        set_mla_kv_buffer_triton(
            local_cache,
            slots,
            source[..., :512],
            source[..., 512:],
            sanitize=True,
            write_mask=owned,
        )
        assert not local_cache[:128].any()
        from tokenspeed.runtime.layers.attention.dcp.metadata import (
            CompactDCPLayout,
            refresh_dcp_page_table_metadata,
        )

        # The backend's layout: one global endpoint per verify query (causal)
        # or the block end for every query (non-causal block draft).
        offsets = (
            torch.arange(1 - query_width, 1, dtype=torch.int32, device=device)
            if causal
            else torch.zeros(query_width, dtype=torch.int32, device=device)
        )
        metadata = refresh_dcp_page_table_metadata(
            page_table=table,
            virtual_block_count=12,
            degree=degree,
            rank=rank,
            layout=CompactDCPLayout(lengths, 64, 128, lengths[:, None] + offsets),
            previous=None,
        )
        local = metadata.local_visible_lens.reshape(-1)
        partial, lse = attention(
            q.reshape(query_width, 1, heads, width),
            local_cache,
            metadata.local_page_table.repeat_interleave(query_width, dim=0),
            local.clamp_min(1),
        )
        outputs.append(
            torch.where((local > 0)[:, None, None, None], partial.float(), 0)
        )
        lses.append(torch.where((local > 0)[:, None, None], lse, -torch.inf))
    lses = torch.stack(lses).transpose(-1, -2).unsqueeze(-1)
    merged = (torch.stack(outputs) * torch.softmax(lses, dim=0)).sum(0)
    torch.testing.assert_close(
        merged.reshape_as(ref), ref.float(), atol=0.002, rtol=0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_graph_replay_refreshes_lengths_and_owners():
    table = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([1], dtype=torch.int32, device="cuda")
    out, local = torch.empty_like(table), torch.empty_like(lengths)

    def refresh():
        compact_kvp_pages(
            table,
            lengths,
            page_size=64,
            block_granularity=128,
            virtual_block_count=12,
            degree=2,
            rank=1,
            out=out,
            local_lengths=local,
        )

    refresh()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refresh()
    graph.replay()
    assert local.item() == 0
    lengths.fill_(145)
    graph.replay()
    assert local.item() == 17
    assert out[0, 0].item() == 2
    table.copy_(torch.tensor([[4, 5, 2, 3]], dtype=torch.int32, device="cuda"))
    graph.replay()
    assert local.item() == 128
    assert out[0, :2].tolist() == [2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_no_sink_combine_preserves_contiguous_mla_output(monkeypatch, batch, dtype):
    from tokenspeed.runtime.layers.attention.dcp import comm

    output = torch.randn(batch, 8, 512, dtype=dtype, device="cuda")
    lse = torch.zeros(batch, 8, dtype=torch.float32, device="cuda")
    # Identical shards let the expected head-owner slice be computed exactly.
    monkeypatch.setattr(
        comm, "all_gather", lambda x, group, dim: torch.cat([x] * len(group), dim=dim)
    )
    monkeypatch.setattr(
        comm, "reduce_scatter", lambda x, group: (x * len(group))[:4].contiguous()
    )

    def exchange(receive, send, group):
        head_owner = send.reshape(2, batch, 4, -1)[0]
        receive.view(2, batch, 4, -1).copy_(
            head_owner.unsqueeze(0).expand(2, -1, -1, -1)
        )

    monkeypatch.setattr(comm, "all_to_all_single", exchange)
    merged = comm.combine_attention_partials(
        output, lse, group=(0, 1), rank=0, sink=None, keep_all_heads=False
    )
    assert merged.is_contiguous()
    torch.testing.assert_close(merged, output[:, :4])
    assert merged.view(batch, -1).shape == (batch, 4 * 512)


def test_hybrid_forwards_runtime_geometry_to_both_children():
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )

    calls = []
    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    backend.full_attn_backend = SimpleNamespace(
        configure_runtime=lambda **kwargs: calls.append(("mla", kwargs))
    )
    backend.linear_attn_backend = SimpleNamespace(
        configure_runtime=lambda **kwargs: calls.append(("kda", kwargs))
    )
    specs, counts = object(), object()
    backend.configure_runtime(cache_group_specs=specs, cache_group_page_counts=counts)
    assert [name for name, _ in calls] == ["mla", "kda"]
    for _, kwargs in calls:
        assert kwargs == {"cache_group_specs": specs, "cache_group_page_counts": counts}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("model_write", [False, True])
def test_physical_mla_writer_with_placement_and_explicit_history_gather(
    monkeypatch, hybrid, model_write
):
    from test.runtime.cache_pool_test_utils import (
        make_arena,
        make_mla_memory_plan,
        plan_group_specs,
    )
    from types import SimpleNamespace

    from tokenspeed.runtime.distributed.mapping import KVP_HISTORY_GROUP_NAME
    from tokenspeed.runtime.layers.attention.backends.paged import flashmla
    from tokenspeed.runtime.layers.attention.dcp import cache as dcp_cache
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        plan_prefix_history_gather,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots
    from tokenspeed.runtime.layers.attention.kv_cache import mla
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDATokenToKVPool,
    )
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.utils.env import global_server_args_dict

    plan = make_mla_memory_plan(
        size=8,
        prefix_granularity=4,
        layer_num=1,
        latent_width=576,
        dtype=torch.bfloat16,
    )
    arena = make_arena(
        plan,
        "cuda",
        cache_group_specs=tuple(
            replace(spec, shard_count=2) for spec in plan_group_specs(plan)
        ),
    )
    pool_cls = HybridKDATokenToKVPool if hybrid else mla.MLATokenToKVPool
    pool = pool_cls(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method=None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=1,
        rank=0,
        **({"layer_types": ("full_attention",)} if hybrid else {}),
    )
    backend = object.__new__(flashmla.FlashMLABackend)
    backend.cache_pool = pool
    backend.kvp_group = (0, 1)
    backend.kvp_rank = 0
    backend.kvp_block_granularity = 4
    backend.kvp_virtual_block_count = arena.runtime_contract.virtual_block_counts[
        "full_attention"
    ]
    backend.kv_lora_rank = 512
    layer = PagedAttention(
        1,
        192,
        1.0,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=128,
        rotary_emb=None,
        qk_norm=None,
    )
    loc = torch.tensor([4, 8, 12], device="cuda")
    values = torch.arange(3 * 576, device="cuda", dtype=torch.bfloat16).reshape(
        3, 1, 576
    )
    cache = pool.get_key_buffer(0)
    cache.zero_()
    if model_write:
        from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA

        model = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            qk_head_dim=192,
            num_local_heads=1,
            v_head_dim=128,
            kv_b_proj=lambda latent: (latent.new_zeros((latent.shape[0], 256)),),
            attn_mha=layer,
            rotary_emb=None,
        )
        DeepseekV3AttentionMLA.forward_normal_chunked_kv_prepare(
            model,
            torch.arange(3, device="cuda"),
            values.new_zeros((3, 192)),
            values.squeeze(1),
            SimpleNamespace(
                attn_backend=backend, token_to_kv_pool=pool, query_shard=None
            ),
            loc,
        )
    else:
        slots, mask = resolve_cache_slots(loc, backend.cache_placement(layer))
        pool.set_mla_kv_buffer(
            layer, slots, values[..., :512], values[..., 512:], write_mask=mask
        )
    torch.testing.assert_close(cache[4], values[0])
    torch.testing.assert_close(cache[8], values[2])
    assert not cache[:4].any()
    assert backend.cache_placement(layer) is not None
    # The pool reads physical slots directly, with no distributed state.
    local_nope, local_rope = pool.get_mla_kv_buffer(layer, loc[:2], torch.float32)
    torch.testing.assert_close(
        torch.cat((local_nope, local_rope), dim=-1), values[[0, 2]].float()
    )
    assert "kvp_group" not in vars(pool)
    assert "kvp_rank" not in vars(pool)

    # One request whose three rows sit on pages of virtual blocks 1, 2, 3:
    # rank 0 owns rows 0 and 2, rank 1 owns row 1.
    (window,) = plan_prefix_history_gather(
        [loc],
        page_table_cpu=torch.tensor([[4, 8, 12]], dtype=torch.int32),
        prefix_lens_cpu=torch.tensor([3]),
        chunk_len=3,
        page_size=1,
        placement=backend.cache_placement(layer),
    )
    assert window.owned_rows_per_rank == (2, 1)
    assert window.send_rows == 2
    assert window.gathered_index.tolist() == [0, 2, 1]
    assert window.fetch_slots.tolist() == [4, 8]
    assert window.loop_bounds == (0, 3)

    class OwnerRows:
        """Rank 0's view of the all-gather: rank 1 owns row 1."""

        def all_gather_single_async(self, gathered, send, group):
            assert group == (0, 1)
            row_bytes = values.view(torch.uint8).view(3, 1, -1)
            torch.testing.assert_close(send, row_bytes[[0, 2]], rtol=0, atol=0)
            gathered[:2] = send
            gathered[2] = row_bytes[1]
            # Rank 1's padding row is never read.
            gathered[3] = 0xFF
            return SimpleNamespace(wait=lambda: None)

    # The leaf's prefill metadata prepares the prefix it reads back; the
    # gathers run on the named KVP history groups.
    monkeypatch.setattr(
        dcp_cache,
        "get_global_backend",
        lambda: SimpleNamespace(
            for_named_groups={KVP_HISTORY_GROUP_NAME: OwnerRows()}.__getitem__
        ),
    )
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 3)
    backend.kernel_page_size = 1
    backend._prefix_history = None
    loops = [loc]
    backend._prepare_prefix_history(
        loops,
        page_table_cpu=torch.tensor([[4, 8, 12]], dtype=torch.int32),
        prefix_lens_cpu=torch.tensor([3]),
    )
    ((nope, rope),) = backend.read_prefix_history(
        pool, layer, loops, dst_dtype=torch.float32
    )
    torch.testing.assert_close(
        torch.cat((nope, rope), dim=-1), values.float(), rtol=0, atol=0
    )
    # A sharded leaf never falls back to reading its own rows alone.
    backend._prefix_history = None
    with pytest.raises(NotImplementedError, match="prepared no prefix read"):
        backend.read_prefix_history(pool, layer, loops, dst_dtype=torch.float32)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots, mask = resolve_cache_slots(loc, backend.cache_placement(layer))
        pool.set_mla_kv_buffer(
            layer, slots, values[..., :512], values[..., 512:], write_mask=mask
        )
    cache.zero_()
    loc.copy_(torch.tensor([8, 12, 4], device="cuda"))
    graph.replay()
    torch.testing.assert_close(cache[4], values[2])
    torch.testing.assert_close(cache[8], values[1])
    assert not cache[:4].any()


@pytest.mark.parametrize("composite", ["hybrid", "router"])
def test_cache_placement_routes_by_layer(composite):
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    placement = CachePlacement(
        block_granularity=64, virtual_block_count=16, group=(0, 1), rank=0
    )
    layer = SimpleNamespace(layer_id=7)

    def select(value):
        assert value == (7 if composite == "hybrid" else layer)
        return SimpleNamespace(
            cache_placement=lambda selected: placement if selected is layer else None
        )

    node = SimpleNamespace(_backend_for_layer=select, _leaf_for=select)
    cls = HybridLinearAttnBackend if composite == "hybrid" else CacheGroupRouter
    assert cls.cache_placement(node, layer) is placement


@pytest.mark.parametrize("kind", ["mla", "dsa", "trtllm_mla", "flashmla"])
def test_unsharded_backends_preserve_physical_slots(kind):
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
    from tokenspeed.runtime.layers.attention.backends.paged.flashmla import (
        FlashMLABackend,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
        TRTLLMMLABackend,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots

    cls = {
        "mla": MLAAttnBackend,
        "dsa": DSABackend,
        "trtllm_mla": TRTLLMMLABackend,
        "flashmla": FlashMLABackend,
    }[kind]
    backend = object.__new__(cls)
    if kind in ("flashmla", "dsa"):
        backend.kvp_group = (0,)
    loc = torch.tensor([0, 3, 8])
    slots, mask = resolve_cache_slots(
        loc, backend.cache_placement(SimpleNamespace(layer_id=0))
    )
    assert slots is loc
    assert mask is None


def test_masked_quantized_mla_writer_rejects_before_mutation():
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    pool = SimpleNamespace(quant_method="per_token_head", latent_write_sanitizes=False)
    values = torch.ones(1, 1, 4)
    with pytest.raises(ValueError, match="do not support a mask"):
        MLATokenToKVPool.set_mla_kv_buffer(
            pool,
            SimpleNamespace(layer_id=0),
            torch.tensor([0]),
            values,
            values,
            write_mask=torch.tensor([False]),
        )


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_kimi_capacity_fits_physical_parents_with_state_reservations(degree):
    from test.runtime.conftest import kimi_recipe

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack

    recipe = kimi_recipe(tp_size=8, max_bs=4)
    recipe.server_args.max_total_tokens = None
    components = recipe.attn_config.components
    recipe.attn_config = replace(
        recipe.attn_config,
        device="cuda",
        components=(replace(components[0], backend_name="flashmla"), *components[1:]),
        pd_disaggregation_enabled=False,
        kvp_size=degree,
        kvp_rank=0,
        kvp_group=tuple(range(degree)),
    )
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    budget = 128
    capacity = recipe.token_capacity(layout, budget)
    assert recipe.parents_needed(layout, capacity) <= budget
    assert recipe.parents_needed(layout, capacity + 1) > budget
    # Cross-check with the unsharded topology at identical physical capacity:
    # only history capacity scales, while state working sets stay replicated.
    assert all(
        spec.shard_count == (degree if spec.group_id == "full_attention" else 1)
        for spec, _ in groups
    )


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("token_limit", [None, 64, 128, 576])
def test_ordinary_mla_kvp_capacity_and_token_limit(degree, token_limit):
    from test.runtime.test_cache_setup import _mla_config
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        prepare_cache_setup,
    )

    config = _mla_config()
    config = replace(
        config,
        device="cuda",
        components=(replace(config.components[0], backend_name="flashmla"),),
        kvp_size=degree,
        kvp_rank=0,
        kvp_group=tuple(range(degree)),
    )
    setup = prepare_cache_setup(
        family="mla",
        server_args=SimpleNamespace(
            max_total_tokens=token_limit, kv_offload_config=None
        ),
        model_config=SimpleNamespace(
            num_attention_layers=2, hf_config=SimpleNamespace(), model_profile=None
        ),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=24_576,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    assert setup.spec.memory_plan.arena_bytes <= 24_576
    assert setup.spec.token_capacity == (
        960 * degree if token_limit is None else token_limit
    )
    assert all(spec.shard_count == degree for spec in setup.spec.cache_group_specs)


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_pure_dsa_kvp_shards_index_and_latent_capacity(degree):
    from dataclasses import asdict
    from test.runtime.test_cache_setup import _mla_config
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        prepare_cache_setup,
    )

    base = _mla_config()
    fields = asdict(base.components[0])
    fields.update(backend_name="dsa")
    spec = DSAConfig(
        **fields,
        index_topk=2048,
        index_n_heads=16,
        index_head_dim=128,
        index_k_format="fp8_scaled",
    )
    config = replace(
        base,
        device="cuda",
        components=(spec,),
        kvp_size=degree,
        kvp_group=tuple(range(degree)),
        kvp_rank=0,
    )
    for draft in (False, True):
        replace(
            config,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            is_draft=draft,
            pd_disaggregation_enabled=True,
        )
    if degree > 1:
        with pytest.raises(ValueError, match="non-causal block drafts"):
            replace(config, is_draft=True, draft_block_decode=True)
    setup = prepare_cache_setup(
        family="dsa",
        server_args=SimpleNamespace(max_total_tokens=None, kv_offload_config=None),
        model_config=SimpleNamespace(
            num_attention_layers=2, hf_config=SimpleNamespace(), model_profile=None
        ),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1_048_576,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    assert all(group.shard_count == degree for group in setup.spec.cache_group_specs)
    assert (
        setup.spec.token_capacity == setup.spec.memory_plan.num_lcm_blocks * 64 * degree
    )


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("query_width", [1, 6])
def test_dsa_decode_partitions_candidates_and_merges_gathered_heads(
    monkeypatch, rank, query_width
):
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.backends.paged import dsa

    backend = object.__new__(dsa.DSABackend)
    backend.kernel_page_size = 64
    backend.kernel_solution = None
    backend.prefill_kernel_plan = None
    backend.decode_kernel_plan = None
    backend.slot_order = "selection"
    backend.data_type = torch.bfloat16
    backend.kv_lora_rank = 128
    backend.qk_nope_head_dim = 128
    backend.qk_rope_head_dim = 0
    backend.index_topk = 512
    backend.max_context_len = 512
    backend.kvp_group = (0, 1, 2, 3)
    backend.kvp_rank = rank
    backend.kvp_block_granularity = 64
    backend.kvp_virtual_block_count = 5
    backend.qcp_group = (0,)
    # The layer holds the attention-TP slice (2 of 8 heads): the sharded-head
    # combine form.
    backend.num_attention_heads = 8
    backend.num_local_heads = 2
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=0, seq_lens_k=torch.tensor([128]), max_seq_len_k=128
        )
    )
    query = torch.zeros(query_width, 2, 128, dtype=torch.bfloat16)
    slots = torch.full((query_width, 512), -1, dtype=torch.int32)
    slots[:, :4] = torch.tensor([64, 128, 192, 256])
    pool = SimpleNamespace(
        quant_method=None, get_key_buffer=lambda layer_id: torch.empty(320, 128)
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        head_dim=128,
        v_head_dim=128,
        scaling=0.1,
        logit_cap=0.0,
    )

    def gather(q, group):
        assert group == backend.kvp_group
        return q.repeat(1, 4, 1)

    def decode(**kwargs):
        assert kwargs["return_lse"] is True
        torch.testing.assert_close(
            kwargs["kv_seq_lens"],
            torch.arange(129 - query_width, 129, dtype=torch.int32),
        )
        expected = torch.full_like(slots, -1)
        expected[:, rank] = 64
        torch.testing.assert_close(kwargs["topk_slots"], expected)
        assert kwargs["q"].shape == (query_width, 8, 128)
        return torch.full((query_width, 8, 128), 7.0), torch.zeros(query_width, 8)

    def combine(out, lse, *, group, rank, sink, keep_all_heads):
        assert sink is None and group == backend.kvp_group
        assert keep_all_heads is False
        assert lse.shape == out.shape[:-1]
        return out[:, rank * 2 : (rank + 1) * 2]

    monkeypatch.setattr(dsa, "gather_query_heads", gather)
    monkeypatch.setattr(dsa, "dsa_decode", decode)
    monkeypatch.setattr(dsa, "combine_attention_partials", combine)
    out = backend.forward_sparse_decode(
        q=query,
        layer=layer,
        token_to_kv_pool=pool,
        bs=1,
        topk_indices=slots,
        topk_lens=None,
    )
    assert out.shape == (query_width, 256)
    assert (out == 7).all()


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize(
    "offset_dtype,score_dtype",
    [
        (torch.int32, torch.float32),
        (torch.int64, torch.float32),
        (torch.int32, torch.float64),
    ],
)
def test_global_index_candidate_merge_ties_and_empty_shards(
    monkeypatch, degree, tied, offset_dtype, score_dtype
):
    from tokenspeed.runtime.layers.attention.dcp import indexer

    offsets = torch.tensor([[9, 3, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    scores = torch.tensor(
        [[2.0, float("inf"), -float("inf"), float("nan")], [-float("inf")] * 4]
    )
    all_offsets = [offsets] + [
        torch.tensor([[r * 10, r * 10 + 1, -1, -1], [-1] * 4], dtype=torch.int32)
        for r in range(1, degree)
    ]
    all_scores = [scores] + [
        torch.tensor(
            [
                [
                    2.0 if tied else r * 2.0 + 1,
                    2.0 if tied else r * 2.0 + 2,
                    -float("inf"),
                    -float("inf"),
                ],
                [-float("inf")] * 4,
            ]
        )
        for r in range(1, degree)
    ]

    all_offsets = [part.to(offset_dtype) for part in all_offsets]
    all_scores = [part.to(score_dtype) for part in all_scores]
    offsets, scores = all_offsets[0], all_scores[0]

    def gather(x, group, dim):
        if offset_dtype != torch.int32 or score_dtype != torch.float32:
            return torch.cat(
                all_offsets if x.dtype == offset_dtype else all_scores, dim=dim
            )
        return torch.cat(
            [
                torch.stack((ids, values.view(torch.int32)), dim=-1).flatten(1)
                for ids, values in zip(all_offsets, all_scores)
            ],
            dim=dim,
        )

    monkeypatch.setattr(indexer, "all_gather", gather)
    selected, counts = indexer.merge_index_candidates(
        offsets, scores, topk=4, group=tuple(range(degree))
    )
    chosen = selected[0][selected[0] >= 0].tolist()
    assert selected[1].tolist() == [-1] * 4
    assert selected[0, 0].item() == 3  # Mandatory candidate always survives.
    assert len(set(chosen)) == len(chosen)
    candidates = {9} | {r * 10 + i for r in range(1, degree) for i in range(2)}
    if tied:
        assert set(chosen[1:]) <= candidates
    else:
        expected = [3] + sorted(candidates, reverse=True)[:3]
        assert chosen == expected
    assert (selected[0, len(chosen) :] == -1).all()
    assert counts.tolist() == [min(4, degree * 2), 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dsa_indexer_16_heads_with_padded_quantization_scales():
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter

    from tokenspeed.runtime.layers.attention.dcp.indexer import select_dsa_topk
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    torch.manual_seed(317)
    q = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    values, scales = quantize_fp8(
        keys, granularity="token_group", group_size=128, scale_encoding="float32"
    )
    cache = torch.zeros(192, 132, device="cuda", dtype=torch.uint8)
    index_k_block_split_scatter(
        cache,
        values,
        scales,
        torch.arange(192, device="cuda"),
        page_size=64,
        head_dim=128,
        group_size=128,
        write_mask=None,
    )
    indices, counts = select_dsa_topk(
        q,
        weights,
        cache,
        torch.tensor([[2, 1]], device="cuda", dtype=torch.int32),
        torch.zeros(3, device="cuda", dtype=torch.int32),
        torch.tensor([0, 17, 127], device="cuda", dtype=torch.int32),
        placement=CachePlacement(64, 3, (0,), 0),
        page_size=64,
        topk=512,
        softmax_scale=0.1,
        initial_tokens=4,
        local_tokens=8,
        max_logits_bytes=4096,
    )
    assert counts.tolist() == [0, 17, 127]
    assert (indices[0] == -1).all()
    assert set(indices[1, :17].tolist()) == set(range(17))
    assert set(indices[2, :127].tolist()) == set(range(127))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flashmla_draft_refreshes_dcp_after_advance_and_rollback(monkeypatch):
    """The per-query DCP rows are refreshed in place: a captured seq_lens edit
    (drafter hook) and a captured full refresh both replay against the same
    buffers, so the kernel inputs follow advances and rollbacks."""
    from tokenspeed.runtime.layers.attention.backends.paged.flashmla import (
        FlashMLABackend,
    )

    backend = object.__new__(FlashMLABackend)
    backend.kvp_group = (0, 1, 2, 3)
    backend.kvp_rank = 0
    backend.kvp_block_granularity = 128
    backend.kvp_virtual_block_count = 10
    backend.kernel_page_size = 64
    backend.is_draft = False
    backend.draft_block_decode = False
    backend.spec_num_tokens = 3
    backend._decode_views_by_bs = {}
    # Blocks 2, 5 (pages 4-5, 10-11); rank 0 owns block 5 only, which is its
    # local block 2 (pages 4-5).
    backend.page_table_buf = torch.tensor(
        [[4, 5, 10, 11]], device="cuda", dtype=torch.int32
    )
    backend.seq_lens_buf = torch.tensor([127], device="cuda", dtype=torch.int32)
    backend._init_dcp_state()
    monkeypatch.setattr(
        FlashMLABackend, "_renew_decode_tile_metadata", lambda self, *, for_graph: None
    )
    lens = backend.seq_lens_buf.clone()
    table = backend.page_table_buf.clone()
    # A drafter hook edits a published refresh; eager refresh lays the rows out.
    backend.refresh_decode_metadata(1, 1, lens, table)
    advance = torch.cuda.CUDAGraph()
    with torch.cuda.graph(advance):
        backend.advance_draft_forward_metadata(lens)
    refresh = torch.cuda.CUDAGraph()
    with torch.cuda.graph(refresh):
        backend.refresh_decode_metadata(1, 1, lens, table, for_graph_replay=True)
    metadata = backend.dcp_metadata
    cases = ((129, [0, 0, 1]), (132, [2, 3, 4]), (128, [0, 0, 0]))
    for graph in (advance, refresh):
        for length, expected in cases:
            lens.fill_(length)
            graph.replay()
            assert metadata.local_visible_lens.tolist() == [expected]
            assert metadata.local_seq_lens.item() == expected[-1]
            assert backend._dcp_kernel_seq_lens.tolist() == [
                max(value, 1) for value in expected
            ]
            assert backend._dcp_page_table_rows.tolist() == [[4, 5, 0, 0]] * 3
    # A narrower table on refresh clears the stale columns (null page 0) so a
    # later advance cannot resolve into them.
    lens.fill_(132)
    backend.refresh_decode_metadata(1, 1, lens, table[:, :2])
    assert backend.page_table_buf.tolist() == [[4, 5, 0, 0]]
    assert metadata.local_visible_lens.tolist() == [[0, 0, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("degree", [2, 4])
def test_flashmla_backend_kvp_verify_rows_merge_to_dense_verify(monkeypatch, degree):
    """``_run_flash_mla_decode`` under KVP: the per-step rows a refresh laid
    out (one kernel row per verify query, local causal lengths, repeated
    page table) merge across the owners to the dense verify, with the
    decode request behind a mixed round's extend row."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashMLA dense decode requires SM90")
    from types import SimpleNamespace

    from tokenspeed_kernel.ops.attention.mla.cuda import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    from tokenspeed.runtime.layers.attention.backends.paged import flashmla

    torch.manual_seed(5)
    device = "cuda"
    width, heads, length, queries = 576, 16, 258, 6
    source = torch.randn(12 * 128, 1, width, device=device, dtype=torch.bfloat16) * 0.1
    source[5 * 128 : 6 * 128] += 0.4
    source[2 * 128 : 3 * 128] -= 0.3
    # Row 0 is the round's extend request (never read here), row 1 decodes.
    table = torch.tensor(
        [[2, 3, 0, 0, 0, 0], [10, 11, 4, 5, 16, 17]], dtype=torch.int32, device=device
    )
    lengths = torch.tensor([64, length], dtype=torch.int32, device=device)
    q = torch.randn(queries, heads, width, device=device, dtype=torch.bfloat16) + 0.25
    ref, _ = flash_mla_with_kvcache(
        q[None],
        source.view(-1, 64, 1, width),
        table[1:],
        lengths[1:],
        512,
        get_mla_metadata()[0],
        softmax_scale=width**-0.5,
        causal=True,
    )
    layer = SimpleNamespace(
        layer_id=0, tp_q_head_num=heads, head_dim=width, scaling=width**-0.5
    )
    loc = torch.arange(source.shape[0], device=device)
    partials = []
    for rank in range(degree):
        backend = object.__new__(flashmla.FlashMLABackend)
        backend.kvp_group = tuple(range(degree))
        backend.kvp_rank = rank
        backend.kvp_block_granularity = 128
        backend.kvp_virtual_block_count = 12
        backend.kernel_page_size = 64
        backend.is_draft = False
        backend.draft_block_decode = False
        backend.spec_num_tokens = queries
        backend.num_q_heads = heads
        backend.kv_lora_rank = 512
        backend.kv_cache_dim = width
        backend._decode_views_by_bs = {}
        backend._decode_tile_metadata = None
        backend._decode_tile_metadata_keepalive = []
        backend.page_table_buf = torch.zeros((4, 6), dtype=torch.int32, device=device)
        backend.seq_lens_buf = torch.zeros(4, dtype=torch.int32, device=device)
        backend._init_dcp_state()
        local_cache = torch.zeros(
            ((12 - 1 + degree - 1) // degree + 1) * 128,
            1,
            width,
            device=device,
            dtype=source.dtype,
        )
        slots, owned = virtual_slots_to_local(
            loc, rows_per_page=128, virtual_block_count=12, degree=degree, rank=rank
        )
        set_mla_kv_buffer_triton(
            local_cache,
            slots,
            source[..., :512],
            source[..., 512:],
            sanitize=True,
            write_mask=owned,
        )
        pool = SimpleNamespace(get_key_buffer=lambda layer_id, c=local_cache: c)
        recorded = []

        def combine(out, lse, *, group, rank, sink, keep_all_heads, rec=recorded):
            assert group == backend.kvp_group and sink is None and not keep_all_heads
            rec.append((out.float().clone(), lse.clone()))
            return out[:, :heads]

        monkeypatch.setattr(
            flashmla, "gather_query_heads", lambda q, g: q.repeat(1, len(g), 1)
        )
        monkeypatch.setattr(flashmla, "combine_attention_partials", combine)
        backend.refresh_decode_metadata(2, 2, lengths, table, num_extends=1)
        out, lse = backend._run_flash_mla_decode(
            q.reshape(-1, heads * width), layer, pool, 1
        )
        assert lse is None and out.shape == (queries, 1, heads, 512)
        (partial,) = recorded
        partials.append(partial)
    outputs = torch.stack([p[0] for p in partials])
    lses = torch.stack([p[1] for p in partials]).unsqueeze(-1)
    merged = (outputs * torch.softmax(lses, dim=0)).sum(0)
    torch.testing.assert_close(merged[:, :heads], ref[0].float(), atol=0.002, rtol=0.03)
