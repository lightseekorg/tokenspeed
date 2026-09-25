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

from dataclasses import replace

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    compact_dcp_pages,
    virtual_slots_to_local,
)


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
        compact_dcp_pages(
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
        ({"speculative_num_steps": 1}, "does not yet support speculation"),
        ({"speculative_num_draft_tokens": 2}, "does not yet support speculation"),
        ({"is_draft": True}, "does not yet support speculation"),
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
        dcp_size=4,
        dcp_rank=0,
        dcp_group=(0, 1, 2, 3),
    )
    with pytest.raises(ValueError, match=message):
        replace(config, **override)


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
            dcp_size=degree,
            dcp_rank=0,
            dcp_group=tuple(range(degree)),
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
def test_flashmla_shards_merge_to_full_attention(degree):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashMLA dense decode requires SM90")
    from tokenspeed_kernel.ops.attention.mla.cuda import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    torch.manual_seed(17)
    device = "cuda"
    width, heads, length = 576, 32, 273
    source = torch.randn(12 * 128, 1, width, device=device, dtype=torch.bfloat16) * 0.1
    table = torch.tensor([[10, 11, 4, 5, 16, 17]], dtype=torch.int32, device=device)
    lengths = torch.tensor([length], dtype=torch.int32, device=device)
    q = torch.randn(1, 1, heads, width, device=device, dtype=torch.bfloat16)
    q += 0.25
    source[5 * 128 : 6 * 128] += 0.4
    source[2 * 128 : 3 * 128] -= 0.3
    source[8 * 128 : 9 * 128] += 0.1

    def attention(cache, pages, lens):
        return flash_mla_with_kvcache(
            q,
            cache.view(-1, 64, 1, width),
            pages,
            lens,
            512,
            get_mla_metadata()[0],
            softmax_scale=width**-0.5,
            causal=True,
        )

    ref, _ = attention(source, table, lengths)
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
        out, local = torch.empty_like(table), torch.empty_like(lengths)
        compact_dcp_pages(
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
        partial, lse = attention(local_cache, out, local.clamp_min(1))
        outputs.append(
            torch.where((local > 0)[:, None, None, None], partial.float(), 0)
        )
        lses.append(torch.where((local > 0)[:, None, None], lse, -torch.inf))
    lses = torch.stack(lses).transpose(-1, -2).unsqueeze(-1)
    merged = (torch.stack(outputs) * torch.softmax(lses, dim=0)).sum(0)
    torch.testing.assert_close(merged, ref.float(), atol=0.002, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_graph_replay_refreshes_lengths_and_owners():
    table = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([1], dtype=torch.int32, device="cuda")
    out, local = torch.empty_like(table), torch.empty_like(lengths)

    def refresh():
        compact_dcp_pages(
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
def test_no_sink_combine_preserves_contiguous_mla_output(monkeypatch, batch):
    from tokenspeed.runtime.layers.attention.dcp import comm

    output = torch.randn(batch, 8, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(batch, 8, dtype=torch.float32, device="cuda")
    # Identical shards let the expected head-owner slice be computed exactly.
    monkeypatch.setattr(
        comm, "all_gather", lambda x, group, dim: torch.cat([x] * len(group), dim=dim)
    )
    monkeypatch.setattr(
        comm, "reduce_scatter", lambda x, group: (x * len(group))[:4].contiguous()
    )
    merged = comm.combine_attention_partials(
        output, lse, group=(0, 1), rank=0, sink=None
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
    from types import SimpleNamespace

    from cache_pool_test_utils import make_arena, make_mla_memory_plan, plan_group_specs

    from tokenspeed.runtime.layers.attention.backends.paged import flashmla
    from tokenspeed.runtime.layers.attention.dcp.cache import gather_mla_history
    from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots
    from tokenspeed.runtime.layers.attention.kv_cache import mla
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDATokenToKVPool,
    )

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
    backend.dcp_group = (0, 1)
    backend.dcp_rank = 0
    backend.dcp_block_granularity = 4
    backend.dcp_virtual_block_count = arena.runtime_contract.virtual_block_counts[
        "full_attention"
    ]
    backend.kv_lora_rank = 512
    layer = SimpleNamespace(layer_id=0)
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
            _mla_kv_is_fp8=lambda ctx, scale: False,
        )
        DeepseekV3AttentionMLA.forward_normal_chunked_kv_prepare(
            model,
            torch.arange(3, device="cuda"),
            values.new_zeros((3, 192)),
            values.squeeze(1),
            SimpleNamespace(attn_backend=backend, token_to_kv_pool=pool),
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
    assert "dcp_group" not in vars(pool)
    assert "dcp_rank" not in vars(pool)

    def gather_owner_rows(local, group):
        assert group == (0, 1)
        torch.testing.assert_close(local[0], values[0].float())
        assert not local[1].any()
        torch.testing.assert_close(local[2], values[2].float())
        return values.float()

    from tokenspeed.runtime.layers.attention.dcp import comm

    monkeypatch.setattr(comm, "all_reduce", gather_owner_rows)
    nope, rope = gather_mla_history(
        pool,
        layer,
        loc,
        dst_dtype=torch.float32,
        placement=backend.cache_placement(layer),
    )
    torch.testing.assert_close(torch.cat((nope, rope), dim=-1), values.float())

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
        backend.dcp_group = (0,)
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
        dcp_size=degree,
        dcp_rank=0,
        dcp_group=tuple(range(degree)),
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


@pytest.mark.parametrize("shape", [(3, 4), (3, 1, 8)])
@pytest.mark.parametrize("group", [(0,), (0, 1)])
def test_gather_owned_rows_masks_nan_and_preserves_order(monkeypatch, shape, group):
    from tokenspeed.runtime.layers.attention.dcp import comm

    values = torch.arange(
        torch.tensor(shape).prod().item(), dtype=torch.float32
    ).reshape(shape)
    values[1] = float("nan")
    owned = torch.tensor([True, False, True])
    expected = values.clone()
    expected[1] = 0

    def reduce(rows, actual_group):
        assert actual_group == group
        torch.testing.assert_close(rows, expected)
        rows[1] = 42
        return rows

    monkeypatch.setattr(comm, "all_reduce", reduce)
    result = comm.gather_owned_rows(values, owned, group)
    if len(group) > 1:
        expected[1] = 42
    torch.testing.assert_close(result, expected)
    assert result.is_contiguous()


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("token_limit", [None, 64, 128, 576])
def test_ordinary_mla_dcp_capacity_and_token_limit(degree, token_limit):
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
        dcp_size=degree,
        dcp_rank=0,
        dcp_group=tuple(range(degree)),
    )
    setup = prepare_cache_setup(
        family="mla",
        server_args=SimpleNamespace(max_total_tokens=token_limit),
        model_config=SimpleNamespace(
            num_attention_layers=2, hf_config=SimpleNamespace()
        ),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=24_576,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    assert setup.spec.memory_plan.arena_bytes <= 24_576
    assert setup.spec.token_capacity == (
        960 * degree if token_limit is None else token_limit
    )
    assert all(spec.shard_count == degree for spec in setup.spec.cache_group_specs)


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_pure_dsa_dcp_shards_index_and_latent_capacity(degree):
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
    spec = DSAConfig(**fields, index_topk=2048, index_n_heads=16, index_head_dim=128)
    config = replace(
        base,
        device="cuda",
        components=(spec,),
        dcp_size=degree,
        dcp_group=tuple(range(degree)),
        dcp_rank=0,
    )
    setup = prepare_cache_setup(
        family="dsa",
        server_args=SimpleNamespace(max_total_tokens=None),
        model_config=SimpleNamespace(
            num_attention_layers=2, hf_config=SimpleNamespace()
        ),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1_048_576,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    assert all(group.shard_count == degree for group in setup.spec.cache_group_specs)
    assert (
        setup.spec.token_capacity == setup.spec.memory_plan.num_lcm_blocks * 64 * degree
    )


@pytest.mark.parametrize("rank", range(4))
def test_dsa_decode_partitions_candidates_and_merges_gathered_heads(monkeypatch, rank):
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.backends.paged import dsa

    backend = object.__new__(dsa.DSABackend)
    backend.kernel_page_size = 64
    backend.data_type = torch.bfloat16
    backend.kv_lora_rank = 128
    backend.qk_nope_head_dim = 128
    backend.qk_rope_head_dim = 0
    backend.index_topk = 512
    backend.max_context_len = 512
    backend.dcp_group = (0, 1, 2, 3)
    backend.dcp_rank = rank
    backend.dcp_block_granularity = 64
    backend.dcp_virtual_block_count = 5
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=0, seq_lens_k=torch.tensor([128]), max_seq_len_k=128
        )
    )
    query = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    slots = torch.full((1, 512), -1, dtype=torch.int32)
    slots[0, :4] = torch.tensor([64, 128, 192, 256])
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
        assert group == backend.dcp_group
        return q.repeat(1, 4, 1)

    def decode(**kwargs):
        assert kwargs["return_lse"] is True
        expected = torch.full_like(slots, -1)
        expected[0, rank] = 64
        torch.testing.assert_close(kwargs["topk_slots"], expected)
        assert kwargs["q"].shape == (1, 8, 128)
        return torch.full((1, 8, 128), 7.0), torch.zeros(1, 8)

    def combine(out, lse, *, group, rank, sink):
        assert sink is None and group == backend.dcp_group
        assert lse.shape == out.shape[:-1]
        return out[:, rank * 2 : (rank + 1) * 2]

    monkeypatch.setattr(dsa, "gather_query_heads", gather)
    monkeypatch.setattr(dsa, "dsa_decode", decode)
    monkeypatch.setattr(dsa, "combine_attention_partials", combine)
    out = backend.forward_sparse_decode(
        q=query,
        k=None,
        v=None,
        layer=layer,
        out_cache_loc=torch.tensor([64]),
        token_to_kv_pool=pool,
        bs=1,
        save_kv_cache=False,
        topk_indices=slots,
        topk_lens=None,
    )
    assert out.shape == (1, 256)
    assert (out == 7).all()


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("tied", [False, True])
def test_global_index_candidate_merge_ties_and_empty_shards(monkeypatch, degree, tied):
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

    def gather(x, group, dim):
        return torch.cat(all_offsets if x.dtype == torch.int32 else all_scores, dim=dim)

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
    from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

    from tokenspeed.runtime.layers.attention.dcp.indexer import select_dsa_topk
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

    torch.manual_seed(317)
    q = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    values, scales = quantize_fp8_with_scale(
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
