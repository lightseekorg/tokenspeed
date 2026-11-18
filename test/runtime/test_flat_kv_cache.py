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

"""FlatKV atom addressing and ordinary padded-cache regression checks."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    FLAT_KV_PLANE,
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.paged_attention import PagedAttention


def test_legacy_flatkv_plane_keeps_native_padded_pages():
    group = CacheGroupSpec("history", "full_history", 4, 2, replayable=False)
    layout = pack(
        (
            (
                group,
                (
                    CacheFieldSpec(
                        "layer.0.global_kv",
                        "flatkv",
                        (4, 8),
                        "uint8",
                        exact_page_stride=False,
                    ),
                    CacheFieldSpec(
                        "layer.0.tail",
                        "flatkv",
                        (8,),
                        "uint8",
                        exact_page_stride=False,
                    ),
                ),
            ),
        ),
        prefix_granularity=8,
        cache_blocks_per_lcm_block={group.group_id: 1},
        alignment=1,
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(2), "cpu", cache_group_specs=(group,))
    field = arena.plan.field("layer.0.global_kv")
    assert field.page_stride_bytes == 40
    arena.buffer.fill_(238)
    expected = arena.buffer.clone()
    block = arena.plan.group(group.group_id).page_count - 1
    arena.field(field.field_id)[block, 2].fill_(37)
    offset = arena.plan.field_page_byte_offset(field.field_id, block) + 2 * 8
    expected[offset : offset + 8].fill_(37)
    torch.testing.assert_close(arena.buffer, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("parents", "kernel_page_size", "error"),
    [
        (262143, 256, None),
        (262144, 256, "kernel write slot 2147483903"),
        (262144, 1, "atom page id"),
    ],
)
def test_flat_kv_pool_validates_atom_capacity_without_allocating(
    parents, kernel_page_size, error
):
    history = CacheGroupSpec("history", "full_history", 256, 1, replayable=False)
    state = CacheGroupSpec("state", "state", None, None, replayable=False)
    fields = tuple(
        CacheFieldSpec(
            f"layer.0.{name}",
            FLAT_KV_PLANE,
            (256, 1, 64),
            "bfloat16",
            exact_page_stride=False,
        )
        for name in ("k", "v")
    )
    layout = pack(
        (
            (history, fields),
            (
                state,
                (
                    CacheFieldSpec(
                        "layer.1.ssm",
                        FLAT_KV_PLANE,
                        (1 << 20,),
                        "uint8",
                        exact_page_stride=False,
                    ),
                ),
            ),
        ),
        prefix_granularity=256,
        cache_blocks_per_lcm_block={"history": 1, "state": 1},
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(parents), "meta", cache_group_specs=(history, state))
    pool = MHATokenToKVPool(
        arena=arena,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=64,
        layer_num=1,
        rank=0,
    )
    if error is None:
        assert pool.atom_pages_per_block("history", kernel_page_size) == 32
    else:
        with pytest.raises(ValueError, match=error):
            pool.atom_pages_per_block("history", kernel_page_size)


@requires_cuda
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_atom_translation_gpu_and_graph(packed, graph):
    specs = (
        GroupTableSpec("full", 4, 4, 6, 4),
        GroupTableSpec("subpages", 4, 1, 13, 16),
    )
    storage = torch.tensor(
        [0, 1, 7, -1, 3, 2, 5, 5, 5, 2, 0, -1, 8, 1, 3, 4, 0, 9, 9, 9, 9],
        dtype=torch.int32,
        device="cuda",
    )
    sources = {"full": storage[:9].view(3, 3), "subpages": storage[9:].view(3, 4)}
    if not packed:
        sources = {gid: src.clone() for gid, src in sources.items()}
    stacks = GroupTableStacks(
        specs, max_bs=3, max_tokens_per_req=2, max_extend_tokens=8, device="cuda"
    )
    lengths = torch.tensor([5, 8, 1], dtype=torch.int32, device="cuda")

    def run():
        stacks.fill(3, 2, sources)
        stacks.compute_decode_locations(3, lengths, 2)

    run()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        sources["full"][0, 1] = 4
        sources["subpages"][1, 1] = 6
        stacks.tables.fill_(99)
        capture.replay()
    expected = torch.zeros_like(stacks.tables, device="cpu")
    for group_index, spec in enumerate(specs):
        raw = sources[spec.group_id].cpu().tolist()
        ratio = spec.block_granularity // spec.kernel_page_size
        for row in range(2):
            expanded = [
                block * spec.atom_pages_per_block + j if block > 0 else 0
                for block in raw[row]
                for j in range(ratio)
            ][: spec.max_num_pages]
            expected[group_index, row, : len(expanded)] = torch.tensor(expanded)
    torch.testing.assert_close(stacks.tables.cpu(), expected, atol=0, rtol=0)
    expected_locations = []
    for group_index, spec in enumerate(specs):
        locations = []
        for row, length in enumerate(lengths.cpu().tolist()):
            for position in range(length - 2, length):
                if position < 0:
                    locations.append(0)
                    continue
                page = expected[
                    group_index, row, position // spec.kernel_page_size
                ].item()
                locations.append(
                    page * spec.kernel_page_size + position % spec.kernel_page_size
                    if page > 0
                    else 0
                )
        expected_locations.append(locations)
    torch.testing.assert_close(
        stacks.decode_locs.cpu(),
        torch.tensor(expected_locations, dtype=torch.int32),
        atol=0,
        rtol=0,
    )


@requires_cuda
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("kernel_page_size", [64, 256])
def test_flat_kv_gpu_store_reaches_original_field(dtype, kernel_page_size, graph):
    grain, heads, dim = 256, 2, 64
    spec = CacheGroupSpec("full_attention", "full_history", grain, 1, replayable=False)
    storage_dtype = "bfloat16" if dtype == torch.bfloat16 else "uint8"
    fields = tuple(
        CacheFieldSpec(
            f"layer.{layer}.{name}",
            FLAT_KV_PLANE,
            (grain, heads, dim),
            storage_dtype,
            exact_page_stride=False,
        )
        for layer in range(2)
        for name in ("k", "v")
    )
    layout = pack(
        ((spec, fields),),
        prefix_granularity=grain,
        cache_blocks_per_lcm_block={spec.group_id: 1},
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(8), "cuda", cache_group_specs=(spec,))
    pool = MHATokenToKVPool(arena, dtype, heads, dim, 2, 0)
    multiplier = pool.atom_pages_per_block(spec.group_id, kernel_page_size)
    tables = GroupTableStacks(
        [
            GroupTableSpec(
                spec.group_id,
                grain,
                kernel_page_size,
                grain // kernel_page_size,
                multiplier,
            )
        ],
        max_bs=1,
        max_tokens_per_req=1,
        max_extend_tokens=0,
        device="cuda",
    )
    raw = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([8], dtype=torch.int32, device="cuda")
    locations = tables.decode_locations(spec.group_id, 1, 1)
    layers = [
        PagedAttention(
            8, dim, dim**-0.5, heads, layer_id, rotary_emb=None, qk_norm=None
        )
        for layer_id in range(2)
    ]
    keys = [
        torch.full((1, heads, dim), layer_id + 1, dtype=torch.bfloat16, device="cuda")
        for layer_id in range(2)
    ]
    values = [torch.full_like(key, layer_id + 5) for layer_id, key in enumerate(keys)]

    def run():
        tables.fill(1, 1, {spec.group_id: raw})
        tables.compute_decode_locations(1, lengths, 1)
        for layer, key, value in zip(layers, keys, values):
            pool.set_kv_buffer(layer, locations, key, value)

    run()
    block = 1
    increment = 0
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        block = arena.plan.group(spec.group_id).page_count - 1
        raw.fill_(block)
        increment = 2
        for data in (*keys, *values):
            data.add_(increment)
        capture.replay()

    for layer_id in range(2):
        for name, value in (
            ("k", layer_id + 1 + increment),
            ("v", layer_id + 5 + increment),
        ):
            field_id = f"layer.{layer_id}.{name}"
            field = arena.plan.field(field_id)
            row_bytes = field.payload_bytes // grain
            byte_offset = (
                arena.plan.field_page_byte_offset(field_id, block) + 7 * row_bytes
            )
            actual = (
                arena.buffer[byte_offset : byte_offset + row_bytes].view(dtype).float()
            )
            torch.testing.assert_close(
                actual, torch.full_like(actual, value), atol=0, rtol=0
            )


def test_flat_kv_rejects_each_misaligned_page_even_when_the_span_aligns():
    group = CacheGroupSpec("full_attention", "full_history", 4, 1, replayable=False)
    fields = (
        CacheFieldSpec(
            "layer.0.k", FLAT_KV_PLANE, (4, 1, 4), "bfloat16", exact_page_stride=False
        ),
        CacheFieldSpec(
            "layer.0.extra", FLAT_KV_PLANE, (2,), "bfloat16", exact_page_stride=False
        ),
    )
    layout = pack(
        ((group, fields),),
        prefix_granularity=4,
        cache_blocks_per_lcm_block={group.group_id: 1},
        alignment=1,
        max_padding_fraction=float("inf"),
    )
    # Three pages make the complete span row-aligned despite a half-row gap
    # between each page; checking only that span would miss the bad stride.
    with pytest.raises(ValueError, match="whole number of atom rows"):
        CacheArena(layout.bind(2), "cpu", cache_group_specs=(group,))


@pytest.mark.parametrize("target_packing", [4, 5])
@pytest.mark.parametrize("layer_kv_head_counts", [None, (2,)])
def test_ordinary_sink_router_keeps_raw_page_addresses(
    target_packing, layer_kv_head_counts
):
    specs = (
        CacheGroupSpec("target", "full_history", 128, 1, replayable=False),
        CacheGroupSpec(
            "draft_swa",
            "sliding_window",
            128,
            1,
            1024,
            sink_tokens=16,
            replayable=False,
        ),
    )
    layout = pack(
        (
            (
                specs[0],
                (
                    CacheFieldSpec(
                        "layer.0.latent_kv", "slot.0", (128, 1, 16), "bfloat16"
                    ),
                ),
            ),
            (
                specs[1],
                tuple(
                    CacheFieldSpec(
                        f"layer.1.{name}",
                        "slot.0",
                        (128, 2, 8),
                        "bfloat16",
                        exact_page_stride=False,
                    )
                    for name in ("k", "v")
                ),
            ),
        ),
        prefix_granularity=128,
        cache_blocks_per_lcm_block={"target": target_packing, "draft_swa": 2},
        max_padding_fraction=float("inf"),
    )
    arena = CacheArena(layout.bind(3), "cpu", cache_group_specs=specs)
    pool = MHATokenToKVPool(
        arena,
        torch.bfloat16,
        2,
        8,
        1,
        0,
        field_layer_offset=1,
        layer_kv_head_counts=layer_kv_head_counts,
    )
    component = MHAConfig(
        backend_name="triton",
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=8,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=512,
        max_bs=1,
        max_resident_requests=1,
        components=(component,),
        speculative_num_draft_tokens=1,
    )
    router = CacheGroupRouter(
        lambda gid, grain: MHAAttnBackend(config, component, kernel_page_size=grain),
        is_draft=False,
        spec_num_tokens=1,
        device="cpu",
        consumed_group_ids=("draft_swa",),
    )
    router.set_cache_pool(pool)
    router.init_cuda_graph_state(1)
    router.refresh_decode_metadata(
        1,
        1,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables={"draft_swa": torch.tensor([[1]], dtype=torch.int32)},
    )
    assert router.leaves["draft_swa"].page_table_buf[0, 0].item() == 1
    assert router.decode_window_locations().tolist() == [135]


def _qwen4_exp_recipe(*, flat, device, kv_dtype, state_layers, draft_num_kv_heads):
    from dataclasses import replace
    from test.runtime.test_cache_setup import _qwen4_exp_fp8_kv_recipe

    from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
    )

    recipe = _qwen4_exp_fp8_kv_recipe(
        layer_types=(LINEAR_ATTENTION,) * state_layers + (FULL_ATTENTION,) * 2,
        ple_layer_ids=[2],
        hidden_size=16,
        linear=LinearAttnConfig(
            num_k_heads=1,
            num_v_heads=2,
            head_k_dim=32,
            head_v_dim=32,
            conv_kernel_size=4,
            layer_ids=tuple(range(state_layers)),
            tp_size=1,
        ),
        num_kv_heads=1,
        head_dim=64,
        mamba_ssm_dtype="float32",
        qsa=True,
        draft_num_kv_heads=draft_num_kv_heads,
        verify_width=4,
        max_bs=2,
        pp_size=1,
        cache_budget_bytes=16 << 20,
        device=device,
    )
    recipe.server_args.qwen4_exp_flat_kv = flat
    recipe.attn_config = replace(recipe.attn_config, kv_cache_dtype=kv_dtype)
    recipe.draft_attn_config = replace(
        recipe.draft_attn_config, kv_cache_dtype=kv_dtype
    )
    return recipe


@pytest.mark.parametrize("draft_index_dim", [64, 128])
@pytest.mark.parametrize("kernel_page_size", [64, 256])
def test_flat_kv_capacity_fits_every_target_and_draft_history_view(
    draft_index_dim, kernel_page_size
):
    from dataclasses import replace
    from test.runtime.test_cache_setup import _qwen4_exp_fp8_kv_recipe

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig
    from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
    )

    layer_types = (
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        FULL_ATTENTION,
    ) * 12
    recipe = _qwen4_exp_fp8_kv_recipe(
        layer_types=layer_types,
        ple_layer_ids=[2],
        hidden_size=16,
        linear=LinearAttnConfig(
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            conv_kernel_size=4,
            layer_ids=tuple(
                i for i, kind in enumerate(layer_types) if kind == LINEAR_ATTENTION
            ),
            tp_size=2,
        ),
        num_kv_heads=4,
        head_dim=128,
        mamba_ssm_dtype="float32",
        qsa=True,
        draft_num_kv_heads=4,
        verify_width=4,
        max_bs=2,
        pp_size=1,
        cache_budget_bytes=150 << 30,
        device="meta",
    )
    recipe.server_args.qwen4_exp_flat_kv = True
    recipe.server_args.mapping = Mapping(rank=0, world_size=2, attn_tp_size=2)
    recipe.draft_model_config.hf_text_config.indexer_head_dim = draft_index_dim

    def tp2_config(config):
        softmax = config.component(SoftmaxAttnConfig)
        return replace(
            config,
            kv_cache_dtype=torch.bfloat16,
            components=tuple(
                (
                    replace(component, attn_tp_size=2)
                    if component is softmax
                    else component
                )
                for component in config.components
            ),
        )

    recipe.attn_config = tp2_config(recipe.attn_config)
    recipe.draft_attn_config = tp2_config(recipe.draft_attn_config)
    setup = recipe.setup()
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )

    def validate_views(plan):
        arena = CacheArena(plan, "meta", cache_group_specs=setup.spec.cache_group_specs)
        for offset, count, config in (
            (0, setup.num_target_layers, recipe.attn_config),
            (setup.num_target_layers, setup.num_draft_layers, recipe.draft_attn_config),
        ):
            pool = create_cache_pool(
                setup.spec.layer_view(first_layer=offset, num_layers=count),
                config,
                arena,
                num_layers=count,
                rank=0,
                field_layer_offset=offset,
            )
            for spec in setup.spec.cache_group_specs:
                if spec.attn_kind == "full_history":
                    pool.atom_pages_per_block(spec.group_id, kernel_page_size)

    # The CI TP2 geometry exceeded int32 after sizing to free GPU memory.
    # Meta views exercise every real pool's checks without allocating the arena.
    validate_views(setup.spec.memory_plan)
    with pytest.raises(ValueError, match="kernel write slot"):
        validate_views(layout.bind(setup.spec.memory_plan.num_lcm_blocks + 1))


def _qwen4_exp_pool(*, flat, device, kv_dtype, draft_num_kv_heads):
    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )

    recipe = _qwen4_exp_recipe(
        flat=flat,
        device=device,
        kv_dtype=kv_dtype,
        state_layers=7,
        draft_num_kv_heads=draft_num_kv_heads,
    )
    setup = recipe.setup()
    arena = create_cache_arena(setup.spec, device=device, enable_memory_saver=False)
    pools = tuple(
        create_cache_pool(
            setup.spec.layer_view(first_layer=offset, num_layers=count),
            config,
            arena,
            num_layers=count,
            rank=0,
            field_layer_offset=offset,
        )
        for offset, count, config in (
            (0, setup.num_target_layers, recipe.attn_config),
            (setup.num_target_layers, setup.num_draft_layers, recipe.draft_attn_config),
        )
    )
    return recipe, arena, pools


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_flat_kv_groups_gdn_layers_on_one_plane(kv_dtype):
    from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
        QWEN4_EXP_PLE_CACHE_GROUP,
        QWEN4_EXP_QSA_CACHE_GROUP,
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    )

    _, arena, (target, draft) = _qwen4_exp_pool(
        flat=True, device="cpu", kv_dtype=kv_dtype, draft_num_kv_heads=1
    )
    ordinary = _qwen4_exp_recipe(
        flat=False,
        device="cpu",
        kv_dtype=kv_dtype,
        state_layers=7,
        draft_num_kv_heads=1,
    ).setup()
    assert {g.group_id for g in arena.plan.groups} == {
        "full_attention",
        "linear_attention_0",
        QWEN4_EXP_PLE_CACHE_GROUP,
        QWEN4_EXP_QSA_CACHE_GROUP,
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    }
    assert tuple(p.plane_id for p in arena.plan.planes) == (FLAT_KV_PLANE,)
    assert {f.plane_id for f in arena.plan.fields} == {FLAT_KV_PLANE}
    assert {f.field_id: f.group_id for f in arena.plan.fields} == {
        field.field_id: (
            "linear_attention_0"
            if field.group_id.startswith("linear_attention_")
            else field.group_id
        )
        for field in ordinary.spec.memory_plan.fields
    }
    assert set(target.state_group_by_layer.values()) == {"linear_attention_0"}
    assert {
        field.group_id
        for field in ordinary.spec.memory_plan.fields
        if field.group_id.startswith("linear_attention_")
    } == {"linear_attention_0", "linear_attention_1"}
    for pool in (target, draft):
        assert pool.arena is arena
        assert pool.atom_pages_per_block("full_attention", 64) > 1
        assert pool.atom_pages_per_block(QWEN4_EXP_QSA_CACHE_GROUP, 256) > 1
        assert pool.atom_pages_per_block(QWEN4_EXP_QSA_RECENT_CACHE_GROUP, 64) == 1
    kv_group = arena.plan.group("full_attention")
    kv_payload = sum(
        f.payload_bytes for f in arena.plan.fields if f.group_id == "full_attention"
    )
    assert (
        kv_payload * kv_group.cache_blocks_per_lcm_block == arena.plan.lcm_block_bytes
    )
    for field in arena.plan.fields:
        if field.group_id in ("full_attention", QWEN4_EXP_QSA_CACHE_GROUP):
            assert field.page_stride_bytes % field.payload_bytes == 0
        elif field.group_id == "linear_attention_0":
            view = arena.field(field.field_id)
            assert view.stride(0) * view.element_size() == field.page_stride_bytes


@pytest.mark.parametrize(
    ("flags", "flat"), [([], True), (["--no-qwen4-exp-flat-kv"], False)]
)
def test_qwen4_exp_default_cache_layout_respects_cli(flags, flat):
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    recipe = _qwen4_exp_recipe(
        flat=flat,
        device="cpu",
        kv_dtype=torch.bfloat16,
        state_layers=7,
        draft_num_kv_heads=1,
    )
    recipe.server_args = prepare_server_args(["--model", "x", *flags])
    plan = recipe.setup().spec.memory_plan
    if flat:
        assert {plane.plane_id for plane in plan.planes} == {FLAT_KV_PLANE}
        assert {
            group.group_id
            for group in plan.groups
            if group.group_id.startswith("linear_attention_")
        } == {"linear_attention_0"}
    else:
        assert FLAT_KV_PLANE not in {plane.plane_id for plane in plan.planes}
        assert len(plan.planes) > 1


@requires_cuda
@pytest.mark.parametrize("graph", [False, True])
def test_shared_plane_qsa_compression_and_recent_write(graph):
    from tokenspeed_kernel.ops.attention.qsa.triton import (
        qwen4_exp_qsa_compress_and_store,
        qwen4_exp_qsa_recent_write,
    )

    from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
        QWEN4_EXP_QSA_CACHE_GROUP,
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
        qsa_compressed_field,
        qsa_raw_key_field,
        qsa_rope_position_field,
    )

    _, arena, (pool, _) = _qwen4_exp_pool(
        flat=True, device="cuda", kv_dtype=torch.bfloat16, draft_num_kv_heads=1
    )
    layers = (7, 8, 9)
    keys = torch.randn(4, 1, 128, device="cuda", dtype=torch.bfloat16)
    logical = torch.arange(4, device="cuda", dtype=torch.int64)
    requests = torch.zeros(4, device="cuda", dtype=torch.int32)
    positions = logical[:, None].expand(-1, 3).contiguous()
    weight = torch.ones(128, device="cuda")
    cos_sin = torch.cat((torch.ones(8, 64), torch.zeros(8, 64)), dim=-1).cuda()
    stacks = GroupTableStacks(
        [
            GroupTableSpec(gid, grain, grain, 1, pool.atom_pages_per_block(gid, grain))
            for gid, grain in (
                (QWEN4_EXP_QSA_CACHE_GROUP, 256),
                (QWEN4_EXP_QSA_RECENT_CACHE_GROUP, 64),
            )
        ],
        max_bs=1,
        max_tokens_per_req=4,
        max_extend_tokens=0,
        device="cuda",
    )
    raw_tables = {
        gid: torch.full(
            (1, 1),
            1 + index * arena.plan.group(gid).cache_blocks_per_lcm_block,
            device="cuda",
            dtype=torch.int32,
        )
        for index, gid in enumerate(
            (QWEN4_EXP_QSA_CACHE_GROUP, QWEN4_EXP_QSA_RECENT_CACHE_GROUP)
        )
    }
    qsa_locs = torch.empty(4, device="cuda", dtype=torch.int32)
    recent_locs = torch.empty_like(qsa_locs)

    def run():
        stacks.fill(1, 1, raw_tables)
        qsa_locs.copy_(stacks.table(QWEN4_EXP_QSA_CACHE_GROUP, 1)[0, 0] * 256 + logical)
        recent_locs.copy_(
            stacks.table(QWEN4_EXP_QSA_RECENT_CACHE_GROUP, 1)[0, 0] * 64 + logical
        )
        for layer in layers:
            raw = arena.field(qsa_raw_key_field(layer))
            compressed = arena.field(qsa_compressed_field(layer))
            rope = arena.field(qsa_rope_position_field(layer))
            qwen4_exp_qsa_compress_and_store(
                keys,
                logical,
                requests,
                recent_locs,
                raw,
                positions,
                rope,
                weight,
                1e-6,
                cos_sin,
                qsa_locs,
                compressed,
                64,
                4,
                256,
                stage_verify_buffers=None,
                stage_draft=False,
            )
            qwen4_exp_qsa_recent_write(
                keys,
                logical,
                requests,
                recent_locs,
                positions,
                raw,
                rope,
                64,
                4,
            )

    run()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        # Pick a different parent for each group, including its final block.
        for index, (gid, table) in enumerate(raw_tables.items()):
            group = arena.plan.group(gid)
            table.fill_(group.page_count - 1 - index * group.cache_blocks_per_lcm_block)
        keys.add_(1)
        capture.replay()
    # The same kernel writing a dense output isolates layout addressing from
    # PyTorch/Triton RMSNorm differences at BF16 rounding boundaries.
    expected = torch.empty((2, 64, 1, 128), device="cuda", dtype=keys.dtype)
    qwen4_exp_qsa_compress_and_store(
        keys,
        logical,
        requests,
        recent_locs,
        arena.field(qsa_raw_key_field(layers[0])),
        positions,
        arena.field(qsa_rope_position_field(layers[0])),
        weight,
        1e-6,
        cos_sin,
        256 + logical,
        expected,
        64,
        4,
        256,
        stage_verify_buffers=None,
        stage_draft=False,
    )
    for layer in layers:
        compressed_id = qsa_compressed_field(layer)
        block = raw_tables[QWEN4_EXP_QSA_CACHE_GROUP].item()
        offset = arena.plan.field_page_byte_offset(compressed_id, block)
        actual = arena.buffer[offset : offset + 256].view(torch.bfloat16).view(1, 128)
        torch.testing.assert_close(actual, expected[1, 0], atol=0, rtol=0)
        recent_block = raw_tables[QWEN4_EXP_QSA_RECENT_CACHE_GROUP].item()
        torch.testing.assert_close(
            arena.field(qsa_raw_key_field(layer))[recent_block], keys, atol=0, rtol=0
        )
        assert arena.field(qsa_rope_position_field(layer))[recent_block].tolist() == [
            0,
            0,
            0,
        ]


@requires_cuda
def test_shared_plane_copy_and_zero():
    import numpy as np

    _, arena, _ = _qwen4_exp_pool(
        flat=True, device="cuda", kv_dtype=torch.bfloat16, draft_num_kv_heads=1
    )
    plan = arena.plan
    arena.buffer.fill_(219)
    for index, group in enumerate(plan.groups):
        packing = group.cache_blocks_per_lcm_block
        # Each group owns different source/destination parents, like the scheduler.
        source = (2 * index) * packing + 1
        destination = (2 * index + 1) * packing + 1
        for field in (f for f in plan.fields if f.group_id == group.group_id):
            offset = plan.field_page_byte_offset(field.field_id, source)
            arena.buffer[offset : offset + field.payload_bytes].fill_(index + 1)
        arena.copy_blocks(
            {group.group_id: np.array([[source, destination]], dtype=np.int32)}
        )
        for field in (f for f in plan.fields if f.group_id == group.group_id):
            offset = plan.field_page_byte_offset(field.field_id, destination)
            assert torch.all(
                arena.buffer[offset : offset + field.payload_bytes] == index + 1
            )
        arena.zero_blocks({group.group_id: np.array([destination], dtype=np.int32)})
        for field in (f for f in plan.fields if f.group_id == group.group_id):
            offset = plan.field_page_byte_offset(field.field_id, destination)
            assert torch.all(arena.buffer[offset : offset + field.payload_bytes] == 0)
            source_offset = plan.field_page_byte_offset(field.field_id, source)
            assert torch.all(
                arena.buffer[source_offset : source_offset + field.payload_bytes]
                == index + 1
            )


@requires_cuda
@pytest.mark.parametrize("graph", [False, True])
def test_shared_plane_ple_and_qsa_verify_commit(graph):
    from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
        Qwen4ExpPLEBackend,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
        QWEN4_EXP_PLE_CACHE_GROUP,
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
        qsa_raw_key_field,
        qwen4_exp_ple_context_field,
        qwen4_exp_ple_conv_field,
    )
    from tokenspeed.runtime.layers.attention.qsa.verify_state import QSAVerifyState

    recipe, arena, (pool, _) = _qwen4_exp_pool(
        flat=True, device="cuda", kv_dtype=torch.bfloat16, draft_num_kv_heads=1
    )
    ple = Qwen4ExpPLEBackend(
        recipe.attn_config, recipe.attn_config.component(MHAConfig)
    )
    ple.set_cache_pool(pool)
    ple.preallocate_verify_workspace(2, 4)
    ple.init_cuda_graph_state(2)
    context_id = qwen4_exp_ple_context_field(1)
    context, conv = ple.ple_verify_scratch(context_id, 1, 2)
    context.copy_(torch.arange(len(context), device="cuda")[:, None].expand_as(context))
    conv.copy_(torch.arange(len(conv), device="cuda")[:, None, None].expand_as(conv))
    qsa = QSAVerifyState(recipe.attn_config, pool)
    qsa.preallocate_verify_workspace(2, 4)
    recent_packing = arena.plan.group(
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP
    ).cache_blocks_per_lcm_block
    recent_blocks = torch.tensor(
        [recent_packing + 1, recent_packing + 2], device="cuda", dtype=torch.int32
    )
    for layer in (7, 8):
        staged, positions, logical, locations = qsa.verify_staging_buffers(layer, 2)
        staged.fill_(layer)
        positions.zero_()
        logical.copy_(torch.arange(4, device="cuda")[None, :].expand_as(logical))
        locations.copy_(recent_blocks[:, None] * 64 + logical)
    accepted = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    pages = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.int32)

    slots = torch.arange(2, device="cuda")
    lengths = torch.tensor([7, 7], device="cuda", dtype=torch.int32)

    def run():
        ple.refresh_decode_metadata(
            2,
            2,
            slots,
            lengths,
            forward_mode=ForwardMode.DECODE,
            block_tables={QWEN4_EXP_PLE_CACHE_GROUP: pages},
        )
        ple.commit_verified_state(accepted)
        qsa.commit_after_mtp_verify(accepted, num_extends=0)

    run()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        accepted.copy_(torch.tensor([1, 4], device="cuda", dtype=torch.int32))
        capture.replay()
    for request, (page, steps) in enumerate(
        zip(pages[:, 1].tolist(), accepted.tolist(), strict=True)
    ):
        source = request * 5 + steps
        torch.testing.assert_close(
            arena.field(context_id)[page], context[source], atol=0, rtol=0
        )
        torch.testing.assert_close(
            arena.field(qwen4_exp_ple_conv_field(1))[page], conv[source], atol=0, rtol=0
        )
        for layer in (7, 8):
            raw = arena.field(qsa_raw_key_field(layer))[recent_blocks[request].item()]
            assert torch.all(raw[:steps] == layer)


@pytest.mark.parametrize("state_layers", [7, 36])
def test_one_gdn_group_keeps_each_layers_input_and_output_states(state_layers):
    from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool

    recipe = _qwen4_exp_recipe(
        flat=True,
        device="cpu",
        kv_dtype=torch.bfloat16,
        state_layers=state_layers,
        draft_num_kv_heads=1,
    )
    setup = recipe.setup()
    arena = CacheArena(
        setup.spec.memory_plan, "cpu", cache_group_specs=setup.spec.cache_group_specs
    )
    pool = create_cache_pool(
        setup.spec.layer_view(first_layer=0, num_layers=setup.num_target_layers),
        recipe.attn_config,
        arena,
        num_layers=setup.num_target_layers,
        rank=0,
        field_layer_offset=0,
    )
    assert pool.state_group_by_layer == {
        layer: "linear_attention_0" for layer in range(state_layers)
    }
    # Every layer uses the same group block IDs, with distinct field offsets.
    # Read after all writes to catch aliasing between layers or input/output.
    for layer in range(state_layers):
        for block in (1, 2):
            pool.get_component(layer, "conv_state")[block].reshape(-1)[0] = (
                100 * block + layer
            )
            pool.get_component(layer, "recurrent_state")[block].reshape(-1)[0] = (
                200 * block + layer
            )
    for layer in range(state_layers):
        for block in (1, 2):
            assert (
                arena.field(f"layer.{layer}.conv")[block].reshape(-1)[0]
                == 100 * block + layer
            )
            assert (
                arena.field(f"layer.{layer}.ssm")[block].reshape(-1)[0]
                == 200 * block + layer
            )
