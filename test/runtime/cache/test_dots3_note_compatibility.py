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

"""CPU checks of legacy arena views after deferring ``field()`` construction.

Uses normal imports and the existing plan helpers; no GPU kernels or model loads.
"""

from dataclasses import replace
from test.runtime.cache.test_dots3_note_recipe import _layout, configure_mtp
from test.runtime.cache.test_dots3_note_recipe import inputs as inputs
from test.runtime.cache.test_dots3_note_recipe import mtp_inputs as mtp_inputs
from test.runtime.cache_pool_test_utils import make_arena, one_group

import pytest
import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)


@pytest.mark.parametrize(
    "shape,dtype,family,expected_shape",
    [
        ((128, 1, 576), "bfloat16", "history", (384, 1, 576)),
        ((128, 2), "float32", "state", (3, 128, 2)),
        ((64, 132), "uint8", "history", (3, 64, 132)),
        ((1, 4, 16), "uint8", "history", (3, 1, 4, 16)),
    ],
    ids=["mla-tokens", "prefix-sized-state", "paged-bytes", "scale-plane"],
)
def test_legacy_field_shapes_and_aliases(shape, dtype, family, expected_shape):
    geometry = (
        {"family": "state", "checkpoint_granularity": 128}
        if family == "state"
        else {"rows_per_page": 128}
    )
    group = one_group(
        "legacy",
        CacheFieldSpec("layer.0.cache", "first", shape, dtype),
        CacheFieldSpec("layer.1.cache", "second", shape, dtype),
        **geometry,
    )
    plan = pack(
        (group,),
        prefix_granularity=128,
        cache_blocks_per_lcm_block={"legacy": 1},
        alignment=1,
        max_padding_fraction=0.0,
    ).bind(2)
    arena = make_arena(plan, "cpu", cache_group_specs=(group[0],))
    field_ids = frozenset({"layer.0.cache", "layer.1.cache"})
    assert arena.field_ids() == field_ids

    for field_id in sorted(field_ids):
        pages = arena.field_pages(field_id)
        assert pages.shape == (3, *shape)  # Includes the null page.
        view = arena.field(field_id)
        assert view.shape == expected_shape
        assert view.dtype == getattr(torch, dtype)
        assert view is arena.field(field_id)
        assert pages is arena.field_pages(field_id)
        assert view.untyped_storage().data_ptr() == arena.buffer.data_ptr()
        assert pages.data_ptr() == view.data_ptr()
        assert view.data_ptr() - arena.buffer.data_ptr() == plan.field_page_byte_offset(
            field_id, 0
        )
        assert (
            pages.stride(0) * pages.element_size()
            == plan.field(field_id).page_stride_bytes
        )
        # A prior field's write must not change this field or its enumeration.
        assert torch.count_nonzero(pages) == 0
        view.fill_(7)
        assert torch.all(pages == 7)
        pages[1].fill_(9)
        assert torch.count_nonzero(view == 9) == pages[1].numel()
        assert arena.field_ids() == field_ids

    arena.clear()
    for field_id in field_ids:
        assert torch.count_nonzero(arena.field(field_id)) == 0
    with pytest.raises(ValueError, match="not planned"):
        arena.field("missing")
    with pytest.raises(ValueError, match="not planned"):
        arena.field_pages("missing")


def build_mtp_pools(inputs, *, device):
    """Bind disjoint production views without allocating a model or a full pool."""
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.dots3_note import (
        Dots3NoteRecipe,
    )
    from tokenspeed.runtime.layers.attention.registry import (
        _create_draft_components,
        _resolve_heterogeneous_draft_family,
    )

    inputs["server_args"].device = device
    configure_mtp(inputs, width=inputs["decode_input_tokens"])
    inputs["draft_model_config"].attention_arch = AttentionArch.MLA
    recipe = Dots3NoteRecipe(**inputs)
    setup = recipe.setup()
    spec = replace(setup.spec, memory_plan=_layout(recipe).bind(32))
    arena = create_cache_arena(spec, device=device, enable_memory_saver=False)
    target_spec = spec.layer_view(first_layer=0, num_layers=46)
    draft_spec = spec.layer_view(
        first_layer=46,
        num_layers=1,
        family=_resolve_heterogeneous_draft_family(
            "dots3_note", "dots3_note", draft_family_declared=True
        ),
    )
    target = create_cache_pool(
        target_spec, inputs["attn_config"], arena, num_layers=46, rank=0
    )
    router, draft = _create_draft_components(
        server_args=inputs["server_args"],
        model_config=inputs["draft_model_config"],
        config=inputs["draft_attn_config"],
        pool=target,
        cache_spec=draft_spec,
        num_target_layers=46,
        full_attn_backend_name="dots3_note",
        is_heterogeneous=False,
        linear_attention=None,
        is_inkling=False,
        backend=None,
    )
    router.set_cache_pool(draft)
    router.init_cuda_graph_state(inputs["draft_attn_config"].max_bs)
    return target, draft, router


def test_mtp_pool_views_are_disjoint_and_keep_packed_stride(mtp_inputs):
    from tokenspeed.runtime.layers.attention.configs.dots3_note import (
        Dots3NoteAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import (
        Dots3NoteCachePool,
    )
    from tokenspeed.runtime.layers.paged_attention import (
        PagedAttention,
        bind_cache_groups,
    )

    target, draft, router = build_mtp_pools(mtp_inputs, device="cpu")
    assert isinstance(target, Dots3NoteCachePool)
    assert isinstance(draft, Dots3NoteCachePool)
    assert draft.arena is target.arena
    assert target.field_layer_range == range(46)
    assert draft.field_layer_range == range(46, 47)
    assert target.paged_group_ids == ("full", "swa.0", "swa.1", "swa.2")
    assert draft.paged_group_ids == ("draft.swa",)
    assert draft.history_group_by_layer() == {0: "draft.swa"}
    assert router.group_ids == ("draft.swa",)
    assert draft.cache_transfer_layout().consumers == (("layer.46.latent_kv",),)
    assert all(
        "layer.46.latent_kv" not in fields
        for fields in target.cache_transfer_layout().consumers
    )
    pages = draft.get_key_buffer(0)
    assert pages.shape == (481, 32, 1, 1088)
    assert pages.stride(0) * pages.element_size() == 71424
    assert pages.untyped_storage().data_ptr() == draft.arena.buffer.data_ptr()
    assert pages.data_ptr() == draft.arena.field_pages("layer.46.latent_kv").data_ptr()
    assert draft.get_value_buffer(0).shape[-1] == 1024
    for layer_id in (-1, 1, 46):
        with pytest.raises(ValueError, match="outside this cache view"):
            draft._field_layer_id(layer_id)
    leaf_spec = mtp_inputs["draft_attn_config"].component(Dots3NoteAttnConfig).swa
    layer = PagedAttention(
        num_heads=8,
        num_kv_heads=8,
        head_dim=1088,
        v_head_dim=1024,
        scaling=leaf_spec.scaling,
        layer_id=0,
        logit_cap=0.0,
        sliding_window_size=512,
        rotary_emb=None,
        qk_norm=None,
    )
    bind_cache_groups(torch.nn.ModuleList([layer]), draft)
    assert layer.group_id == "draft.swa"
    assert router.leaf_for(layer).kernel_page_size == 32
    assert router.leaf_for(layer).is_draft
    assert router.leaf_for(layer).kernel_solution == "triton"
