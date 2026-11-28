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

"""``--kv-cache-dtype mxfp4`` on Kimi-K3.1: what the config admits, the
400-byte latent rows the cache recipe sizes, and the pool's MXFP4 writes and
reads."""

from __future__ import annotations

import dataclasses
import os
import sys

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.cache_pool_test_utils import make_arena
from test.runtime.conftest import kimi_layout, kimi_recipe

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention.configs.base import resolve_dtype
from tokenspeed.runtime.layers.attention.configs.dsa import dsa_kernel_page_size
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDADSATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    _mla_layer_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
)
from tokenspeed.runtime.layers.attention.registry import _validate_mxfp4_kv_backend

# 256 E2M1 code bytes + 16 UE8M0 exponents + 64 BF16 RoPE values.
_ROW_BYTES = 400


def _mxfp4_recipe(**kwargs):
    return kimi_recipe(
        dsa_index_k_format="fp8_scaled",
        kv_cache_dtype=torch.uint8,
        kv_cache_mxfp4=True,
        **kwargs,
    )


def test_the_dtype_string_names_no_torch_dtype():
    with pytest.raises(ValueError, match="DSA MLA"):
        resolve_dtype("mxfp4")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"kv_cache_dtype": torch.bfloat16}, "uint8"),
        ({"kv_cache_quant_method": "per_token_head"}, "carries its own scales"),
        ({"kvp_size": 2}, "kv-parallel-size"),
    ],
)
def test_attn_config_refuses_what_the_mxfp4_rows_cannot_serve(change, message):
    config = _mxfp4_recipe().attn_config
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(config, **change)


def test_the_kimi_cache_reads_a_uint8_latent_only_as_mxfp4():
    # The config admits uint8 without MXFP4 (DeepSeek V4.1's packed rows);
    # the Kimi MLA pool would read such rows as MXFP4, so its recipe refuses.
    with pytest.raises(ValueError, match="MXFP4 latent format and only that"):
        kimi_layout(
            kimi_recipe(
                dsa_index_k_format="fp8_scaled",
                kv_cache_dtype=torch.uint8,
                kv_cache_mxfp4=False,
            )
        )


def test_attn_config_refuses_mxfp4_without_the_dsa_leaf():
    config = _mxfp4_recipe().attn_config
    dsa, *others = config.components
    plain_mla = MLAConfig(
        **{
            field.name: getattr(dsa, field.name)
            for field in dataclasses.fields(MLAConfig)
        }
    )
    with pytest.raises(ValueError, match="DSA sparse MLA"):
        dataclasses.replace(config, components=(plain_mla, *others))
    with pytest.raises(ValueError, match="DSA sparse MLA"):
        dataclasses.replace(
            config,
            components=(dataclasses.replace(dsa, backend_name="mla"), *others),
        )


@pytest.mark.parametrize("leaf", [None, "dsa"])
def test_the_registry_admits_the_dsa_leaf_behind_the_hybrid_backend(leaf):
    _validate_mxfp4_kv_backend(leaf, AttentionArch.DSA)


@pytest.mark.parametrize("leaf, arch", [("mla", "DSA"), (None, "MLA")])
def test_the_registry_refuses_other_full_attention_leaves(leaf, arch):
    with pytest.raises(ValueError, match="full-attention layers resolve to"):
        _validate_mxfp4_kv_backend(leaf, AttentionArch[arch])


def test_the_kimi_cache_sizes_400_byte_latent_rows():
    recipe = _mxfp4_recipe()
    fp8_recipe = kimi_recipe(dsa_index_k_format="fp8_scaled")
    _, layout = kimi_layout(recipe)
    _, fp8_layout = kimi_layout(fp8_recipe)
    latents = [
        field
        for field in layout.bind(1).fields
        if field.field_id.endswith(".latent_kv")
    ]
    fp8_latents = [
        field
        for field in fp8_layout.bind(1).fields
        if field.field_id.endswith(".latent_kv")
    ]
    assert len(latents) == len(fp8_latents) == 24
    for field, fp8_field in zip(latents, fp8_latents, strict=True):
        assert field.dtype == "uint8"
        assert field.shape == (recipe.prefix_granularity, 1, _ROW_BYTES)
        assert fp8_field.shape == (recipe.prefix_granularity, 1, 576)

    # An LCM block packs more full-attention blocks once the latent page
    # shrinks by 176 of 576 bytes a token, so each cached token costs less.
    def bytes_per_full_attention_token(layout):
        blocks = dict(layout.group_packing)[FULL_ATTENTION]
        return layout.lcm_block_bytes / (blocks * layout.prefix_granularity)

    assert bytes_per_full_attention_token(layout) < 0.8 * (
        bytes_per_full_attention_token(fp8_layout)
    )


def test_the_ordinary_mla_recipe_refuses_mxfp4_rows():
    config = _mxfp4_recipe().attn_config
    with pytest.raises(ValueError, match="Kimi-K3.1 cache recipe only"):
        _mla_layer_fields(config, 64, 0, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_the_pool_writes_mxfp4_rows_and_reads_them_back_decoded():
    from tokenspeed_kernel.ops.attention.prologue import KVCacheFormat
    from tokenspeed_kernel.ops.kvcache.mla_mxfp4 import (
        dequant_gather_mla_latent_mxfp4,
        quantize_store_mla_latent_mxfp4,
    )

    text_config = KimiLinearConfig()
    recipe = _mxfp4_recipe()
    _, layout = kimi_layout(recipe)
    arena = make_arena(layout.bind(2))
    pool = HybridKDADSATokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.uint8,
        quant_method=None,
        kv_lora_rank=text_config.kv_lora_rank,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        layer_num=text_config.num_hidden_layers,
        rank=0,
        layer_types=recipe.layer_types,
        index_head_dim=128,
        index_k_format="fp8_scaled",
        index_k_page_size=dsa_kernel_page_size(recipe.attn_config),
    )
    rank, rope = text_config.kv_lora_rank, text_config.qk_rope_head_dim
    layer_id = recipe.target_group_ids.index(FULL_ATTENTION)
    layer = type("Layer", (), {"layer_id": layer_id})()
    tokens = 50
    g = torch.Generator(device="cuda").manual_seed(0)
    latent = torch.randn(
        tokens, rank + rope, device="cuda", generator=g, dtype=torch.bfloat16
    )
    capacity = pool.get_key_buffer(layer_id)[..., 0].numel()
    loc = torch.randperm(capacity, device="cuda", generator=g)[:tokens]

    pool.set_mla_kv_buffer(
        layer, loc, latent[:, :rank], latent[:, rank:], write_mask=None
    )
    nope, pe = pool.get_mla_kv_buffer(layer, loc)

    standalone = torch.zeros(tokens, 1, _ROW_BYTES, dtype=torch.uint8, device="cuda")
    rows = torch.arange(tokens, device="cuda")
    quantize_store_mla_latent_mxfp4(
        latent[:, :rank],
        latent[:, rank:],
        standalone,
        rows,
        write_mask=None,
        sanitize=pool.latent_write_sanitizes,
    )
    expected = torch.empty(tokens, rank + rope, dtype=torch.bfloat16, device="cuda")
    dequant_gather_mla_latent_mxfp4(standalone, rows, expected, kv_lora_rank=rank)
    assert nope.dtype is torch.bfloat16 and nope.shape == (tokens, 1, rank)
    assert torch.equal(nope[:, 0], expected[:, :rank])
    assert torch.equal(pe[:, 0], latent[:, rank:])
    target = pool.kv_write_target(layer_id, loc, None)
    assert target.format is KVCacheFormat.MXFP4
