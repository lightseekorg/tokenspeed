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

"""The DSA index-key plane has one layout per ``index_k_format``.

The config names the format, the ordinary recipe plans exactly that plane,
and the pool writes keys in the plane's own dtype without converting.
"""

from __future__ import annotations

from test.runtime.dsa_index_k_test_utils import index_k_pool

import pytest
import torch

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import (
    INDEX_K_FORMATS,
    DSAConfig,
    dsa_index_k_row_bytes,
    index_k_plane_dtype,
    index_k_row_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    _index_k_field,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

HEAD_DIM = 128
PREFIX = 64


def _dsa_spec(index_k_format: str) -> DSAConfig:
    return DSAConfig(
        backend_name="dsa",
        num_attention_heads=8,
        num_kv_heads=8,
        attn_tp_size=1,
        head_dim=192,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=1.0,
        kv_cache_dim=576,
        index_topk=4,
        index_head_dim=HEAD_DIM,
        index_n_heads=4,
        index_k_format=index_k_format,
    )


def _attn_config(spec: DSAConfig) -> AttnConfig:
    return AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=256,
        max_bs=1,
        prefix_granularity=PREFIX,
        kernel_page_size=PREFIX,
        kv_cache_quant_method="",
        pd_disaggregation_enabled=False,
        components=(spec,),
    )


def test_formats_map_to_one_plane_dtype_and_row_width_each():
    assert INDEX_K_FORMATS == ("fp8_scaled", "bf16")
    assert index_k_plane_dtype("fp8_scaled") is torch.uint8
    assert index_k_plane_dtype("bf16") is torch.bfloat16
    assert index_k_row_bytes(HEAD_DIM, "fp8_scaled") == dsa_index_k_row_bytes(HEAD_DIM)
    assert index_k_row_bytes(HEAD_DIM, "bf16") == HEAD_DIM * 2
    with pytest.raises(ValueError, match="index_k_format"):
        index_k_plane_dtype("fp16")
    with pytest.raises(ValueError, match="index_k_format"):
        _dsa_spec("fp16")


@pytest.mark.parametrize(
    "index_k_format, dtype_name, width",
    [
        ("fp8_scaled", "uint8", dsa_index_k_row_bytes(HEAD_DIM)),
        ("bf16", "bfloat16", HEAD_DIM),
    ],
)
def test_recipe_plans_the_declared_plane(index_k_format, dtype_name, width):
    spec = _dsa_spec(index_k_format)
    field = _index_k_field(_attn_config(spec), layer_id=3)
    assert field.field_id == "layer.3.index_k"
    assert field.shape == (PREFIX, width)
    assert field.dtype == dtype_name


def test_the_configure_attention_hook_names_the_plane():
    from types import SimpleNamespace

    from tokenspeed.runtime.configs.model_config import configure_dsa_attention

    text_config = SimpleNamespace(
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_topk=4,
        index_head_dim=HEAD_DIM,
        index_n_heads=4,
    )
    model_config = SimpleNamespace(
        hf_text_config=text_config, hf_config=text_config, index_k_format=None
    )
    configure_dsa_attention(model_config, ServerArgs(model="x"))
    # The in-tree hook keeps the FP8-with-scale plane; a plugin hook that
    # scores the checkpoint's bf16 keys overrides it after.
    assert model_config.index_k_format == "fp8_scaled"
    # A hook that named no plane is a construction error, not a fallback.
    with pytest.raises(ValueError, match="index_k_format"):
        _dsa_spec(None)


def test_bf16_plane_costs_its_own_rows():
    fp8 = _dsa_spec("fp8_scaled")
    bf16 = _dsa_spec("bf16")
    config = _attn_config(fp8)
    assert bf16.cache_cell_size(config) - fp8.cache_cell_size(config) == (
        HEAD_DIM * 2 - dsa_index_k_row_bytes(HEAD_DIM)
    )


@pytest.mark.parametrize("index_k_format", INDEX_K_FORMATS)
def test_the_history_gather_workspace_plan_follows_the_plane_format(index_k_format):
    """The query-context-parallel gather workspace the recipe reserves holds
    one history of latent rows plus index-K rows in the plane's own width."""
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        dsa_history_gather_workspace_bytes,
    )

    spec = _dsa_spec(index_k_format)
    config = _attn_config(spec)
    rows = 256  # context_len, already whole kernel pages of PREFIX
    assert dsa_history_gather_workspace_bytes(
        config, max_model_len=config.context_len
    ) == rows * (spec.kv_cache_dim * 2 + index_k_row_bytes(HEAD_DIM, index_k_format))


def _qcp_attn_config(spec: DSAConfig) -> AttnConfig:
    """A query-context-parallel config (the device string and bf16 cache the
    QCP checks require; nothing is allocated)."""
    return AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=256,
        max_bs=1,
        prefix_granularity=PREFIX,
        kernel_page_size=PREFIX,
        kv_cache_quant_method="none",
        pd_disaggregation_enabled=False,
        qcp_size=2,
        qcp_group=(0, 1),
        components=(spec,),
    )


def _recipe(target: AttnConfig, draft: AttnConfig | None):
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        OrdinaryRecipe,
    )

    return OrdinaryRecipe(
        family="dsa",
        server_args=SimpleNamespace(max_total_tokens=None),
        model_config=SimpleNamespace(num_attention_layers=2),
        attn_config=target,
        draft_model_config=None if draft is None else SimpleNamespace(),
        draft_attn_config=draft,
        cache_budget_bytes=1 << 30,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def test_the_recipe_refuses_a_draft_of_another_index_k_format():
    """One history gather workspace serves the target and the draft under
    query context parallelism, so the recipe that plans it refuses a draft
    storing index keys in another format, naming both, before any leaf is
    built; a draft of the same format shares the target's plan."""
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        dsa_history_gather_workspace_bytes,
    )

    target = _qcp_attn_config(_dsa_spec("bf16"))
    same = _recipe(target, _qcp_attn_config(_dsa_spec("bf16")))
    assert same.workspace_bytes() == dsa_history_gather_workspace_bytes(
        target, max_model_len=target.context_len
    )
    with pytest.raises(ValueError, match="'bf16', the draft's 'fp8_scaled'"):
        _recipe(target, _qcp_attn_config(_dsa_spec("fp8_scaled"))).workspace_bytes()
    # Without query context parallelism there is no workspace to share.
    assert _recipe(_attn_config(_dsa_spec("bf16")), None).workspace_bytes() == 0


def _pool(plane: torch.Tensor) -> DSATokenToKVPool:
    return index_k_pool(plane, head_dim=HEAD_DIM, page_size=PREFIX)


def test_bf16_plane_takes_the_keys_unquantized():
    plane = torch.zeros((8, HEAD_DIM), dtype=torch.bfloat16)
    keys = torch.randn(3, HEAD_DIM).to(torch.bfloat16)
    loc = torch.tensor([5, 1, 6])

    _pool(plane).set_index_k_buffer(0, loc, keys, write_mask=None)

    assert torch.equal(plane[loc], keys)
    assert torch.equal(
        plane[[0, 2, 3, 4, 7]], torch.zeros(5, HEAD_DIM, dtype=plane.dtype)
    )


def test_bf16_plane_write_mask_keeps_foreign_rows():
    plane = torch.full((8, HEAD_DIM), 7.0, dtype=torch.bfloat16)
    keys = torch.randn(3, HEAD_DIM).to(torch.bfloat16)
    loc = torch.tensor([5, 1, 6])

    _pool(plane).set_index_k_buffer(
        0, loc, keys, write_mask=torch.tensor([True, False, True])
    )

    assert torch.equal(plane[5], keys[0]) and torch.equal(plane[6], keys[2])
    assert torch.equal(plane[1], torch.full((HEAD_DIM,), 7.0, dtype=plane.dtype))


def test_unknown_plane_dtype_has_no_write_path():
    plane = torch.zeros((8, HEAD_DIM), dtype=torch.float16)
    with pytest.raises(TypeError, match="no write path"):
        _pool(plane).set_index_k_buffer(
            0, torch.tensor([0]), torch.zeros(1, HEAD_DIM), write_mask=None
        )
