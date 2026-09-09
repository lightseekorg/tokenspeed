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

"""Qwen side-cache planning does not require recurrent-attention layers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    qsa_compressed_field,
    qsa_raw_key_field,
    qsa_rope_position_field,
    qwen4_exp_ple_context_field,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen4_exp import (
    Qwen4ExpRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)


def _recipe(*, layer_types: tuple[str, ...], speculative: bool) -> Qwen4ExpRecipe:
    text_config = SimpleNamespace(
        ple_layer_ids=(1, 2),
        short_conv_layer_ids=(0, 1),
        ngram_context_len=2,
        short_conv_state_shape=(4, 3),
        indexer_n_heads=1,
        indexer_head_dim=8,
        indexer_compress_ratio=4,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text_config),
        hf_text_config=text_config,
        num_attention_layers=len(layer_types),
    )
    spec = MHAConfig(
        backend_name="qsa",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        layer_types=layer_types,
        sliding_window_tokens=None,
    )
    width = 3 if speculative else 1
    config = AttnConfig(
        # Planning allocates no device tensors. CUDA plus enabled replay makes
        # this exercise the missing-GDN gate before any kernel capability probe.
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        prefix_granularity=256,
        kernel_page_size=64,
        context_len=1024,
        max_bs=2,
        pd_disaggregation_enabled=False,
        speculative_num_steps=2 if speculative else 0,
        speculative_num_draft_tokens=width,
        is_draft=False,
        draft_block_decode=False,
        components=(spec,),
    )
    draft_config = (
        replace(
            config,
            is_draft=True,
            components=(replace(spec, layer_types=(FULL_ATTENTION,)),),
        )
        if speculative
        else None
    )
    return Qwen4ExpRecipe(
        server_args=SimpleNamespace(
            block_size=64,
            max_total_tokens=None,
            speculative_num_draft_tokens=width,
            enable_replay_ssm=True,
        ),
        model_config=model_config,
        attn_config=config,
        draft_model_config=(
            SimpleNamespace(num_attention_layers=1, hf_text_config=text_config)
            if speculative
            else None
        ),
        draft_attn_config=draft_config,
        cache_budget_bytes=1 << 20,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


@pytest.mark.parametrize("speculative", [False, True])
def test_qwen4_full_attention_keeps_ple_and_qsa_without_gdn(speculative) -> None:
    recipe = _recipe(
        layer_types=(FULL_ATTENTION, FULL_ATTENTION), speculative=speculative
    )
    assert not recipe.replay_ssm
    setup = recipe.setup()
    assert setup.spec.family == "qwen4_exp"
    assert {spec.group_id for spec in setup.spec.cache_group_specs} == {
        FULL_ATTENTION,
        QWEN4_EXP_PLE_CACHE_GROUP,
        QWEN4_EXP_QSA_CACHE_GROUP,
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    }
    fields = {field.field_id for field in setup.spec.memory_plan.fields}
    for layer_id in (0, 1):
        assert f"layer.{layer_id}.k" in fields
        assert f"layer.{layer_id}.v" in fields
        assert qwen4_exp_ple_conv_field(layer_id) in fields
        assert qsa_raw_key_field(layer_id) in fields
        assert qsa_rope_position_field(layer_id) in fields
        assert qsa_compressed_field(layer_id) in fields
    assert qwen4_exp_ple_context_field(0) in fields
    assert not any(field.endswith(".ssm") for field in fields)
    if speculative:
        assert qsa_raw_key_field(2) in fields
        # PLE: eight 64-byte rows plus 64 bytes of commit indices. QSA:
        # two layers' 192-byte keys, 216 shared bytes and 32 address bytes.
        assert setup.fixed_workspace_bytes == 576 + 440
    else:
        assert setup.fixed_workspace_bytes == 0


def test_qwen4_state_layer_still_requires_linear_geometry() -> None:
    with pytest.raises(ValueError, match="linear-attention component"):
        _recipe(layer_types=(LINEAR_ATTENTION, FULL_ATTENTION), speculative=False)
