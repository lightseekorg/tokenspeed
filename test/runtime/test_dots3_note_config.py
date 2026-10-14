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

"""Checkpoint configuration must select the mixed FP8 cache on Hopper."""

import json

import pytest
import torch

from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    ModelConfig,
    is_audio_model,
    is_multimodal_model,
)
from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    Dots3NoteAttentionConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import cache_recipe
from tokenspeed.runtime.layers.attention.registry import (
    _create_attn_config,
    _ordinary_cache_family,
    _resolve_attn_side,
    _resolve_cache_family,
    _resolve_heterogeneous_draft_family,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.fixture
def checkpoint(tmp_path):
    dsa_layers = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45)
    config = dict(
        model_type="dots3_note",
        architectures=["Dots3NoteForCausalLM"],
        torch_dtype="bfloat16",
        vocab_size=152064,
        hidden_size=5120,
        num_hidden_layers=46,
        num_attention_heads=128,
        num_key_value_heads=128,
        intermediate_size=13824,
        moe_intermediate_size=1536,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=2048,
        max_position_embeddings=524288,
        rope_theta=80000000.0,
        rope_scaling=None,
        rms_norm_eps=1e-5,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=64,
        swa_q_lora_rank=1024,
        swa_kv_lora_rank=1024,
        swa_qk_nope_head_dim=192,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=128,
        swa_rope_theta=50000.0,
        sliding_window_size=513,
        apply_mla_qkv_lora_rescale=True,
        attention_gate_type="headwise",
        swa_attention_gate_type="headwise",
        layer_types=[
            "full_attention" if i in dsa_layers else "sliding_attention"
            for i in range(46)
        ],
        quantization_config=dict(
            quant_method="fp8",
            fmt="e4m3",
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        ),
        bos_token_id=151643,
        eos_token_id=151668,
        tie_word_embeddings=False,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return tmp_path


def _load(path, backend=None):
    args = ServerArgs(
        model=str(path),
        device="cpu",
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=8192,
        attention_backend=backend,
        world_size=8,
        max_num_seqs=2,
        chunked_prefill_size=512,
        kv_cache_dtype="fp8_e4m3",
        kv_cache_quant_method="none",
        language_model_only=True,
        enforce_eager=True,
        disable_prefill_graph=True,
    )
    args.mapping.rank = 0
    model = ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
    )
    return args, model


def test_checkpoint_reaches_two_group_cache(checkpoint):
    original = (checkpoint / "config.json").read_bytes()
    args, model = _load(checkpoint)
    config = _create_attn_config(args, model)
    spec = config.component(Dots3NoteAttentionConfig)
    assert model.attention_arch == AttentionArch.DSA
    assert config.kv_cache_dtype == torch.float8_e4m3fn
    assert args.prefix_granularity == 128
    assert args.attention_backend == "dsa"
    assert spec.num_attention_heads // spec.attn_tp_size == 16
    assert spec.swa.num_attention_heads // spec.swa.attn_tp_size == 8
    assert (spec.kv_cache_dim, spec.swa.kv_cache_dim) == (576, 1088)
    family = _resolve_cache_family(
        _resolve_attn_side(model, requested_backend=args.attention_backend), config
    )
    assert family == "dots3_note"
    assert _ordinary_cache_family(config) == "dsa"
    recipe = cache_recipe(
        family,
        server_args=args,
        model_config=model,
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=256 << 20,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    plan = pack(
        recipe.groups(),
        prefix_granularity=config.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(recipe.groups()),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    assert [
        (group.group_id, group.block_granularity) for group, _ in recipe.groups()
    ] == [("dsa", 64), ("swa", 32)]
    assert plan.lcm_block_bytes == 1178112
    assert (checkpoint / "config.json").read_bytes() == original
    assert is_multimodal_model(model.hf_config.architectures)
    assert is_audio_model(model.hf_config.architectures)


def test_unsupported_attention_backend_is_not_ignored(checkpoint):
    args, model = _load(checkpoint, backend="mla")
    with pytest.raises(ValueError, match="dots3-note"):
        _create_attn_config(args, model)


def test_mtp_uses_one_swa_layer_in_the_shared_cache(checkpoint):
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

    args, model = _load(checkpoint)
    args.speculative_algorithm = "MTP"
    draft = ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
        is_draft_worker=True,
    )
    assert model.num_attention_layers == 46
    assert draft.hf_config.architectures == ["Dots3NoteForCausalLMNextN"]
    assert draft.num_attention_layers == 1
    assert draft.hf_text_config.mtp_layer_offset == 46
    assert draft.attention_arch == AttentionArch.MLA
    assert args.attention_backend == "dsa"
    assert args.drafter_attention_backend == "mla"
    assert args.prefix_granularity == 128
    assert (draft.num_attention_heads, draft.kv_lora_rank) == (64, 1024)

    target_config = _create_attn_config(args, model)
    draft_config = _create_attn_config(args, draft, is_draft=True)
    spec = draft_config.component(MLAConfig)
    assert type(spec) is MLAConfig
    assert draft_config.kernel_page_size == 32
    assert spec.cache_layer_types == ("sliding_attention",)
    assert spec.sliding_window_tokens == 513
    assert draft_config.kv_cache_dtype == torch.float8_e4m3fn
    family = _resolve_cache_family(
        _resolve_attn_side(model, requested_backend=args.attention_backend),
        target_config,
    )
    assert family == "dots3_note"
    draft_family = _ordinary_cache_family(draft_config)
    assert draft_family == "mla"
    assert (
        _resolve_heterogeneous_draft_family(
            family, draft_family, draft_family_declared=False
        )
        is None
    )
    recipe = cache_recipe(
        family,
        server_args=args,
        model_config=model,
        attn_config=target_config,
        draft_model_config=draft,
        draft_attn_config=draft_config,
        cache_budget_bytes=256 << 20,
        probe_batch_rows=None,
        decode_input_tokens=4,
        overlap_schedule_depth=0,
    )
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=target_config.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    assert [(group.group_id, len(fields)) for group, fields in groups] == [
        ("dsa", 26),
        ("swa", 34),
    ]
    assert layout.lcm_block_bytes == 1183744


@pytest.mark.parametrize("draft_backend", ["mla", "dsa"])
def test_mtp_preserves_explicit_backend_and_prefix(checkpoint, draft_backend):
    args, _ = _load(checkpoint)
    args.speculative_algorithm = "MTP"
    args.drafter_attention_backend = draft_backend
    args.prefix_granularity = 256
    draft = ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
        is_draft_worker=True,
    )
    assert args.attention_backend == "dsa"
    assert args.drafter_attention_backend == draft_backend
    assert args.prefix_granularity == 256
    if draft_backend == "dsa":
        with pytest.raises(ValueError, match="dots3-note MTP requires the mla backend"):
            _create_attn_config(args, draft, is_draft=True)
    else:
        assert _create_attn_config(args, draft, is_draft=True).prefix_granularity == 256
