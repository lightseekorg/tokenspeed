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

"""Config/cache setup checks with small CPU arenas; no weights or downloads.

Set DEEPSEEK_V41_REFERENCE_DIR to a local Flash snapshot to repeat the portable
fixture checks against its actual config.json. No snapshot path is assumed.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from transformers import AutoConfig, PretrainedConfig

from tokenspeed.runtime.configs import DeepseekV41Config, DeepseekV41TextConfig
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    ModelConfig,
    configure_deepseek_v41_attention,
)
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    DeepseekV41AttentionBackend,
)
from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import (
    DeepseekV41Config as DeepseekV41AttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v41 import (
    DeepseekV41CachePool,
)
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v41 import (
    DeepseekV41Recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    _RECIPES,
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.attention.registry import (
    _create_attn_backend,
    _create_attn_config,
    _resolve_attn_side,
    _resolve_cache_family,
)
from tokenspeed.runtime.utils.hf_transformers_utils import _CONFIG_REGISTRY, get_config
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.fixture
def raw_config():
    return {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": 129264,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 129280,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "max_position_embeddings": 1048576,
            "rope_theta": 10000,
            "compress_rope_theta": 160000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            },
            "sliding_window": 128,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 512,
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "engram_layer_ids": [1, 14],
            "engram_max_ngram_size": 4,
            "num_nextn_predict_layers": 3,
        },
        "vision_config": {"model_type": "deepseek_v41_vision", "hidden_size": 1024},
    }


@pytest.fixture(params=("fixture", "reference"))
def config_dir(request, tmp_path, raw_config):
    if request.param == "reference":
        reference_dir = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
        if not reference_dir:
            pytest.skip("set DEEPSEEK_V41_REFERENCE_DIR to a local Flash snapshot")
        path = Path(reference_dir)
        assert (path / "config.json").is_file(), "reference directory needs config.json"
        return path
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")
    return tmp_path


def _load_config(path):
    return get_config(
        str(path),
        trust_remote_code=False,
        revision=None,
        model_override_args=None,
        is_draft_worker=False,
        speculative_algorithm=None,
        local_files_only=True,
    )


@pytest.fixture
def runtime_config(config_dir):
    args = ServerArgs(
        model=str(config_dir),
        device="cpu",
        dtype="bfloat16",
        trust_remote_code=False,
        revision=None,
        max_model_len=512,
        attention_backend=None,
        prefix_granularity=64,
        load_format="auto",
        speculative_algorithm=None,
        world_size=1,
        attn_tp_size=None,
        data_parallel_size=None,
        pipeline_parallel_size=1,
        max_num_seqs=2,
        chunked_prefill_size=256,
        max_total_tokens=512,
        kv_cache_quant_method="none",
        disaggregation_mode="null",
        enforce_eager=True,
        disable_prefill_graph=True,
        seed=0,
    )
    model = ModelConfig(
        model_path=args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        context_length=args.max_model_len,
        model_override_args="{}",
        dtype=args.dtype,
        quantization=args.quantization,
        override_config_file=None,
        is_draft_worker=False,
        server_args=args,
    )
    return args, model


def test_registry_and_wrapper_roundtrip(config_dir, tmp_path):
    assert _CONFIG_REGISTRY["deepseek_v41"] is DeepseekV41Config
    assert _CONFIG_REGISTRY["deepseek_v41_text"] is DeepseekV41TextConfig
    config = _load_config(config_dir)
    auto_config = AutoConfig.from_pretrained(
        str(config_dir), trust_remote_code=False, local_files_only=True
    )
    roundtrip = tmp_path / "roundtrip"
    config.save_pretrained(str(roundtrip), push_to_hub=False)
    restored = _load_config(roundtrip)
    raw = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))

    for loaded in (config, auto_config, restored):
        assert isinstance(loaded, DeepseekV41Config)
        assert isinstance(loaded.text_config, DeepseekV41TextConfig)
        assert loaded.model_type == "deepseek_v41"
        assert loaded.text_config.model_type == "deepseek_v41_text"
        assert loaded.architectures == ["DeepseekV41ForCausalLM"]
        assert loaded.vision_config == raw["vision_config"]
        assert loaded.image_token_id == 129264
        assert loaded.quantization_config == raw["quantization_config"]
        assert str(loaded.dtype).removeprefix("torch.") == "bfloat16"
        assert str(loaded.text_config.dtype).removeprefix("torch.") == "bfloat16"
        for key, expected in (
            ("hidden_size", 5120),
            ("num_hidden_layers", 40),
            ("num_attention_heads", 64),
            ("num_key_value_heads", 1),
            ("head_dim", 512),
            ("qk_rope_head_dim", 64),
            ("q_lora_rank", 1280),
            ("o_lora_rank", 1024),
            ("o_groups", 8),
            ("index_head_dim", 128),
            ("max_position_embeddings", 1048576),
        ):
            assert getattr(loaded.text_config, key) == expected
            assert getattr(loaded, key) == expected
        for key, expected in (
            ("bos_token_id", 0),
            ("eos_token_id", 1),
            ("pad_token_id", 2),
        ):
            assert getattr(loaded, key) == getattr(loaded.text_config, key) == expected
        text = loaded.text_config
        assert text.compress_ratios == [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
        assert text.num_nextn_predict_layers == 3
        assert text.kv_source_layers is text.kv_source_layer_ids
        assert text.kv_source_layers == [2, 8, 14, 20]
        assert text.index_source_layers is text.index_source_layer_ids
        assert text.index_source_layers == [2, 8, 14, 20, 24, 28, 32, 36]
        assert text.candidate_source_layer == text.candidate_source_layer_id == 20
        assert text.ngram_context_len == 3
        assert (text.num_hash_layers, text.n_group, text.topk_group) == (0, 1, 1)
        assert text.rope_scaling["factor"] == 16
        assert text.rope_scaling["original_max_position_embeddings"] == 65536
    assert restored.text_config.quantization_config == raw["quantization_config"]


def test_text_config_sets_dimensions_before_hf_rope_setup(raw_config, monkeypatch):
    text = raw_config["text_config"]
    original = PretrainedConfig.convert_rope_params_to_dict
    calls = []

    def check_rope_inputs(config, **kwargs):
        for name in (
            "max_position_embeddings",
            "rope_theta",
            "head_dim",
            "hidden_size",
            "num_attention_heads",
            "rope_scaling",
        ):
            assert getattr(config, name) == text[name]
        calls.append(config)
        return original(config, **kwargs)

    monkeypatch.setattr(
        PretrainedConfig, "convert_rope_params_to_dict", check_rope_inputs
    )
    config = DeepseekV41TextConfig(**text)
    assert calls == [config]
    assert config.max_position_embeddings == 1048576


@pytest.mark.parametrize("engram_layers, expected", [([1, 14], 3), ([], 0)])
def test_text_only_roundtrip_and_engram_alias(
    raw_config, tmp_path, engram_layers, expected
):
    text = raw_config["text_config"]
    text["engram_layer_ids"] = engram_layers
    config = DeepseekV41TextConfig(**text)
    config.save_pretrained(str(tmp_path), push_to_hub=False)
    restored = _load_config(tmp_path)
    assert isinstance(restored, DeepseekV41TextConfig)
    assert restored.ngram_context_len == expected
    assert restored.engram_layer_ids == engram_layers
    assert restored.kv_source_layers == text["kv_source_layer_ids"]
    assert restored.num_hidden_layers == 40


def test_model_config_uses_nested_mla_dims_without_yarn_scale(runtime_config):
    args, model = runtime_config
    assert isinstance(model.hf_config, DeepseekV41Config)
    assert model.hf_text_config is model.hf_config.text_config
    assert model.attention_arch is AttentionArch.MLA
    assert args.attention_backend == "deepseek_v41"
    assert args.prefix_granularity == 256
    assert model.num_hidden_layers == model.num_attention_layers == 40
    assert len(model.hf_text_config.compress_ratios) == 43
    assert model.hidden_size == 5120
    assert model.num_attention_heads == 64
    assert model.num_key_value_heads == 1
    assert model.head_dim == model.kv_lora_rank == model.v_head_dim == 512
    assert model.qk_rope_head_dim == 64
    assert model.qk_nope_head_dim == 448
    assert model.index_head_dim == 128
    assert model.hf_text_config.rope_scaling["factor"] == 16
    assert model.scaling == pytest.approx(512**-0.5)
    # V4 would multiply the attention scale for this flag; V4.1 must not.
    model.hf_text_config.rope_scaling["mscale_all_dim"] = True
    configure_deepseek_v41_attention(model)
    assert model.scaling == pytest.approx(512**-0.5)
    assert model.quantization == "fp8"
    assert model.hf_text_config.quantization_config == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }


@pytest.mark.parametrize("overlap_depth", [0, 1])
def test_config_selects_flash_recipe_and_checks_geometry(runtime_config, overlap_depth):
    args, model = runtime_config
    attn = _create_attn_config(args, model, is_draft=False)
    spec = attn.component(DeepseekV41AttnConfig)
    assert attn.device == "cpu"
    assert attn.kernel_page_size == 64
    assert spec.compress_ratios == (0, 0) + (2,) * 18 + (1,) * 20
    assert spec.kv_owners == (-1, -1) + (2,) * 6 + (8,) * 6 + (14,) * 6 + (20,) * 20
    assert spec.index_sources[20:] == tuple(
        source for source in (20, 24, 28, 32, 36) for _ in range(4)
    )
    assert (spec.index_topk, spec.candidate_topk, spec.candidate_block_size) == (
        512,
        2048,
        8,
    )
    profile = _resolve_attn_side(model, requested_backend=args.attention_backend)
    family = _resolve_cache_family(profile, model, attn)
    assert family == "deepseek_v41"
    assert _RECIPES[family] is DeepseekV41Recipe
    recipe = _RECIPES[family](
        server_args=args,
        model_config=model,
        attn_config=attn,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=256 << 20,
        decode_input_tokens=1,
        overlap_schedule_depth=overlap_depth,
    )
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    recipe.check_layout(layout)
    assert layout.lcm_block_bytes == 1_382_400
    assert layout.plane_bytes == (("flatkv", 1_382_400),)
    assert dict(layout.group_packing) == {
        "v41.swa": 1,
        "v41.global_r2": 20,
        "v41.global_r1": 60,
        "v41.compressor_tail_r2": 54,
    }
    assert [group.block_granularity for group, _ in groups] == [64, 128, 64, 2]
    assert [len(fields) for _, fields in groups] == [40, 6, 2, 3]
    assert [group.sliding_window_tokens for group, _ in groups] == [
        129 + overlap_depth,
        None,
        None,
        3 + overlap_depth,
    ]
    assert all(group.family == "history" for group, _ in groups)
    assert all(field.page_stride_bytes % 256 == 0 for field in layout.fields)
    with pytest.raises(ValueError, match="one 1,382,400-byte plane"):
        recipe.check_layout(replace(layout, lcm_block_bytes=1_382_400 + 256))


@pytest.mark.parametrize("overlap_depth", [0, 1])
def test_real_server_args_prepare_cache_pool_and_backend(runtime_config, overlap_depth):
    args, model = runtime_config
    assert isinstance(args, ServerArgs)
    assert args.pipeline_parallel_size == model.mapping.pp_size == 1
    attn = _create_attn_config(args, model, is_draft=False)
    profile = _resolve_attn_side(model, requested_backend=args.attention_backend)
    setup = prepare_cache_setup(
        family=_resolve_cache_family(profile, model, attn),
        server_args=args,
        model_config=model,
        attn_config=attn,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=128 << 20,
        decode_input_tokens=1,
        overlap_schedule_depth=overlap_depth,
    )
    assert setup.spec.family == "deepseek_v41"
    assert setup.num_target_layers == 40
    assert setup.num_draft_layers == 0
    assert setup.spec.token_capacity == args.max_total_tokens
    plan = setup.spec.memory_plan
    assert 0 < plan.arena_bytes < 64 << 20
    assert plan.arena_bytes + setup.fixed_workspace_bytes <= setup.cache_budget_bytes
    arena = create_cache_arena(
        setup.spec, device=attn.device, enable_memory_saver=False
    )
    pool = create_cache_pool(
        setup.spec,
        attn,
        arena,
        num_layers=model.num_attention_layers,
        rank=0,
        field_layer_offset=0,
    )
    backend = _create_attn_backend(model.attention_arch, attn)
    assert isinstance(pool, DeepseekV41CachePool)
    assert isinstance(backend, DeepseekV41AttentionBackend)
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(
        attn.max_bs, max_tokens_per_req=1, overlap_schedule_depth=overlap_depth
    )
    assert backend.cache_pool is pool
    assert pool.arena is arena
    assert arena.buffer.device.type == "cpu"
    assert arena.buffer.numel() == plan.arena_bytes
    assert arena.runtime_contract.token_capacity == args.max_total_tokens
    assert arena.runtime_contract.group_specs == setup.spec.cache_group_specs
    for view, shape in (
        (pool.swa(39), (64, 528)),
        (pool.global_kv(2), (64, 288)),
        (pool.index_k(20), (64, 68)),
        (pool.compressor_tail(14), (2, 2, 512)),
    ):
        assert tuple(view.shape[1:]) == shape
        assert (
            view.untyped_storage().data_ptr()
            == arena.buffer.untyped_storage().data_ptr()
        )
    assert backend.cuda_graph_support.decode_graph
    assert not backend.cuda_graph_support.prefill_graph
