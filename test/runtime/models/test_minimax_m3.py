from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import MiniMaxM3VLTextConfig

from tokenspeed.runtime.configs import MiniMaxM3Config
from tokenspeed.runtime.configs.minimax_m3_config import MiniMaxM3VisionConfig
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    _resolve_attention_family,
    is_multimodal_model,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config
from tokenspeed.runtime.models.minimax_m3 import (
    MiniMaxM3MLP,
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3SparseMoeBlock,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import _CONFIG_REGISTRY


def _tiny_config() -> MiniMaxM3Config:
    return MiniMaxM3Config(
        text_config=MiniMaxM3VLTextConfig(
            vocab_size=96,
            hidden_size=128,
            intermediate_size=128,
            dense_intermediate_size=256,
            shared_intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=16,
            rotary_dim=8,
            max_position_embeddings=4096,
            num_local_experts=8,
            num_experts_per_tok=4,
            mlp_layer_types=["dense", "dense", "dense", "sparse"],
            layer_types=[
                "full_attention",
                "full_attention",
                "full_attention",
                "minimax_m3_sparse",
            ],
            index_n_heads=4,
            index_head_dim=128,
            index_block_size=128,
            index_topk_blocks=16,
            index_local_blocks=1,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 5_000_000,
                "partial_rotary_factor": 0.5,
            },
            dtype="bfloat16",
        ),
        vision_config=MiniMaxM3VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            vision_segment_max_frames=2,
        ),
        projector_hidden_size=64,
    )


def _mxfp8_config() -> Mxfp8Config:
    return Mxfp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [1, 32],
            "ignored_layers": [
                "lm_head",
                "model.embed_tokens",
                "model.layers.3.block_sparse_moe.gate",
            ],
        }
    )


def _tp4_mapping() -> Mapping:
    return Mapping(
        rank=0,
        world_size=4,
        attn_tp_size=4,
        attn_cp_size=1,
        attn_dp_size=1,
        dense_tp_size=4,
        dense_dp_size=1,
        moe_tp_size=4,
        moe_ep_size=1,
        moe_dp_size=1,
        nprocs_per_node=4,
        nnodes=1,
    )


def _build_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    quant_config: Mxfp8Config | None = None,
    is_multimodal_active: bool = False,
) -> MiniMaxM3SparseForConditionalGeneration:
    mapping = _tp4_mapping()
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 2048)
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 2048)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return MiniMaxM3SparseForConditionalGeneration(
                _tiny_config(),
                mapping,
                quant_config=quant_config,
                is_multimodal_active=is_multimodal_active,
                mm_attention_backend="triton_attn",
            )
    finally:
        torch.set_default_dtype(old_dtype)


def test_minimax_m3_config() -> None:
    config = _tiny_config()

    assert _CONFIG_REGISTRY["minimax_m3_vl"] is MiniMaxM3Config
    assert config.runtime_attention_arch == "MSA"
    assert config.text_config.layer_types[-1] == "minimax_m3_sparse"
    assert config.text_config.mlp_layer_types[-1] == "sparse"
    assert config.text_config.index_block_size == 128
    assert isinstance(config.vision_config, MiniMaxM3VisionConfig)
    assert is_multimodal_model(["MiniMaxM3SparseForConditionalGeneration"])


def test_minimax_m3_attention_family_selects_msa() -> None:
    config = _tiny_config()
    config.architectures = ["MiniMaxM3SparseForConditionalGeneration"]

    spec = _resolve_attention_family(config, config.text_config)
    assert spec is not None
    assert spec.name == "MiniMax MSA"
    assert spec.default_block_size == 128
    # --attention-backend must keep selecting the dense sub-backend; the
    # top-level backend is pinned by MSAConfig itself.
    assert spec.default_backend is None

    model_config = SimpleNamespace(attention_arch=None)
    spec.configure(model_config)
    assert model_config.attention_arch is AttentionArch.MSA


def test_minimax_m3_tp4_meta_layout_and_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _build_model(monkeypatch, quant_config=_mxfp8_config())

    assert isinstance(model.model.layers[0].mlp, MiniMaxM3MLP)
    assert isinstance(model.model.layers[3].mlp, MiniMaxM3SparseMoeBlock)
    experts = model.model.layers[3].mlp.experts
    assert experts.w13_weight.shape == (8, 64, 128)
    assert experts.w13_weight_scale_inv.dtype == torch.uint8

    loaded = model.load_weights(
        [
            (
                "language_model.model.layers.3.block_sparse_moe."
                "experts.0.w1.weight_scale_inv",
                torch.empty(128, 4, dtype=torch.uint8, device="meta"),
            ),
            (
                "language_model.model.layers.3.self_attn.index_q_norm.weight",
                torch.empty(128, dtype=torch.bfloat16, device="meta"),
            ),
        ]
    )
    assert loaded == {
        "model.layers.3.mlp.experts.w13_weight_scale_inv",
        "model.layers.3.self_attn.indexer.q_norm.weight",
    }


def test_minimax_m3_active_multimodal_layout_and_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch, is_multimodal_active=True)

    assert model.vision_tower is not None
    assert model.multi_modal_projector is not None
    assert model.patch_merge_mlp is not None
    assert model.multimodal_embedder is not None

    loaded = model.load_weights(
        [
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
                torch.empty((32, 32), device="meta"),
            ),
        ]
    )
    assert loaded == {
        "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj.weight",
    }


def test_minimax_m3_language_only_keeps_vision_modules_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch)

    assert model.vision_tower is None
    assert model.multimodal_embedder is None

    with pytest.raises(RuntimeError, match="vision tower is disabled"):
        model.get_image_feature([])
