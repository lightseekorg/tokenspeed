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


def _msa_model_config() -> SimpleNamespace:
    config = _tiny_config()
    return SimpleNamespace(
        hf_config=config,
        hf_text_config=config.text_config,
        context_len=4096,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        dtype=torch.bfloat16,
    )


def _msa_server_args(**overrides) -> SimpleNamespace:
    args = dict(
        device="cuda",
        kv_cache_dtype="fp8_e4m3",
        kv_cache_quant_method="none",
        speculative_algorithm=None,
        attention_backend="trtllm",
        drafter_attention_backend=None,
        block_size=128,
        max_num_seqs=16,
        data_parallel_size=1,
        attn_tp_size=4,
        max_cudagraph_capture_size=16,
        chunked_prefill_size=8192,
        disaggregation_mode="null",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_msa_config_kv_cache_dtype_guards() -> None:
    from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig

    model_config = _msa_model_config()

    config = MSAConfig.generate(_msa_server_args(), model_config)
    assert config.kv_cache_dtype is torch.float8_e4m3fn
    # The index side cache stays in model dtype regardless of kv_cache_dtype.
    assert config.dtype is torch.bfloat16

    with pytest.raises(ValueError, match="mxfp8"):
        MSAConfig.generate(_msa_server_args(kv_cache_dtype="mxfp8"), model_config)
    with pytest.raises(ValueError, match="kv_cache_quant_method"):
        MSAConfig.generate(
            _msa_server_args(kv_cache_quant_method="per_token_head"), model_config
        )


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


def _mixed_precision_config() -> "ModelOptMixedConfig":
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        ModelOptMixedConfig,
    )

    layers: dict = {}
    for i in range(4):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layers[f"language_model.model.layers.{i}.self_attn.{proj}"] = {
                "quant_algo": "MXFP8"
            }
    for i in range(3):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            layers[f"language_model.model.layers.{i}.mlp.{proj}"] = {
                "quant_algo": "MXFP8"
            }
    for proj in ("index_q_proj", "index_k_proj"):
        layers[f"language_model.model.layers.3.self_attn.{proj}"] = {
            "quant_algo": "MXFP8"
        }
    for proj in ("gate_proj", "up_proj", "down_proj"):
        layers[
            "language_model.model.layers.3.block_sparse_moe." f"shared_experts.{proj}"
        ] = {"quant_algo": "MXFP8"}
    for expert in range(8):
        for proj in ("w1", "w2", "w3"):
            layers[
                "language_model.model.layers.3.block_sparse_moe."
                f"experts.{expert}.{proj}"
            ] = {"quant_algo": "NVFP4", "group_size": 16}

    config = ModelOptMixedConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt",
            "exclude_modules": [
                "lm_head",
                "model.embed_tokens",
                "language_model.model.layers.3.block_sparse_moe.gate",
            ],
            "quantized_layers": layers,
        }
    )
    config.apply_checkpoint_name_replacements(
        MiniMaxM3SparseForConditionalGeneration.quant_module_name_replacements
    )
    return config


def test_minimax_m3_mixed_precision_quant_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.layers.dense import Fp8LinearMethod

    model = _build_model(monkeypatch, quant_config=_mixed_precision_config())

    attn = model.model.layers[3].self_attn
    assert isinstance(attn.qkv_proj.quant_method, Fp8LinearMethod)
    assert isinstance(attn.o_proj.quant_method, Fp8LinearMethod)
    assert isinstance(attn.indexer.index_q_proj.quant_method, Fp8LinearMethod)
    assert isinstance(attn.indexer.index_k_proj.quant_method, Fp8LinearMethod)

    assert isinstance(
        model.model.layers[0].mlp.gate_up_proj.quant_method, Fp8LinearMethod
    )

    moe = model.model.layers[3].mlp
    assert isinstance(moe.shared_experts.gate_up_proj.quant_method, Fp8LinearMethod)
    experts = moe.experts
    assert experts._quant_kind == "nvfp4"
    assert experts.w13_weight.dtype == torch.uint8
    assert experts.w13_weight_scale.dtype == torch.float8_e4m3fn
    assert experts.w13_weight_scale_2.shape == (8, 2)


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
