from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.minimax_m3_config import (
    MiniMaxM3TextConfig,
    MiniMaxM3VLConfig,
)
from tokenspeed.runtime.configs.model_config import is_multimodal_model
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.minimax_m3 import (
    MiniMaxM3MLP,
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3SparseMoeBlock,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def _tiny_config() -> MiniMaxM3VLConfig:
    return MiniMaxM3VLConfig(
        text_config=MiniMaxM3TextConfig(
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
            moe_layer_freq=[0, 0, 0, 1],
            dtype="bfloat16",
        )
    )


def _mxfp8_config() -> Fp8Config:
    return Fp8Config.from_config(
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


def test_minimax_m3_config_describes_msa_contract() -> None:
    config = MiniMaxM3VLConfig()

    assert config.model_type == "minimax_m3_vl"
    assert config.text_config.max_position_embeddings == 1048576
    assert not hasattr(config.text_config, "tokenspeed_context_limit")
    assert config.text_config.moe_layer_freq[:3] == [0, 0, 0]
    assert all(config.text_config.moe_layer_freq[3:])
    assert config.text_config.sparse_attention_config["sparse_topk_blocks"] == 16
    assert config.text_config.sparse_attention_config["sparse_block_size"] == 128
    assert config.text_config.sparse_attention_config["sparse_attention_freq"][:3] == [
        0,
        0,
        0,
    ]
    assert is_multimodal_model(["MiniMaxM3SparseForConditionalGeneration"])

    with pytest.raises(ValueError, match="one entry per decoder layer"):
        MiniMaxM3TextConfig(num_hidden_layers=4, moe_layer_freq=[0, 1])


def test_minimax_m3_tp4_meta_layout_and_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = _tp4_mapping()
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 2048)
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 2048)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = MiniMaxM3SparseForConditionalGeneration(
                _tiny_config(),
                mapping,
                quant_config=_mxfp8_config(),
                is_multimodal_active=False,
            )
    finally:
        torch.set_default_dtype(old_dtype)

    assert isinstance(model.model.layers[0].mlp, MiniMaxM3MLP)
    assert isinstance(model.model.layers[3].mlp, MiniMaxM3SparseMoeBlock)
    experts = model.model.layers[3].mlp.experts
    assert experts.plan["apply_kernel_name"] == "triton_mxfp8_precomputed_moe_apply"
    assert experts.w13_weight.shape == (8, 64, 128)
    assert experts.w13_weight_scale_inv.shape == (8, 64, 4)
    assert experts.w13_weight_scale_inv.dtype == torch.uint8
    assert experts.w2_weight.shape == (8, 128, 32)
    assert experts.w2_weight_scale_inv.shape == (8, 128, 1)

    loaded = model.load_weights(
        [
            (
                "language_model.model.layers.3.block_sparse_moe."
                "e_score_correction_bias",
                torch.empty(8, dtype=torch.float32, device="meta"),
            ),
            (
                "language_model.model.layers.3.block_sparse_moe." "experts.0.w1.weight",
                torch.empty(128, 128, dtype=torch.float8_e4m3fn, device="meta"),
            ),
            (
                "language_model.model.layers.3.block_sparse_moe."
                "experts.0.w1.weight_scale_inv",
                torch.empty(128, 4, dtype=torch.uint8, device="meta"),
            ),
            (
                "language_model.model.layers.3.block_sparse_moe."
                "shared_experts.gate_proj.weight_scale_inv",
                torch.empty(128, 4, dtype=torch.uint8, device="meta"),
            ),
            (
                "language_model.model.layers.3.self_attn.index_q_proj.weight",
                torch.empty(
                    512,
                    128,
                    dtype=torch.float8_e4m3fn,
                    device="meta",
                ),
            ),
            (
                "language_model.model.layers.3.self_attn."
                "index_q_proj.weight_scale_inv",
                torch.empty(512, 4, dtype=torch.uint8, device="meta"),
            ),
            (
                "language_model.model.layers.3.self_attn.index_k_proj.weight",
                torch.empty(
                    128,
                    128,
                    dtype=torch.float8_e4m3fn,
                    device="meta",
                ),
            ),
            (
                "language_model.model.layers.3.self_attn."
                "index_k_proj.weight_scale_inv",
                torch.empty(128, 4, dtype=torch.uint8, device="meta"),
            ),
            (
                "language_model.model.layers.3.self_attn.index_q_norm.weight",
                torch.empty(128, dtype=torch.bfloat16, device="meta"),
            ),
            (
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                torch.empty(1, device="meta"),
            ),
        ]
    )
    assert loaded == {
        "model.layers.3.mlp.routing_bias",
        "model.layers.3.mlp.experts.w13_weight",
        "model.layers.3.mlp.experts.w13_weight_scale_inv",
        "model.layers.3.mlp.shared_experts.gate_up_proj.weight_scale_inv",
        "model.layers.3.self_attn.indexer.index_q_proj.weight",
        "model.layers.3.self_attn.indexer.index_q_proj.weight_scale_inv",
        "model.layers.3.self_attn.indexer.index_k_proj.weight",
        "model.layers.3.self_attn.indexer.index_k_proj.weight_scale_inv",
        "model.layers.3.self_attn.indexer.q_norm.weight",
    }


def test_minimax_m3_rejects_active_multimodal_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _tp4_mapping()
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 2048)

    with pytest.raises(ValueError, match="language-only"):
        MiniMaxM3SparseForConditionalGeneration(
            _tiny_config(),
            mapping,
            quant_config=None,
            is_multimodal_active=True,
        )


def _msa_server_args(*, block_size: int = 128, kv_cache_dtype: str = "auto"):
    return SimpleNamespace(
        device="cuda",
        attention_backend="mha",
        drafter_attention_backend=None,
        attn_tp_size=4,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_size=4, dp_size=1)),
        kv_cache_dtype=kv_cache_dtype,
        max_num_seqs=8,
        data_parallel_size=None,
        block_size=block_size,
        max_cudagraph_capture_size=8,
        kv_cache_quant_method="none",
        speculative_algorithm=None,
        chunked_prefill_size=8192,
        disaggregation_mode="null",
    )


def _msa_model_config():
    return SimpleNamespace(
        context_len=1048576,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        index_head_dim=128,
        hf_config=SimpleNamespace(),
    )


def test_minimax_m3_mha_cache_contract() -> None:
    config = MHAConfig.generate(_msa_server_args(), _msa_model_config())

    assert config.page_size == 128
    assert config.kv_cache_dtype == torch.bfloat16
    assert config.index_head_dim == 128
    assert config.cache_cell_size() == 768

    with pytest.raises(ValueError, match="block-size 128"):
        MHAConfig.generate(
            _msa_server_args(block_size=64),
            _msa_model_config(),
        )
    with pytest.raises(ValueError, match="BF16 KV cache"):
        MHAConfig.generate(
            _msa_server_args(kv_cache_dtype="fp8"),
            _msa_model_config(),
        )
