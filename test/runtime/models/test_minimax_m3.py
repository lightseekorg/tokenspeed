from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import (
    MiniMaxM3VLTextConfig,
    MiniMaxM3VLVisionConfig,
    PretrainedConfig,
)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLMultiModalProjector as HFMiniMaxM3MultiModalProjector,
)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLVisionModel as HFMiniMaxM3VisionModel,
)

import tokenspeed.runtime.configs.model_config as model_config_module
import tokenspeed.runtime.layers.attention.backends.minimax_sparse as minimax_sparse_backend_module
from tokenspeed.runtime.configs import MiniMaxM3Config
from tokenspeed.runtime.configs.minimax_m3_config import MiniMaxM3VisionConfig
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    ModelConfig,
    is_multimodal_model,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.flat_groups import (
    FlatCacheGroupsMixin,
)
from tokenspeed.runtime.layers.attention.backends.mha import (
    MHAAttnBackend,
    MHADecodeMetadata,
    MHAExtendMetadata,
)
from tokenspeed.runtime.layers.attention.backends.minimax_sparse import (
    MinimaxHybridAttnBackend,
    MinimaxSparseAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.minimax_sparse import (
    MinimaxSparseConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.minimax_sparse import (
    MinimaxSparseKVPool,
)
from tokenspeed.runtime.layers.attention.registry import (
    _create_attn_backend,
    _create_attn_config,
)
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.minimax_m3 import (
    MiniMaxM3Attention,
    MiniMaxM3MLP,
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3SparseMoeBlock,
)
from tokenspeed.runtime.models.minimax_m3_vision import (
    MiniMaxM3MultiModalProjector,
    MiniMaxM3PatchMergeMLP,
    MiniMaxM3VisionTower,
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
    config = _tiny_config()

    assert MiniMaxM3Config.__bases__ == (PretrainedConfig,)
    assert MiniMaxM3VisionConfig.__bases__ == (PretrainedConfig,)
    assert _CONFIG_REGISTRY["minimax_m3_vl"] is MiniMaxM3Config
    assert isinstance(config.text_config, MiniMaxM3VLTextConfig)
    assert isinstance(config.vision_config, MiniMaxM3VisionConfig)
    assert config.model_type == "minimax_m3_vl"
    assert config.runtime_attention_arch == "MSA"
    assert config.runtime_attention_layer_type == "minimax_m3_sparse"
    assert config.text_config.runtime_attention_layer_type == "minimax_m3_sparse"
    assert config.text_config.max_position_embeddings == 4096
    assert not hasattr(config.text_config, "tokenspeed_context_limit")
    assert config.text_config.mlp_layer_types == [
        "dense",
        "dense",
        "dense",
        "sparse",
    ]
    assert config.text_config.layer_types == [
        "full_attention",
        "full_attention",
        "full_attention",
        "minimax_m3_sparse",
    ]
    assert config.text_config.index_topk_blocks == 16
    assert config.text_config.index_block_size == 128
    assert config.vision_config.temporal_patch_size == 2
    assert config.vision_config.spatial_merge_size == 2
    assert config.vision_config.rope_theta == 10000.0
    assert config.projector_hidden_act == "gelu"
    assert config.multimodal_projector_bias
    assert is_multimodal_model(["MiniMaxM3SparseForConditionalGeneration"])

    legacy_config = MiniMaxM3Config(
        text_config={
            "num_hidden_layers": 4,
            "moe_layer_freq": [0, 0, 0, 1],
            "sparse_attention_config": {
                "sparse_attention_freq": [0, 0, 0, 1],
                "sparse_num_index_heads": 4,
                "sparse_index_dim": 16,
                "sparse_block_size": 128,
                "sparse_topk_blocks": 16,
                "sparse_local_block": 1,
            },
        }
    )
    assert legacy_config.text_config.mlp_layer_types[-1] == "sparse"
    assert legacy_config.text_config.layer_types[-1] == "minimax_m3_sparse"
    assert legacy_config.text_config.index_head_dim == 16


def test_minimax_m3_config_selects_msa_arch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    config.architectures = ["MiniMaxM3SparseForConditionalGeneration"]
    server_args = SimpleNamespace(
        mapping=None,
        language_model_only=True,
        disaggregation_mode="null",
        mm_attention_backend=None,
        load_format="auto",
        ext_yaml=None,
        speculative_algorithm=None,
    )
    monkeypatch.setattr(
        model_config_module, "get_config", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        model_config_module,
        "get_generation_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ModelConfig,
        "_verify_quantization",
        lambda self: None,
    )

    model_config = ModelConfig(
        "stub",
        model_override_args="{}",
        dtype="bfloat16",
        server_args=server_args,
    )

    assert model_config.attention_arch == AttentionArch.MSA


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


def test_minimax_m3_active_multimodal_layout_and_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _tp4_mapping()
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 2048)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 2048)

    with torch.device("meta"):
        model = MiniMaxM3SparseForConditionalGeneration(
            _tiny_config(),
            mapping,
            quant_config=None,
            is_multimodal_active=True,
            mm_attention_backend="triton_attn",
        )

    assert model.vision_tower is not None
    assert model.multi_modal_projector is not None
    assert model.patch_merge_mlp is not None
    assert model.multimodal_embedder is not None
    assert model.vision_tower.vision_model.encoder.layers[0].self_attn.tp_size == 4

    loaded = model.load_weights(
        [
            (
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                torch.empty((32, 3, 2, 2, 2), device="meta"),
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
                torch.empty((32, 32), device="meta"),
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.bias",
                torch.empty((32,), device="meta"),
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight",
                torch.empty((32, 32), device="meta"),
            ),
            (
                "multi_modal_projector.linear_1.weight",
                torch.empty((64, 32), device="meta"),
            ),
            (
                "patch_merge_mlp.linear_1.bias",
                torch.empty((64,), device="meta"),
            ),
        ]
    )
    assert loaded == {
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj.weight",
        "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj.bias",
        "vision_tower.vision_model.encoder.layers.0.self_attn.proj.weight",
        "multi_modal_projector.linear_1.weight",
        "patch_merge_mlp.linear_1.bias",
    }


def _reference_vision_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    output = torch.empty_like(query)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        query_slice = query[start:end]
        key_slice = key[start:end]
        value_slice = value[start:end]
        scores = torch.einsum("thd,shd->hts", query_slice, key_slice)
        scores = scores * (query.shape[-1] ** -0.5)
        probabilities = scores.softmax(dim=-1)
        output[start:end] = torch.einsum(
            "hts,shd->thd",
            probabilities,
            value_slice,
        )
    return output


def test_minimax_m3_vision_tower_projects_image_and_video_patches() -> None:
    config = _tiny_config()
    mapping = Mapping(rank=0, world_size=1)
    reference_vision_config = MiniMaxM3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    reference_tower = HFMiniMaxM3VisionModel(reference_vision_config).eval()
    reference_projector = HFMiniMaxM3MultiModalProjector(
        SimpleNamespace(
            vision_config=reference_vision_config,
            text_config=SimpleNamespace(hidden_size=128),
            projector_hidden_size=64,
            merged_hidden_size=512,
        )
    ).eval()
    tower = MiniMaxM3VisionTower(
        config.vision_config,
        mapping,
        prefix="vision_tower",
        mm_attention_backend="triton_attn",
    )
    projector = MiniMaxM3MultiModalProjector(
        vision_hidden_size=config.vision_config.hidden_size,
        projector_hidden_size=config.projector_hidden_size,
        text_hidden_size=config.text_config.hidden_size,
        mapping=mapping,
        prefix="multi_modal_projector",
    )
    merger = MiniMaxM3PatchMergeMLP(
        spatial_merge_size=config.vision_config.spatial_merge_size,
        text_hidden_size=config.text_config.hidden_size,
        projector_hidden_size=config.projector_hidden_size,
        mapping=mapping,
        prefix="patch_merge_mlp",
    )
    for layer in tower.vision_model.encoder.layers:
        layer.self_attn._backend_fn = _reference_vision_attention

    with torch.no_grad():
        tower.vision_model.embeddings.patch_embedding.weight.copy_(
            reference_tower.embeddings.proj.weight
        )
        tower.vision_model.pre_layrnorm.load_state_dict(
            reference_tower.pre_layrnorm.state_dict()
        )
        for layer, reference_layer in zip(
            tower.vision_model.encoder.layers,
            reference_tower.layers,
        ):
            layer.layer_norm1.load_state_dict(reference_layer.layer_norm1.state_dict())
            layer.layer_norm2.load_state_dict(reference_layer.layer_norm2.state_dict())
            layer.self_attn.qkv_proj.weight.copy_(
                torch.cat(
                    [
                        reference_layer.self_attn.q_proj.weight,
                        reference_layer.self_attn.k_proj.weight,
                        reference_layer.self_attn.v_proj.weight,
                    ]
                )
            )
            layer.self_attn.qkv_proj.bias.copy_(
                torch.cat(
                    [
                        reference_layer.self_attn.q_proj.bias,
                        reference_layer.self_attn.k_proj.bias,
                        reference_layer.self_attn.v_proj.bias,
                    ]
                )
            )
            layer.self_attn.proj.load_state_dict(
                reference_layer.self_attn.out_proj.state_dict()
            )
            layer.mlp.fc1.load_state_dict(reference_layer.mlp.fc1.state_dict())
            layer.mlp.fc2.load_state_dict(reference_layer.mlp.fc2.state_dict())

        projector.linear_1.load_state_dict(reference_projector.linear_1.state_dict())
        projector.linear_2.load_state_dict(reference_projector.linear_2.state_dict())
        merger.linear_1.load_state_dict(reference_projector.merge_linear_1.state_dict())
        merger.linear_2.load_state_dict(reference_projector.merge_linear_2.state_dict())

    patch_width = 3 * 2 * 2 * 2
    pixel_values = torch.randn(8, patch_width)
    grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.long)
    with torch.no_grad():
        reference_hidden_states = reference_tower(
            pixel_values,
            image_grid_thw=grid_thw,
        ).last_hidden_state.squeeze(0)
        reference_merged = reference_projector(reference_hidden_states)
        hidden_states = tower(pixel_values, grid_thw)
        projected = projector(hidden_states)
        merged = merger(projected)

    assert hidden_states.shape == (8, 32)
    assert projected.shape == (8, 128)
    assert merged.shape == (2, 128)
    torch.testing.assert_close(hidden_states, reference_hidden_states)
    torch.testing.assert_close(merged, reference_merged)


def test_minimax_m3_language_only_keeps_vision_modules_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _tp4_mapping()
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 2048)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)

    with torch.device("meta"):
        model = MiniMaxM3SparseForConditionalGeneration(
            _tiny_config(),
            mapping,
            quant_config=None,
            is_multimodal_active=False,
        )

    assert model.vision_tower is None
    assert model.multimodal_embedder is None

    with pytest.raises(RuntimeError, match="vision tower is disabled"):
        model.get_image_feature([])


def _msa_server_args(
    *,
    block_size: int = 128,
    kv_cache_dtype: str = "auto",
    attention_backend: str | None = "mha",
):
    return SimpleNamespace(
        device="cuda",
        attention_backend=attention_backend,
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


def _msa_model_config(sparse_layer_type: str = "indexed_sparse"):
    text_config = SimpleNamespace(
        layer_types=(
            "full_attention",
            "full_attention",
            "full_attention",
            sparse_layer_type,
        ),
        index_head_dim=128,
        index_n_heads=4,
        index_block_size=128,
        index_topk_blocks=16,
        index_init_blocks=0,
        index_local_blocks=1,
    )
    model_config = SimpleNamespace(
        attention_arch=AttentionArch.MSA,
        context_len=1048576,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(
            runtime_attention_layer_type=sparse_layer_type,
        ),
        hf_text_config=text_config,
    )
    return model_config


def test_msa_cache_contract_uses_declared_layer_type() -> None:
    model_config = _msa_model_config()
    config = _create_attn_config(_msa_server_args(), model_config)

    assert isinstance(config, MinimaxSparseConfig)
    assert MinimaxSparseConfig.__bases__ == (BaseAttnConfig,)
    assert config.page_size == 128
    assert config.kv_cache_dtype == torch.bfloat16
    assert config.index_head_dim == 128
    assert config.compute_layer_types == (
        "full_attention",
        "full_attention",
        "full_attention",
        "indexed_sparse",
    )
    assert config.layer_types == ()
    assert config.cache_cell_size() == 576

    pool = config.create_pool(
        num_layers=4,
        max_total_num_tokens=128,
        rank=0,
        enable_memory_saver=False,
    )
    assert isinstance(pool, MinimaxSparseKVPool)
    assert isinstance(pool, MHATokenToKVPool)
    assert tuple(pool.index_k_buffer) == (3,)
    assert pool.get_index_k_buffer(3) is pool.index_k_buffer[3]
    with pytest.raises(RuntimeError, match="Layer 0"):
        pool.get_index_k_buffer(0)
    assert not pool.supports_hierarchical_kv_cache
    index_cache = pool.get_index_k_buffer(3)
    key_cache, value_cache = pool.get_kv_buffer(3)
    index_cache[1].fill_(1)
    key_cache[1].fill_(2)
    value_cache[1].fill_(3)
    pool.move_kv_cache(
        torch.tensor([2], device=index_cache.device),
        torch.tensor([1], device=index_cache.device),
    )
    torch.testing.assert_close(index_cache[2], index_cache[1])
    torch.testing.assert_close(key_cache[2], key_cache[1])
    torch.testing.assert_close(value_cache[2], value_cache[1])
    key_bytes, _ = pool.get_kv_size_bytes()
    assert (
        key_bytes == sum(cache.nbytes for cache in pool.k_buffer) + index_cache.nbytes
    )
    pool.clear_kv_buffers()
    assert not index_cache.any()
    assert not key_cache.any()
    assert not value_cache.any()

    hybrid_backend = _create_attn_backend(AttentionArch.MSA, config)
    sparse_backend = hybrid_backend.sparse_attn_backend
    assert isinstance(hybrid_backend, MinimaxHybridAttnBackend)
    assert config.backend_name == "minimax_sparse"
    assert config.full_attn_backend_name == "mha"
    assert sparse_backend.sparse_layer_ids == frozenset({3})
    assert sparse_backend.kernel_solution is None
    assert MinimaxSparseAttnBackend.__bases__ == (
        FlatCacheGroupsMixin,
        AttentionBackend,
    )
    assert not issubclass(MinimaxSparseAttnBackend, MHAAttnBackend)
    assert not hasattr(minimax_sparse_backend_module, "_MSA_BLOCK_SIZE")
    assert not hasattr(minimax_sparse_backend_module, "_MSA_HEAD_DIM")
    assert not hasattr(minimax_sparse_backend_module, "_MSA_TOPK")

    fa4_config = _create_attn_config(
        _msa_server_args(attention_backend="fa4"), model_config
    )
    fa4_backend = _create_attn_backend(AttentionArch.MSA, fa4_config)
    assert fa4_config.backend_name == "minimax_sparse"
    assert fa4_config.full_attn_backend_name == "fa4"
    assert fa4_backend.full_attn_backend.kernel_solution == "fa4"

    default_args = _msa_server_args(block_size=64, attention_backend=None)
    default_config = _create_attn_config(default_args, _msa_model_config())
    assert isinstance(default_config, MinimaxSparseConfig)
    assert default_args.block_size == 64
    assert default_config.page_size == 64
    assert default_args.attention_backend is None

    custom_page_config = MinimaxSparseConfig.generate(
        _msa_server_args(block_size=32),
        _msa_model_config(),
    )
    assert custom_page_config.page_size == 32

    fp8_config = MinimaxSparseConfig.generate(
        _msa_server_args(kv_cache_dtype="fp8"),
        _msa_model_config(),
    )
    assert fp8_config.kv_cache_dtype != torch.bfloat16

    dense_model_config = _msa_model_config()
    dense_model_config.hf_text_config.layer_types = ("full_attention",) * 4
    dense_config = MinimaxSparseConfig.generate(_msa_server_args(), dense_model_config)
    assert dense_config.sparse_layer_ids == frozenset()
    assert dense_config.cache_cell_size() == 512

    wrong_index_block_config = _msa_model_config()
    wrong_index_block_config.hf_text_config.index_block_size = 64
    custom_block_config = MinimaxSparseConfig.generate(
        _msa_server_args(block_size=64),
        wrong_index_block_config,
    )
    assert custom_block_config.page_size == 64
    assert custom_block_config.index_block_size == 64


def test_plain_mha_does_not_infer_sparse_attention_from_legacy_fields() -> None:
    model_config = SimpleNamespace(
        context_len=4096,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        # This derived field used to act as an implicit MiniMax-M3 sentinel.
        index_head_dim=128,
        hf_config=SimpleNamespace(layer_types=("full_attention",)),
    )
    config = MHAConfig.generate(_msa_server_args(), model_config)
    assert not hasattr(config, "index_head_dim")
    assert config.layer_types == ("full_attention",)
    assert isinstance(_create_attn_backend(AttentionArch.MHA, config), MHAAttnBackend)


class _FakeM3Indexer(nn.Module):
    head_dim = 2

    def __init__(self) -> None:
        super().__init__()

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        del positions
        tokens = hidden_states.shape[0]
        return (
            hidden_states.new_zeros((tokens, 1, self.head_dim)),
            hidden_states.new_zeros((tokens, self.head_dim)),
        )


class _FakeIndexedPool:
    def __init__(self) -> None:
        self.saved_loc = None
        self.key = torch.zeros((128, 1, 2))
        self.value = torch.zeros_like(self.key)
        self.index_key = torch.zeros((128, 2))

    def set_kv_buffer(self, layer, loc, k, v, k_scale, v_scale) -> None:
        del layer, k, v, k_scale, v_scale
        self.saved_loc = loc

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self.key

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self.value

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self.index_key


class _IdentityProjection:
    def __call__(self, value: torch.Tensor):
        return value, None


def _fake_sparse_layer() -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=0,
        group_id="full_attention",
        k_scale=None,
        v_scale=None,
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=2,
        v_head_dim=2,
        scaling=2**-0.5,
    )


def _fake_sparse_backend(
    forward_mode: ForwardMode,
    metadata: MHADecodeMetadata | MHAExtendMetadata,
) -> MinimaxSparseAttnBackend:
    backend = MinimaxSparseAttnBackend.__new__(MinimaxSparseAttnBackend)
    backend.forward_decode_metadata = (
        metadata if forward_mode == ForwardMode.DECODE else None
    )
    backend.forward_extend_metadata = (
        metadata if forward_mode == ForwardMode.EXTEND else None
    )
    backend.page_size = 128
    backend.max_context_len = 4096
    backend.index_topk_blocks = 16
    backend.index_head_dim = 2
    backend.index_init_blocks = 0
    backend.index_local_blocks = 1
    backend.kernel_solution = "triton"
    backend.kv_cache_dtype = torch.float32
    return backend


def _fake_sparse_metadata(
    forward_mode: ForwardMode,
    *,
    page_table: torch.Tensor | None,
    page_tables: dict[str, torch.Tensor] | None,
    out_cache_locs: dict[str, torch.Tensor] | None,
):
    seq_lens = torch.tensor([1], dtype=torch.int32)
    if forward_mode == ForwardMode.DECODE:
        return MHADecodeMetadata(
            page_table=page_table,
            seq_lens=seq_lens,
            page_tables=page_tables,
            out_cache_locs=out_cache_locs,
        )
    return MHAExtendMetadata(
        page_table=page_table,
        seq_lens=seq_lens,
        extend_seq_lens=torch.tensor([1], dtype=torch.int32),
        cu_extend_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 1], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
        extend_seq_lens_cpu=[1],
        cu_extend_seq_lens_cpu=[0, 1],
        max_extend_seq_len=1,
        max_extend_prefix_len=0,
        page_tables=page_tables,
        out_cache_locs=out_cache_locs,
    )


@pytest.mark.parametrize("forward_mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
@pytest.mark.parametrize("flat_cache", [False, True])
def test_minimax_m3_sparse_routes_flat_page_table_and_write_loc(
    monkeypatch: pytest.MonkeyPatch,
    forward_mode: ForwardMode,
    flat_cache: bool,
) -> None:
    radix_table = torch.tensor([[7]], dtype=torch.int32)
    flat_table = torch.tensor([[11]], dtype=torch.int32)
    caller_loc = torch.tensor([3], dtype=torch.int32)
    flat_loc = torch.tensor([1408], dtype=torch.int32)
    metadata = _fake_sparse_metadata(
        forward_mode,
        page_table=None if flat_cache else radix_table,
        page_tables={"full_attention": flat_table} if flat_cache else None,
        out_cache_locs={"full_attention": flat_loc} if flat_cache else None,
    )
    backend = _fake_sparse_backend(forward_mode, metadata)
    pool = _FakeIndexedPool()
    calls = {}

    def fake_msa_with_kvcache(**kwargs):
        calls["index_tokens"] = kwargs["index_q"].shape[0]
        calls["index_slot"] = kwargs["slot_mapping"]
        calls["index_table"] = kwargs["page_table"]
        calls["attention_table"] = kwargs["page_table"]
        calls["max_seqlen_q"] = kwargs["max_seqlen_q"]
        calls["max_seqlen_k"] = kwargs.get("max_seqlen_k")
        calls["page_size"] = kwargs["page_size"]
        return kwargs["q"] + 1

    if forward_mode == ForwardMode.DECODE:
        monkeypatch.setattr(
            minimax_sparse_backend_module,
            "msa_decode_with_kvcache",
            fake_msa_with_kvcache,
        )
    else:
        monkeypatch.setattr(
            minimax_sparse_backend_module,
            "msa_extend_with_kvcache",
            fake_msa_with_kvcache,
        )

    total_tokens = 1 if forward_mode == ForwardMode.DECODE else 2
    forward = (
        backend.forward_decode
        if forward_mode == ForwardMode.DECODE
        else backend.forward_extend
    )
    output = forward(
        q=torch.zeros((total_tokens, 1, 2)),
        k=torch.zeros((1, 1, 2)),
        v=torch.zeros((1, 1, 2)),
        layer=_fake_sparse_layer(),
        out_cache_loc=caller_loc,
        token_to_kv_pool=pool,
        bs=1,
        index_q=torch.zeros((total_tokens, 1, 2)),
        index_k=torch.zeros((total_tokens, 2)),
    )

    expected_table = flat_table if flat_cache else radix_table
    expected_loc = flat_loc if flat_cache else caller_loc
    torch.testing.assert_close(pool.saved_loc, expected_loc)
    assert calls["index_slot"] is pool.saved_loc
    assert calls["index_table"] is expected_table
    assert calls["attention_table"] is expected_table
    if forward_mode == ForwardMode.DECODE:
        assert calls["max_seqlen_q"] == 1
        assert calls["max_seqlen_k"] == backend.max_context_len
    else:
        assert calls["max_seqlen_q"] == 1
        assert calls["max_seqlen_k"] == 1
    assert calls["page_size"] == backend.page_size
    assert calls["index_tokens"] == 1
    assert output.shape == (total_tokens, 2)
    torch.testing.assert_close(output[0], torch.ones(2))
    if total_tokens > 1:
        torch.testing.assert_close(output[1], torch.zeros(2))


class _FakeQKVProjection(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states.new_zeros((hidden_states.shape[0], 6)), None


class _IdentityRotary(nn.Module):
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        del positions
        return q, k


class _CapturingPagedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call = None

    def forward(self, q, k, v, *, ctx, out_cache_loc, **kwargs):
        self.call = (q, k, v, ctx, out_cache_loc, kwargs)
        return q.flatten(1)


def test_minimax_m3_sparse_model_delegates_to_paged_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = MiniMaxM3Attention.__new__(MiniMaxM3Attention)
    nn.Module.__init__(attention)
    attention.q_size = 2
    attention.kv_size = 2
    attention.num_heads = 1
    attention.num_kv_heads = 1
    attention.head_dim = 2
    attention.is_sparse = True
    attention.qkv_proj = _FakeQKVProjection()
    attention.q_norm = SimpleNamespace(
        gemma_weight=torch.ones(2),
        variance_epsilon=1e-6,
    )
    attention.k_norm = SimpleNamespace(gemma_weight=torch.ones(2))
    attention.rotary_emb = _IdentityRotary()
    attention.indexer = _FakeM3Indexer()
    paged_attention = _CapturingPagedAttention()
    attention.attn = paged_attention
    attention.o_proj = _IdentityProjection()
    monkeypatch.setattr(
        "tokenspeed.runtime.models.minimax_m3.qk_rmsnorm",
        lambda q, k, *args: (q, k),
    )

    ctx = SimpleNamespace()
    out_cache_loc = torch.tensor([3], dtype=torch.int32)
    output = attention(
        positions=torch.tensor([0]),
        hidden_states=torch.zeros((1, 2)),
        ctx=ctx,
        out_cache_loc=out_cache_loc,
    )

    assert paged_attention.call is not None
    _, _, _, captured_ctx, captured_loc, kwargs = paged_attention.call
    assert captured_ctx is ctx
    assert captured_loc is out_cache_loc
    assert kwargs["index_q"].shape == (1, 1, 2)
    assert kwargs["index_k"].shape == (1, 2)
    assert output.shape == (1, 2)


class _RouteBackend:
    def __init__(self, label: str) -> None:
        self.label = label
        self.device = torch.device("cpu")
        self.calls = []

    @property
    def sinks_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def forward_decode(self, *args, **kwargs):
        self.calls.append(("decode", args[3].layer_id, kwargs))
        return args[0]

    def forward_extend(self, *args, **kwargs):
        self.calls.append(("extend", args[3].layer_id, kwargs))
        return args[0]


def test_minimax_m3_hybrid_routes_by_layer_id() -> None:
    dense = _RouteBackend("dense")
    sparse = _RouteBackend("sparse")
    sparse.sparse_layer_ids = frozenset({3})
    sparse.page_size = 128
    sparse.max_num_pages = 32
    backend = MinimaxHybridAttnBackend.__new__(MinimaxHybridAttnBackend)
    backend.device = dense.device
    backend.full_attn_backend = dense
    backend.sparse_attn_backend = sparse
    backend.sparse_layer_ids = sparse.sparse_layer_ids
    q = torch.zeros((1, 1, 2))
    common = (q, q, q)

    backend.forward(
        *common,
        _fake_sparse_layer(),
        torch.tensor([0], dtype=torch.int32),
        object(),
        ForwardMode.DECODE,
        1,
    )
    sparse_layer = _fake_sparse_layer()
    sparse_layer.layer_id = 3
    backend.forward(
        *common,
        sparse_layer,
        torch.tensor([0], dtype=torch.int32),
        object(),
        ForwardMode.EXTEND,
        1,
        index_q=q,
        index_k=q[:, 0],
    )

    assert dense.calls[0][:2] == ("decode", 0)
    assert sparse.calls[0][:2] == ("extend", 3)
    assert "index_q" in sparse.calls[0][2]
    assert not backend.support_kv_cache_prewrite(ForwardMode.DECODE)
