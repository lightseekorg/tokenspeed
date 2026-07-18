from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.configs.minimax_m3_config import (
    MiniMaxM3TextConfig,
    MiniMaxM3VisionConfig,
    MiniMaxM3VLConfig,
)
from tokenspeed.runtime.configs.model_config import (
    _ATTENTION_FAMILY_SPECS,
    _apply_attention_family_defaults,
    is_multimodal_model,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.minimax_m3 import (
    MiniMaxM3MLP,
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3SparseMoeBlock,
    _msa_score_block_upper_bound,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
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
        ),
        vision_config=MiniMaxM3VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=8,
            patch_size=2,
            num_channels=3,
            projection_dim=128,
            img_token_compression_config={
                "image_token_compression_method": "patch_merge",
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        ),
        projector_hidden_size=128,
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
    assert config.vision_config.temporal_patch_size == 2
    assert config.vision_config.spatial_merge_size == 2
    assert config.vision_config.head_dim == 80
    assert config.image_token_id == config.image_token_index == 200025
    assert config.video_token_id == config.video_token_index == 200026
    assert config.merged_hidden_size == 24576
    assert is_multimodal_model(["MiniMaxM3SparseForConditionalGeneration"])

    with pytest.raises(ValueError, match="one entry per decoder layer"):
        MiniMaxM3TextConfig(num_hidden_layers=4, moe_layer_freq=[0, 1])


def test_minimax_m3_fp8_selects_dense_cache_backend_without_env() -> None:
    spec = next(spec for spec in _ATTENTION_FAMILY_SPECS if spec.name == "MiniMax M3")

    fp8_args = SimpleNamespace(
        attention_backend=None,
        block_size=64,
        kv_cache_dtype="fp8",
    )
    _apply_attention_family_defaults(fp8_args, spec)
    assert fp8_args.attention_backend == "mha"
    assert fp8_args.block_size == 128

    bf16_args = SimpleNamespace(
        attention_backend=None,
        block_size=64,
        kv_cache_dtype="auto",
    )
    _apply_attention_family_defaults(bf16_args, spec)
    assert bf16_args.attention_backend == "triton"

    explicit_args = SimpleNamespace(
        attention_backend="flashinfer",
        block_size=128,
        kv_cache_dtype="fp8",
    )
    _apply_attention_family_defaults(explicit_args, spec)
    assert explicit_args.attention_backend == "flashinfer"


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
    indexer = model.model.layers[3].self_attn.indexer
    assert indexer.num_index_heads == 4
    assert indexer.index_q_proj.gather_output

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

    scale_loader_call = None

    def _fake_scale_loader(*args, **kwargs):
        nonlocal scale_loader_call
        scale_loader_call = (args, kwargs)
        return [(0, 0.25), (3, 0.5)]

    monkeypatch.setattr(
        "tokenspeed.runtime.models.minimax_m3.kv_cache_scales_loader",
        _fake_scale_loader,
    )
    model.load_kv_cache_scales("scales.json")
    assert scale_loader_call == (
        ("scales.json", 0, 4, 4, "minimax_m3_text"),
        {"strict": True},
    )
    assert model.model.layers[0].self_attn.attn.k_scale == 0.25
    assert model.model.layers[0].self_attn.attn.v_scale == 0.25
    assert model.model.layers[3].self_attn.attn.k_scale == 0.5
    assert model.model.layers[3].self_attn.attn.v_scale == 0.5


@pytest.mark.parametrize(
    "bad_scale",
    [0.0, -0.25, float("inf"), float("-inf"), float("nan")],
    ids=["zero", "negative", "positive-inf", "negative-inf", "nan"],
)
def test_minimax_m3_rejects_nonpositive_or_nonfinite_kv_cache_scale(
    monkeypatch: pytest.MonkeyPatch,
    bad_scale: float,
) -> None:
    class _FakeConfig:
        model_type = "minimax_m3_text"
        num_hidden_layers = 2

    attentions = [
        SimpleNamespace(k_scale=None, v_scale=None),
        SimpleNamespace(k_scale=None, v_scale=None),
    ]
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(attn=attention))
                for attention in attentions
            ]
        ),
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_rank=0, tp_size=1)),
        config=_FakeConfig(),
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.models.minimax_m3.kv_cache_scales_loader",
        lambda *_args, **_kwargs: [(0, 0.25), (1, bad_scale)],
    )

    with pytest.raises(
        ValueError,
        match=r"finite and greater than zero.*TP rank 0, layer 1",
    ):
        MiniMaxM3SparseForConditionalGeneration.load_kv_cache_scales(
            model, "scales.json"
        )

    # Validation is atomic: an invalid later layer must not apply earlier scales.
    assert all(attention.k_scale is None for attention in attentions)
    assert all(attention.v_scale is None for attention in attentions)


def test_minimax_m3_multimodal_embedder_uses_default_timing() -> None:
    expected = torch.zeros((1, 2))
    apply = Mock(return_value=(expected, {}))
    model = SimpleNamespace(
        vision_embedder=SimpleNamespace(apply=apply),
        image_encoder=lambda items: items,
        get_input_embeddings=lambda: torch.nn.Embedding(2, 2),
    )
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False),
    )
    multimodal_context = SimpleNamespace(has_extend_inputs=lambda: True)

    actual = MiniMaxM3SparseForConditionalGeneration.multimodal_input_embeds(
        model,
        torch.tensor([0]),
        ctx,
        multimodal_context,
    )

    assert actual is expected
    assert apply.call_args.kwargs["log_timing"] is False


def test_minimax_m3_builds_active_multimodal_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _tp4_mapping()
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 2048)
    config = _tiny_config()
    config.encoder_only = True

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = MiniMaxM3SparseForConditionalGeneration(
                config,
                mapping,
                quant_config=None,
                is_multimodal_active=True,
            )
    finally:
        torch.set_default_dtype(old_dtype)

    assert model.model is None
    assert model.vision_tower is not None
    assert model.multi_modal_projector is not None
    assert model.patch_merge_mlp is not None
    assert model.image_encoder is not None
    assert model.vision_tower.embeddings.patch_embedding.weight.dtype is torch.float32
    assert model.vision_tower.dtype is torch.bfloat16

    wrappers = model.make_encoder_cudagraph_wrappers(mapping)
    assert set(wrappers) == {"image_encoder"}
    assert wrappers["image_encoder"].adapter.out_div == 1
    assert wrappers["image_encoder"].adapter.merge == 2
    assert wrappers["image_encoder"].adapter.dtype == model.vision_tower.dtype
    assert wrappers["image_encoder"].encoder_output_token_budgets == [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        2304,
    ]
    assert wrappers["image_encoder"].max_batch_size == 10

    loaded = model.load_weights(
        [
            (
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                torch.empty(32, 3, 2, 2, 2, device="meta"),
            ),
            (
                "vision_tower.vision_model.pre_layrnorm.weight",
                torch.empty(32, device="meta"),
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
                torch.empty(32, 32, device="meta"),
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.bias",
                torch.empty(32, device="meta"),
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight",
                torch.empty(32, 32, device="meta"),
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
                torch.empty(64, 32, device="meta"),
            ),
            (
                "multi_modal_projector.linear_1.weight",
                torch.empty(128, 32, device="meta"),
            ),
            (
                "patch_merge_mlp.linear_1.weight",
                torch.empty(128, 512, device="meta"),
            ),
        ]
    )
    assert loaded == {
        "vision_tower.embeddings.patch_embedding.weight",
        "vision_tower.pre_layrnorm.weight",
        "vision_tower.layers.0.self_attn.qkv_proj.weight",
        "vision_tower.layers.0.self_attn.qkv_proj.bias",
        "vision_tower.layers.0.self_attn.proj.weight",
        "vision_tower.layers.0.mlp.fc1.weight",
        "multi_modal_projector.linear_1.weight",
        "patch_merge_mlp.linear_1.weight",
    }

    monkeypatch.setattr(
        "tokenspeed.runtime.models.minimax_m3.kv_cache_scales_loader",
        lambda *_args: pytest.fail("encoder-only model must not load KV scales"),
    )
    model.load_kv_cache_scales("unused.json")


def test_minimax_m3_image_encoder_enforces_grid_and_placeholder_contract() -> None:
    config = _tiny_config()
    config.encoder_only = True
    config.vision_config.num_hidden_layers = 0
    model = MiniMaxM3SparseForConditionalGeneration(
        config,
        Mapping(rank=0, world_size=1),
        is_multimodal_active=True,
        mm_attention_backend="triton_attn",
    ).eval()
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.randn(4, 3 * 2 * 2 * 2),
        offsets=[(3, 3)],
        model_specific_data={
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64)
        },
    )

    with torch.inference_mode():
        output = model.get_image_feature([item])
    assert output.shape == (1, config.text_config.hidden_size)

    item.offsets = [(3, 4)]
    with pytest.raises(ValueError, match="placeholder lengths"):
        model.get_image_feature([item])

    item.model_specific_data["image_grid_thw"] = torch.tensor([[1.0, 2.0, 2.0]])
    with pytest.raises(TypeError, match="integer dtype"):
        model.get_image_feature([item])

    item.modality = Modality.VIDEO
    with pytest.raises(ValueError, match="accepts image items only"):
        model.get_image_feature([item])


def _msa_server_args(
    *,
    block_size: int = 128,
    kv_cache_dtype: str = "auto",
    kv_cache_quant_method: str = "none",
    attention_backend: str = "mha",
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
        kv_cache_quant_method=kv_cache_quant_method,
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


def test_minimax_m3_mha_cache_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tokenspeed.runtime.layers.attention.configs.mha.current_platform",
        lambda: SimpleNamespace(is_blackwell=True),
    )
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
    fp8_config = MHAConfig.generate(
        _msa_server_args(kv_cache_dtype="fp8"),
        _msa_model_config(),
    )
    assert fp8_config.kv_cache_dtype == torch.float8_e4m3fn
    assert fp8_config.cache_cell_size() == 384

    with pytest.raises(ValueError, match="static-scale"):
        MHAConfig.generate(
            _msa_server_args(
                kv_cache_dtype="fp8",
                kv_cache_quant_method="per_token_head",
            ),
            _msa_model_config(),
        )
    for unsupported_backend in ("fa3", "fa4", "triton", "trtllm"):
        with pytest.raises(ValueError, match="automatic MHA backend"):
            MHAConfig.generate(
                _msa_server_args(
                    kv_cache_dtype="fp8",
                    attention_backend=unsupported_backend,
                ),
                _msa_model_config(),
            )

    for supported_backend in ("mha", "flashinfer"):
        MHAConfig.generate(
            _msa_server_args(
                kv_cache_dtype="fp8",
                attention_backend=supported_backend,
            ),
            _msa_model_config(),
        )

    monkeypatch.setattr(
        "tokenspeed.runtime.layers.attention.configs.mha.current_platform",
        lambda: SimpleNamespace(is_blackwell=False),
    )
    with pytest.raises(ValueError, match="requires NVIDIA Blackwell"):
        MHAConfig.generate(
            _msa_server_args(kv_cache_dtype="fp8"),
            _msa_model_config(),
        )


def test_msa_score_block_upper_bound_clamps_skewed_mixed_batch() -> None:
    # Request A contributes max_prefix_len and request B max_extend_len. Their
    # sum can exceed the table even though each request individually fits.
    assert _msa_score_block_upper_bound(32700, 8192, 128, 256) == 256
    assert _msa_score_block_upper_bound(1024, 512, 128, 256) == 12
