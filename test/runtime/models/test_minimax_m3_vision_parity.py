"""CUDA parity coverage for the MiniMax-M3 vision pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import MiniMaxM3VLConfig as HFMiniMaxM3VLConfig
from transformers import MiniMaxM3VLTextConfig as HFMiniMaxM3VLTextConfig
from transformers import MiniMaxM3VLVisionConfig as HFMiniMaxM3VLVisionConfig
from transformers import (
    MiniMaxM3VLVisionModel,
)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLMultiModalProjector,
)

from tokenspeed.runtime.configs.minimax_m3_config import (
    MiniMaxM3TextConfig,
    MiniMaxM3VisionConfig,
    MiniMaxM3VLConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.minimax_m3_vision import (
    MiniMaxM3MultiModalProjector,
    MiniMaxM3PatchMergeMLP,
    MiniMaxM3VisionTower,
)
from tokenspeed.runtime.multimodal.encoder_cudagraph import (
    EncoderCudaGraphWrapper,
    VisionEncoderCudaGraphAdapter,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MiniMax-M3 vision parity requires CUDA and the Triton attention backend.",
)


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def _reference_configs() -> tuple[
    HFMiniMaxM3VLVisionConfig,
    HFMiniMaxM3VLConfig,
]:
    vision_config = HFMiniMaxM3VLVisionConfig(
        hidden_size=80,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_channels=3,
        image_size=8,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        dtype=torch.bfloat16,
    )
    text_config = HFMiniMaxM3VLTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=16,
        dense_intermediate_size=64,
        shared_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        num_local_experts=1,
        num_experts_per_tok=1,
        mlp_layer_types=["dense"],
        layer_types=["full_attention"],
        dtype=torch.bfloat16,
    )
    config = HFMiniMaxM3VLConfig(
        vision_config=vision_config,
        text_config=text_config,
        projector_hidden_size=64,
        dtype=torch.bfloat16,
    )
    return vision_config, config


def _runtime_config() -> MiniMaxM3VLConfig:
    vision_config = MiniMaxM3VisionConfig(
        hidden_size=80,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_channels=3,
        image_size=8,
        patch_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        rope_theta=10_000.0,
        img_token_compression_config={
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        dtype=torch.bfloat16,
    )
    text_config = MiniMaxM3TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=16,
        dense_intermediate_size=64,
        shared_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        num_local_experts=1,
        num_experts_per_tok=1,
        moe_layer_freq=[0],
        dtype=torch.bfloat16,
    )
    return MiniMaxM3VLConfig(
        vision_config=vision_config,
        text_config=text_config,
        projector_hidden_size=64,
        dtype=torch.bfloat16,
    )


def _copy_parameter(destination: torch.nn.Parameter, source: torch.Tensor) -> None:
    with torch.no_grad():
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def _load_reference_tower(
    runtime_tower: MiniMaxM3VisionTower,
    reference_tower: MiniMaxM3VLVisionModel,
) -> None:
    """Map native split attention weights into the runtime fused QKV layout."""
    source = reference_tower.state_dict()
    destination = dict(runtime_tower.named_parameters())
    direct_mapping = {
        "embeddings.patch_embedding.weight": "embeddings.proj.weight",
        "pre_layrnorm.weight": "pre_layrnorm.weight",
        "pre_layrnorm.bias": "pre_layrnorm.bias",
        "layers.0.layer_norm1.weight": "layers.0.layer_norm1.weight",
        "layers.0.layer_norm1.bias": "layers.0.layer_norm1.bias",
        "layers.0.self_attn.proj.weight": "layers.0.self_attn.out_proj.weight",
        "layers.0.self_attn.proj.bias": "layers.0.self_attn.out_proj.bias",
        "layers.0.layer_norm2.weight": "layers.0.layer_norm2.weight",
        "layers.0.layer_norm2.bias": "layers.0.layer_norm2.bias",
        "layers.0.mlp.fc1.weight": "layers.0.mlp.fc1.weight",
        "layers.0.mlp.fc1.bias": "layers.0.mlp.fc1.bias",
        "layers.0.mlp.fc2.weight": "layers.0.mlp.fc2.weight",
        "layers.0.mlp.fc2.bias": "layers.0.mlp.fc2.bias",
    }
    for destination_name, source_name in direct_mapping.items():
        _copy_parameter(destination[destination_name], source[source_name])

    qkv_weight = destination["layers.0.self_attn.qkv_proj.weight"]
    qkv_bias = destination["layers.0.self_attn.qkv_proj.bias"]
    for projection_name, shard_id in (
        ("q_proj", "q"),
        ("k_proj", "k"),
        ("v_proj", "v"),
    ):
        qkv_weight.weight_loader(
            qkv_weight,
            source[f"layers.0.self_attn.{projection_name}.weight"],
            shard_id,
        )
        qkv_bias.weight_loader(
            qkv_bias,
            source[f"layers.0.self_attn.{projection_name}.bias"],
            shard_id,
        )

    expected_parameters = set(direct_mapping) | {
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.qkv_proj.bias",
    }
    assert set(destination) == expected_parameters


def _load_reference_projectors(
    runtime_projector: MiniMaxM3MultiModalProjector,
    runtime_merger: MiniMaxM3PatchMergeMLP,
    reference_projector: MiniMaxM3VLMultiModalProjector,
) -> None:
    """Split the native combined projector across the two checkpoint modules."""
    source = reference_projector.state_dict()
    projector_parameters = dict(runtime_projector.named_parameters())
    merger_parameters = dict(runtime_merger.named_parameters())

    for parameter_name, destination in projector_parameters.items():
        _copy_parameter(destination, source[parameter_name])
    for parameter_name, destination in merger_parameters.items():
        reference_name = parameter_name.replace("linear_", "merge_linear_")
        _copy_parameter(destination, source[reference_name])

    assert set(projector_parameters) == {
        "linear_1.weight",
        "linear_1.bias",
        "linear_2.weight",
        "linear_2.bias",
    }
    assert set(merger_parameters) == set(projector_parameters)


@torch.inference_mode()
def test_minimax_m3_tiny_multi_image_pipeline_matches_independent_references() -> None:
    torch.manual_seed(1234)
    device = torch.device("cuda", torch.cuda.current_device())
    reference_vision_config, reference_config = _reference_configs()
    runtime_config = _runtime_config()

    with torch.device(device):
        reference_tower = MiniMaxM3VLVisionModel(reference_vision_config)
        reference_projector = MiniMaxM3VLMultiModalProjector(reference_config)
    reference_tower.to(dtype=torch.bfloat16)
    reference_tower.embeddings.proj.to(dtype=torch.float32)
    reference_projector.to(dtype=torch.bfloat16)

    with _default_dtype(torch.bfloat16), torch.device(device):
        runtime_tower = MiniMaxM3VisionTower(
            runtime_config.vision_config,
            mapping=Mapping(rank=0, world_size=1),
            mm_attention_backend="triton_attn",
        )
        runtime_projector = MiniMaxM3MultiModalProjector(
            runtime_config,
            mapping=Mapping(rank=0, world_size=1),
        )
        runtime_merger = MiniMaxM3PatchMergeMLP(
            runtime_config,
            mapping=Mapping(rank=0, world_size=1),
        )

    reference_tower.eval()
    reference_projector.eval()
    runtime_tower.eval()
    runtime_projector.eval()
    runtime_merger.eval()
    _load_reference_tower(runtime_tower, reference_tower)
    _load_reference_projectors(
        runtime_projector,
        runtime_merger,
        reference_projector,
    )

    assert reference_tower.embeddings.proj.weight.dtype is torch.float32
    assert runtime_tower.embeddings.patch_embedding.weight.dtype is torch.float32
    assert reference_tower.pre_layrnorm.weight.dtype is torch.bfloat16
    assert runtime_tower.pre_layrnorm.weight.dtype is torch.bfloat16
    assert all(
        parameter.dtype is torch.bfloat16
        for parameter in reference_tower.layers.parameters()
    )
    assert all(
        parameter.dtype is torch.bfloat16
        for parameter in runtime_tower.layers.parameters()
    )
    assert all(
        parameter.dtype is torch.bfloat16
        for parameter in runtime_projector.parameters()
    )
    assert all(
        parameter.dtype is torch.bfloat16 for parameter in runtime_merger.parameters()
    )

    grid_thw = torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.int64, device=device)
    pixel_values = torch.randn(24, 3 * 2 * 2 * 2, dtype=torch.float32, device=device)

    reference_patches = reference_tower.embeddings(pixel_values)
    runtime_patches = runtime_tower.embeddings(pixel_values)
    torch.testing.assert_close(
        runtime_patches,
        reference_patches,
        rtol=1e-4,
        atol=1e-4,
    )

    # TokenSpeed can pack images from unrelated requests into one encoder call,
    # so each grid is an independent varlen sequence. Compare that packed call
    # with one native Transformers invocation per image; a single native call
    # concatenates all patches into one attention sequence and is not a safe
    # batching reference for a serving runtime.
    reference_features = torch.cat(
        [
            reference_tower(
                pixel_values=pixel_values[:16],
                image_grid_thw=grid_thw[:1],
            ).last_hidden_state.squeeze(0),
            reference_tower(
                pixel_values=pixel_values[16:],
                image_grid_thw=grid_thw[1:],
            ).last_hidden_state.squeeze(0),
        ]
    )
    runtime_tokens = runtime_tower.prepare_patch_embed(pixel_values, grid_thw)
    runtime_features = runtime_tower.forward_blocks(
        runtime_tokens,
        runtime_tower.prepare_metadata(grid_thw),
    ).squeeze(1)
    torch.testing.assert_close(
        runtime_features,
        reference_features,
        rtol=2e-2,
        atol=2e-2,
    )

    reference_projected = reference_projector.linear_2(
        reference_projector.act(reference_projector.linear_1(reference_features))
    )
    runtime_projected = runtime_projector(runtime_features)
    torch.testing.assert_close(
        runtime_projected,
        reference_projected,
        rtol=2e-2,
        atol=2e-2,
    )

    reference_merged = reference_projector(reference_features)
    runtime_merged = runtime_merger(runtime_projected)
    assert reference_merged.shape == runtime_merged.shape == (6, 32)
    torch.testing.assert_close(
        runtime_merged,
        reference_merged,
        rtol=2e-2,
        atol=2e-2,
    )


@torch.inference_mode()
def test_minimax_m3_encoder_cudagraph_replays_dynamic_3d_rope() -> None:
    """Replay one graph across aspect ratios and packed image orderings."""

    torch.manual_seed(20260715)
    device = torch.device("cuda", torch.cuda.current_device())
    config = _runtime_config()
    with _default_dtype(torch.bfloat16), torch.device(device):
        tower = MiniMaxM3VisionTower(
            config.vision_config,
            mapping=Mapping(rank=0, world_size=1),
            mm_attention_backend="triton_attn",
        ).eval()

    # Runtime parameters are allocated with ``torch.empty`` because production
    # immediately loads checkpoint weights. Give this standalone graph test
    # deterministic finite weights instead of depending on allocator contents
    # left by an earlier CUDA test.
    for parameter in tower.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    for module in tower.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    items = [
        SimpleNamespace(
            tokens=torch.randn(
                16,
                1,
                tower.hidden_size,
                dtype=tower.dtype,
                device=device,
            ),
            grid=torch.tensor([[1, 4, 4]], dtype=torch.int64, device=device),
        ),
        SimpleNamespace(
            tokens=torch.randn(
                8,
                1,
                tower.hidden_size,
                dtype=tower.dtype,
                device=device,
            ),
            grid=torch.tensor([[1, 2, 4]], dtype=torch.int64, device=device),
        ),
    ]

    def pre_encode(selected_items):
        return (
            torch.cat([item.tokens for item in selected_items]),
            torch.cat([item.grid for item in selected_items]),
        )

    def post_encode(encoder_outs, _grid):
        return torch.cat(encoder_outs).squeeze(1)

    wrapper = EncoderCudaGraphWrapper(
        adapter=VisionEncoderCudaGraphAdapter(
            tower=tower,
            pre_encode=pre_encode,
            post_encode=post_encode,
            out_div=1,
            merge=tower.spatial_merge_size,
            input_feature_shape=(1, tower.hidden_size),
            modality_name="image",
            input_dtype=tower.dtype,
        ),
        # One 24-patch graph deliberately replays both a 16+8 packed batch and
        # the opposite ordering. Its synthetic capture grid is neither real
        # grid, which makes stale capture-time RoPE immediately observable.
        budget_range=(24, 24),
        max_batch_size=2,
    )

    def eager(selected_items):
        tokens, grid = pre_encode(selected_items)
        return tower.forward_blocks(tokens, tower.prepare_metadata(grid)).squeeze(1)

    expected = eager(items)
    actual = wrapper(items)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert set(wrapper.budget_graphs) == {24}

    reversed_items = list(reversed(items))
    expected_reversed = eager(reversed_items)
    actual_reversed = wrapper(reversed_items)
    torch.testing.assert_close(
        actual_reversed,
        expected_reversed,
        rtol=2e-2,
        atol=2e-2,
    )
    assert set(wrapper.budget_graphs) == {24}
