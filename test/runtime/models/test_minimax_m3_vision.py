"""CPU/meta tests for the MiniMax-M3 vision tower contract."""

from __future__ import annotations

import pytest
import torch

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
    apply_minimax_m3_vision_rotary,
)


def _tiny_text_config(*, hidden_size: int = 12) -> MiniMaxM3TextConfig:
    return MiniMaxM3TextConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=16,
        dense_intermediate_size=24,
        shared_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=[0],
        dtype="float32",
    )


def _tiny_vision_config(*, num_hidden_layers: int = 0) -> MiniMaxM3VisionConfig:
    return MiniMaxM3VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
        num_channels=3,
        projection_dim=12,
        rope_theta=10_000.0,
        img_token_compression_config={
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        dtype="float32",
    )


def _tiny_vl_config(*, num_hidden_layers: int = 0) -> MiniMaxM3VLConfig:
    return MiniMaxM3VLConfig(
        text_config=_tiny_text_config(),
        vision_config=_tiny_vision_config(num_hidden_layers=num_hidden_layers),
        projector_hidden_size=16,
        dtype="float32",
    )


def _tp1_mapping() -> Mapping:
    return Mapping(rank=0, world_size=1)


def test_nested_patch_merge_config_is_normalized() -> None:
    config = MiniMaxM3VLConfig(
        text_config=_tiny_text_config(),
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 0,
            "num_attention_heads": 2,
            "patch_size": 2,
            "num_channels": 3,
            "img_token_compression_config": {
                "image_token_compression_method": "patch_merge",
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        },
        projector_hidden_size=16,
    )

    assert isinstance(config.vision_config, MiniMaxM3VisionConfig)
    assert config.vision_config.head_dim == 8
    assert config.vision_config.spatial_merge_size == 2
    assert config.vision_config.temporal_patch_size == 2
    assert config.img_token_compression_config == (
        config.vision_config.img_token_compression_config
    )
    assert config.merged_hidden_size == 4 * config.text_config.hidden_size

    with pytest.raises(ValueError, match="spatial_merge_size must match"):
        MiniMaxM3VLConfig(
            text_config=_tiny_text_config(),
            vision_config=_tiny_vision_config(),
            img_token_compression_config={
                "image_token_compression_method": "patch_merge",
                "spatial_merge_size": 4,
                "temporal_patch_size": 2,
            },
        )


def test_tiny_meta_parameter_shapes_match_checkpoint_layout() -> None:
    config = _tiny_vl_config()
    mapping = _tp1_mapping()

    with torch.device("meta"):
        tower = MiniMaxM3VisionTower(config.vision_config, mapping=mapping)
        projector = MiniMaxM3MultiModalProjector(config, mapping=mapping)
        merger = MiniMaxM3PatchMergeMLP(config, mapping=mapping)

    assert tower.embeddings.patch_embedding.weight.shape == (16, 3, 2, 2, 2)
    assert len(tower.layers) == 0
    assert projector.linear_1.weight.shape == (16, 16)
    assert projector.linear_2.weight.shape == (12, 16)
    assert merger.linear_1.weight.shape == (16, 4 * 12)
    assert merger.linear_2.weight.shape == (12, 16)


def test_meta_patch_embedding_keeps_checkpoint_float32_dtype() -> None:
    config = _tiny_vision_config(num_hidden_layers=1)
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            tower = MiniMaxM3VisionTower(config, mapping=_tp1_mapping())
    finally:
        torch.set_default_dtype(default_dtype)

    assert tower.embeddings.patch_embedding.weight.dtype is torch.float32
    assert not tower.embeddings.patch_embedding.enable_linear
    assert tower.pre_layrnorm.weight.dtype is torch.bfloat16
    block_parameters = list(tower.layers.parameters())
    assert block_parameters
    assert all(parameter.dtype is torch.bfloat16 for parameter in block_parameters)


def test_partial_3d_rotary_preserves_the_two_unrotated_tail_dims() -> None:
    # head_dim=8 gives axis_dim=2 for T/H/W: six rotary dimensions and a
    # two-dimensional pass-through tail.
    query = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
    key = query + 100
    cos = torch.zeros(2, 6)
    sin = torch.ones(2, 6)

    rotated_query, rotated_key = apply_minimax_m3_vision_rotary(
        query,
        key,
        (cos, sin),
    )

    expected_query = torch.cat((-query[..., 3:6], query[..., :3]), dim=-1)
    expected_key = torch.cat((-key[..., 3:6], key[..., :3]), dim=-1)
    torch.testing.assert_close(rotated_query[..., :6], expected_query)
    torch.testing.assert_close(rotated_key[..., :6], expected_key)
    torch.testing.assert_close(rotated_query[..., 6:], query[..., 6:])
    torch.testing.assert_close(rotated_key[..., 6:], key[..., 6:])


def test_patch_embed_validates_flat_pixels_against_grid() -> None:
    tower = MiniMaxM3VisionTower(
        _tiny_vision_config(),
        mapping=_tp1_mapping(),
    ).eval()
    pixel_values = torch.randn(12, 3 * 2 * 2 * 2)
    grid_thw = torch.tensor([[1, 2, 4], [1, 2, 2]], dtype=torch.int64)

    embedded = tower.prepare_patch_embed(pixel_values, grid_thw)
    assert embedded.shape == (12, 1, 16)

    with pytest.raises(ValueError):
        tower.prepare_patch_embed(pixel_values[:, :-1], grid_thw)
    with pytest.raises(ValueError):
        tower.prepare_patch_embed(pixel_values[:-1], grid_thw)
    with pytest.raises(ValueError):
        tower.prepare_patch_embed(pixel_values, grid_thw.flatten())
    with pytest.raises(ValueError):
        tower.prepare_patch_embed(
            torch.randn(6, pixel_values.shape[1]),
            torch.tensor([[1, 3, 2]], dtype=torch.int64),
        )


def test_projector_and_patch_merge_reduce_each_consecutive_group_of_four() -> None:
    config = _tiny_vl_config()
    mapping = _tp1_mapping()
    projector = MiniMaxM3MultiModalProjector(config, mapping=mapping).eval()
    merger = MiniMaxM3PatchMergeMLP(config, mapping=mapping).eval()

    projected = projector(torch.randn(8, config.vision_config.hidden_size))
    assert projected.shape == (8, config.text_config.hidden_size)

    # Capture the exact tensor consumed by the first merger linear. This tests
    # that processor-adjacent patches [0:4], [4:8] are flattened together,
    # rather than merely checking the final row count.
    patches = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12)
    captured: dict[str, torch.Tensor] = {}

    def capture_grouped_input(_module, inputs) -> None:
        captured["input"] = inputs[0].detach().clone()

    hook = merger.linear_1.register_forward_pre_hook(capture_grouped_input)
    try:
        merged = merger(patches)
    finally:
        hook.remove()

    assert merged.shape == (2, config.text_config.hidden_size)
    torch.testing.assert_close(captured["input"], patches.reshape(2, 4 * 12))

    with pytest.raises(ValueError, match="divisible"):
        merger(patches[:6])


def test_multi_image_metadata_isolates_attention_and_matches_merge_group_rope() -> None:
    tower = MiniMaxM3VisionTower(
        _tiny_vision_config(),
        mapping=_tp1_mapping(),
    ).eval()
    grid_thw = torch.tensor([[1, 2, 2], [1, 2, 4]], dtype=torch.int64)

    metadata = tower.prepare_metadata(grid_thw)

    assert metadata["cu_seqlens"].dtype == torch.int32
    assert metadata["cu_seqlens"].tolist() == [0, 4, 12]
    assert metadata["max_seqlen"] == 8
    assert metadata["sequence_lengths"] is None

    # With head_dim=8 each axis owns one frequency. Coordinates follow the
    # processor's 2x2 merge-group ordering and restart for the second image.
    coordinates = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 2],
            [0, 1, 3],
        ],
        dtype=torch.float32,
    )
    expected_angles = coordinates.repeat(1, 2)
    cos, sin = metadata["position_embeddings"]
    torch.testing.assert_close(cos, expected_angles.cos())
    torch.testing.assert_close(sin, expected_angles.sin())
