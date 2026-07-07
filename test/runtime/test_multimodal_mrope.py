from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.rotary_embedding import MRotaryEmbedding
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.multimodal.mrope import (
    _qwen35_mrope_positions_from_segments,
    compute_mrope_positions,
    compute_mrope_positions_with_scalar,
    copy_expanded_mrope_delta,
)


class _VisionConfig:
    spatial_merge_size = 2


class _Qwen35Config:
    architectures = ["Qwen3_5ForConditionalGeneration"]
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    model_type = "qwen3_5"
    vision_config = _VisionConfig()


def test_mrope_expanded_copy_uses_scalar_delta_without_repeated_cache():
    out = torch.empty((3, 1), dtype=torch.int64)
    mm_input = SimpleNamespace(
        mrope_position_delta_scalar=5,
        mrope_position_delta=torch.tensor([5], dtype=torch.int64),
        mrope_position_delta_repeated_cache=None,
    )

    copy_expanded_mrope_delta(out, mm_input, sequence_length=9)

    assert torch.equal(out, torch.full((3, 1), 13, dtype=torch.int64))
    assert mm_input.mrope_position_delta_repeated_cache is None


def test_qwen35_image_fast_path_matches_generic_positions():
    config = _Qwen35Config()
    input_ids = [
        10,
        config.vision_start_token_id,
        *([config.image_token_id] * 6),
        11,
    ]
    grid = torch.tensor([[1, 4, 6]], dtype=torch.long)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(2, 7)],
        model_specific_data={"image_grid_thw": grid},
    )

    positions, delta = compute_mrope_positions(config, input_ids, [item])
    expected_positions, expected_delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=config.vision_config.spatial_merge_size,
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
        model_type=config.model_type,
        input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        image_grid_thw=grid,
        video_grid_thw=None,
    )

    assert torch.equal(positions, expected_positions.squeeze(1))
    assert torch.equal(delta, expected_delta)


def test_qwen35_split_video_fast_path_matches_generic_positions():
    config = _Qwen35Config()
    input_ids = [
        10,
        config.vision_start_token_id,
        *([config.video_token_id] * 4),
        11,
        config.vision_start_token_id,
        *([config.video_token_id] * 4),
        12,
    ]
    video_grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
    split_grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    item = MultimodalDataItem(
        modality=Modality.VIDEO,
        offsets=[(2, 5), (8, 11)],
        model_specific_data={"video_grid_thw": video_grid},
    )

    positions, delta = compute_mrope_positions(config, input_ids, [item])
    expected_positions, expected_delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=config.vision_config.spatial_merge_size,
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
        model_type=config.model_type,
        input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        image_grid_thw=None,
        video_grid_thw=split_grid,
    )

    assert torch.equal(positions, expected_positions.squeeze(1))
    assert torch.equal(delta, expected_delta)


def test_qwen35_positions_are_reused_for_the_same_layout():
    _qwen35_mrope_positions_from_segments.cache_clear()
    config = _Qwen35Config()
    input_ids = [
        10,
        config.vision_start_token_id,
        *([config.image_token_id] * 4),
        11,
    ]
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)

    def compute():
        return compute_mrope_positions_with_scalar(
            config,
            input_ids,
            [
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(2, 5)],
                    model_specific_data={"image_grid_thw": grid},
                )
            ],
        )

    first_positions, first_delta, first_scalar = compute()
    second_positions, second_delta, second_scalar = compute()

    assert first_positions is second_positions
    assert first_delta is second_delta
    assert first_scalar == second_scalar
