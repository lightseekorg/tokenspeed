import torch

from tokenspeed.runtime.layers.rotary_embedding import MRotaryEmbedding
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.multimodal.mrope import compute_mrope_positions


class _VisionConfig:
    spatial_merge_size = 2


class _Qwen35Config:
    architectures = ["Qwen3_5ForConditionalGeneration"]
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    model_type = "qwen3_5"
    vision_config = _VisionConfig()


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


def test_qwen35_timestamped_video_matches_token_type_positions():
    config = _Qwen35Config()
    vision_end_token_id = 151653
    timestamp_token_id = 42
    input_ids = [
        config.vision_start_token_id,
        timestamp_token_id,
        *([config.video_token_id] * 4),
        vision_end_token_id,
        timestamp_token_id,
        config.vision_start_token_id,
        *([config.video_token_id] * 4),
        vision_end_token_id,
    ]
    item = MultimodalDataItem(
        modality=Modality.VIDEO,
        offsets=[(2, 5), (9, 12)],
        model_specific_data={
            "video_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long)
        },
    )

    positions, delta = compute_mrope_positions(config, input_ids, [item])

    expected = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 7, 7, 7, 9],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 9],
            [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    assert torch.equal(positions, expected)
    assert torch.equal(delta, torch.tensor([[-4]], dtype=torch.long))


def test_qwen35_mixed_items_fast_path_matches_generic_positions():
    config = _Qwen35Config()
    input_ids = [
        10,
        config.vision_start_token_id,
        *([config.image_token_id] * 4),
        11,
        config.vision_start_token_id,
        *([config.video_token_id] * 4),
        12,
        config.vision_start_token_id,
        *([config.video_token_id] * 4),
        13,
    ]
    image_grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    video_grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
    split_video_grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(2, 5)],
            model_specific_data={"image_grid_thw": image_grid},
        ),
        MultimodalDataItem(
            modality=Modality.VIDEO,
            offsets=[(8, 11), (14, 17)],
            model_specific_data={"video_grid_thw": video_grid},
        ),
    ]

    positions, delta = compute_mrope_positions(config, input_ids, items)
    expected_positions, expected_delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=config.vision_config.spatial_merge_size,
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
        model_type=config.model_type,
        input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        image_grid_thw=image_grid,
        video_grid_thw=split_video_grid,
    )

    assert torch.equal(positions, expected_positions.squeeze(1))
    assert torch.equal(delta, expected_delta)


def test_qwen35_invalid_offsets_fall_back_to_generic_positions():
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
        offsets=[(3, 8)],
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
