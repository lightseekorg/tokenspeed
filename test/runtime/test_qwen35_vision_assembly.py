from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from tokenspeed.runtime.multimodal.inputs import Modality


class _Visual:
    dtype = torch.float32

    def prepare_patch_embed(self, pixel_values, grid):
        self.pixel_values = pixel_values
        self.grid = grid
        return pixel_values


@pytest.mark.parametrize(
    ("modality", "grid_field"),
    [
        (Modality.IMAGE, "image_grid_thw"),
        (Modality.VIDEO, "video_grid_thw"),
    ],
)
def test_single_item_pre_encode_reuses_feature_and_grid(modality, grid_field):
    visual = _Visual()
    model = SimpleNamespace(visual=visual)
    feature = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    item = SimpleNamespace(feature=feature, modality=modality, **{grid_field: grid})

    tokens, result_grid = Qwen3_5ForConditionalGeneration.pre_encode(model, [item])

    assert tokens.data_ptr() == feature.data_ptr()
    assert visual.pixel_values.data_ptr() == feature.data_ptr()
    assert result_grid.data_ptr() == grid.data_ptr()
    assert visual.grid.data_ptr() == grid.data_ptr()


def test_multi_item_pre_encode_preserves_concatenation():
    visual = _Visual()
    model = SimpleNamespace(visual=visual)
    first = SimpleNamespace(
        feature=torch.ones((2, 4), dtype=torch.float32),
        modality=Modality.IMAGE,
        image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.int64),
    )
    second = SimpleNamespace(
        feature=torch.full((3, 4), 2.0, dtype=torch.float32),
        modality=Modality.IMAGE,
        image_grid_thw=torch.tensor([[1, 3, 2]], dtype=torch.int64),
    )

    tokens, grid = Qwen3_5ForConditionalGeneration.pre_encode(model, [first, second])

    assert torch.equal(tokens, torch.cat([first.feature, second.feature], dim=0))
    assert torch.equal(
        grid,
        torch.cat([first.image_grid_thw, second.image_grid_thw], dim=0),
    )


def test_single_encoder_output_is_returned_without_concatenation():
    output = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    result = Qwen3_5ForConditionalGeneration.post_encode(None, [output], None)

    assert result is output
