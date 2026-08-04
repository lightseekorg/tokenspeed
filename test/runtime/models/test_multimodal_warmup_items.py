# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.models.kimi_k25 import KimiK25ForConditionalGeneration
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from tokenspeed.runtime.multimodal.inputs import Modality


def test_kimi_encoder_warmup_item_matches_native_contract():
    model = SimpleNamespace(
        vision_tower=SimpleNamespace(
            merge_kernel_size=(2, 2),
            patch_embed=SimpleNamespace(
                patch_size=(14, 16),
                proj=SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16)),
            ),
        ),
    )

    batches = KimiK25ForConditionalGeneration.make_encoder_warmup_items(model, 4)
    item = batches[Modality.IMAGE][0]

    assert item.feature.shape == (16, 3, 14, 16)
    assert item.feature.dtype == torch.bfloat16
    assert item.grid_thws.tolist() == [[1, 4, 4]]


def test_qwen_encoder_warmup_items_cover_image_and_video_contracts():
    patch_embed = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=16,
    )
    model = SimpleNamespace(
        visual=SimpleNamespace(
            spatial_merge_size=2,
            patch_embed=patch_embed,
            dtype=torch.bfloat16,
        )
    )

    batches = Qwen3_5ForConditionalGeneration.make_encoder_warmup_items(model, 4)
    image = batches[Modality.IMAGE][0]
    video = batches[Modality.VIDEO][0]

    assert image.feature.shape == (16, 1536)
    assert image.image_grid_thw.tolist() == [[1, 4, 4]]
    assert video.feature.shape == (32, 1536)
    assert video.video_grid_thw.tolist() == [[2, 4, 4]]


@pytest.mark.parametrize(
    ("model_cls", "model"),
    [
        (
            KimiK25ForConditionalGeneration,
            SimpleNamespace(vision_tower=SimpleNamespace(merge_kernel_size=(2, 2))),
        ),
        (
            Qwen3_5ForConditionalGeneration,
            SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2)),
        ),
    ],
)
def test_encoder_warmup_items_reject_invalid_spatial_grid(model_cls, model):
    with pytest.raises(ValueError, match="divisible"):
        model_cls.make_encoder_warmup_items(model, 3)
