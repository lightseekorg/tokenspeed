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

import torch

from tokenspeed.runtime.multimodal import mrope


def test_preparation_uses_unpadded_ids_and_preserves_existing_positions():
    config = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
    )
    ids = [10, config.vision_start_token_id, *([config.image_token_id] * 6), 11]
    item = SimpleNamespace(
        modality=SimpleNamespace(name="IMAGE"),
        offsets=[(2, 7)],
        model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 6]])},
    )
    inputs = SimpleNamespace(
        mm_items=[item],
        mrope_positions=None,
        mrope_position_delta=None,
        mrope_position_delta_scalar=None,
    )
    mrope.prepare_mrope_inputs(config, ids, inputs)
    assert inputs.mrope_positions.tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5],
        [0, 1, 2, 2, 2, 3, 3, 3, 5],
        [0, 1, 2, 3, 4, 2, 3, 4, 5],
    ]
    assert inputs.mrope_position_delta.tolist() == [[-3]]
    assert inputs.mrope_position_delta_scalar == -3
    positions = inputs.mrope_positions
    mrope.prepare_mrope_inputs(config, [1, 2, 3], inputs)
    assert inputs.mrope_positions is positions


def test_non_mrope_preparation_keeps_position_overrides_absent():
    inputs = SimpleNamespace(
        mm_items=[],
        mrope_positions=None,
        mrope_position_delta=None,
        mrope_position_delta_scalar=None,
    )
    mrope.prepare_mrope_inputs(
        SimpleNamespace(architectures=["OtherModel"]), [1, 2], inputs
    )
    assert inputs.mrope_positions is None
    assert inputs.mrope_position_delta is None
    assert inputs.mrope_position_delta_scalar is None


def _mrope_tokenized_request():
    item = SimpleNamespace(
        modality=SimpleNamespace(name="IMAGE"),
        offsets=[(2, 7)],
        model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 6]])},
    )
    original = [10, 151652, *([151655] * 6), 11]
    request = SimpleNamespace(
        input_ids=[10, 151652, *([1000001] * 6), 11],
        input_ids_unpadded=original,
        multimodal_inputs=SimpleNamespace(
            mm_items=[item],
            mrope_positions=None,
            mrope_position_delta=None,
            mrope_position_delta_scalar=None,
        ),
    )
    config = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
    )
    return request, config


def test_full_multimodal_decode_computes_the_same_delta_as_prefill():
    from tokenspeed.runtime.engine.multimodal_request import (
        prepare_tokenized_multimodal_request,
    )

    request, config = _mrope_tokenized_request()
    prepare_tokenized_multimodal_request(
        request,
        hf_config=config,
        model_is_multimodal=True,
        language_model_only=False,
        disaggregation_mode="decode",
    )
    assert request.multimodal_inputs.mrope_position_delta_scalar == -3
    assert request.input_ids[2:8] == [1000001] * 6


def test_language_only_multimodal_decode_restores_original_ids_and_drops_mm():
    from tokenspeed.runtime.engine.multimodal_request import (
        prepare_tokenized_multimodal_request,
    )

    request, config = _mrope_tokenized_request()
    prepare_tokenized_multimodal_request(
        request,
        hf_config=config,
        model_is_multimodal=True,
        language_model_only=True,
        disaggregation_mode="decode",
    )
    assert request.input_ids == request.input_ids_unpadded
    assert request.multimodal_inputs is None
