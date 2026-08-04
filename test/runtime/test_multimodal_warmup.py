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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
from tokenspeed.runtime.models.moonvit import MoonViTVisionPath
from tokenspeed.runtime.models.qwen3_vision import Qwen3VLMoeVisionModel
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    warmup_multimodal_encoders,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.utils.env import envs


class _WarmupModel:
    is_multimodal_active = True

    def __init__(self):
        self.mapping = object()
        self.calls = []
        self.wrapper_builds = 0
        self.wrapper_calls = 0
        self.image_encoder = self._image_encoder
        self.video_encoder = self._video_encoder
        self.audio_encoder = self._audio_encoder

    @staticmethod
    def _image_warmup_items():
        return [MultimodalDataItem(modality=Modality.IMAGE, feature=torch.zeros(1))]

    @staticmethod
    def _audio_warmup_items():
        return [MultimodalDataItem(modality=Modality.AUDIO, feature=torch.zeros(2))]

    def get_multimodal_encoder_specs(self):
        return {
            Modality.IMAGE: EncoderSpec(
                self.image_encoder,
                make_warmup_items=self._image_warmup_items,
            ),
            # A registered modality without a factory is intentionally skipped.
            Modality.VIDEO: EncoderSpec(self.video_encoder),
            Modality.AUDIO: EncoderSpec(
                self.audio_encoder,
                make_warmup_items=self._audio_warmup_items,
            ),
        }

    def _image_encoder(self, items):
        self.calls.append((Modality.IMAGE, items, torch.is_inference_mode_enabled()))
        return torch.zeros(1)

    def _video_encoder(self, items):
        self.calls.append((Modality.VIDEO, items, torch.is_inference_mode_enabled()))
        return torch.zeros(1)

    def _audio_encoder(self, items):
        self.calls.append((Modality.AUDIO, items, torch.is_inference_mode_enabled()))
        return torch.zeros(1)

    def _wrapped_encoder(self, items):
        self.wrapper_calls += 1
        return self._image_encoder(items)

    def make_encoder_cudagraph_wrappers(self, mapping):
        assert mapping is self.mapping
        self.wrapper_builds += 1
        wrapper = self._wrapped_encoder
        return {"image_encoder": wrapper, "video_encoder": wrapper}


def _set_graph_env(monkeypatch, *, graph_enabled=False):
    monkeypatch.setattr(
        envs.TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH,
        "get",
        lambda: graph_enabled,
    )


def test_warmup_dispatches_registered_factories_in_inference_mode():
    model = _WarmupModel()

    warmup_multimodal_encoders(model, device=torch.device("cpu"))

    assert [call[0] for call in model.calls] == [Modality.IMAGE, Modality.AUDIO]
    assert all(call[2] for call in model.calls)


def test_warmup_skips_inactive_multimodal_model():
    model = _WarmupModel()
    model.is_multimodal_active = False

    warmup_multimodal_encoders(model, device=torch.device("cpu"))

    assert model.calls == []


def test_warmup_skips_model_without_encoder_registry():
    model = type("LegacyModel", (), {"is_multimodal_active": True})()

    warmup_multimodal_encoders(model, device=torch.device("cpu"))


def test_warmup_propagates_encoder_failure():
    model = _WarmupModel()

    def fail(_items):
        raise RuntimeError("warmup failed")

    model.image_encoder = fail
    with pytest.raises(RuntimeError, match="warmup failed"):
        warmup_multimodal_encoders(model, device=torch.device("cpu"))


def test_model_runner_prepares_wrapper_before_encoder_warmup(monkeypatch):
    _set_graph_env(monkeypatch, graph_enabled=True)
    model = _WarmupModel()
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.server_args = type("ServerArgs", (), {"mm_attention_backend": "fa4"})()
    runner.device = "cpu"
    runner.gpu_id = 0

    runner.prepare_multimodal_runtime()

    assert runner.encoder_graph_wrappers == {
        "image_encoder": model.image_encoder,
        "video_encoder": model.video_encoder,
    }
    assert model.wrapper_calls == 1
    assert [call[0] for call in model.calls] == [Modality.IMAGE, Modality.AUDIO]


def test_install_encoder_graphs_rejects_cudnn_backend(monkeypatch):
    _set_graph_env(monkeypatch, graph_enabled=True)
    model = _WarmupModel()

    server_args = SimpleNamespace(mm_attention_backend="flashinfer_cudnn")
    assert MultimodalRuntime.install_encoder_graphs(model, server_args) == {}
    assert callable(model.image_encoder)
    assert model.wrapper_builds == 0


def _assert_warmup_item(item, modality, shape, dtype):
    assert item.modality == modality
    assert item.feature.shape == shape
    assert item.feature.dtype == dtype


def test_moonvit_encoder_warmup_item_matches_native_contract():
    vision_path = SimpleNamespace(
        vision_spec=SimpleNamespace(
            init_pos_emb_height=4,
            init_pos_emb_width=6,
        ),
        vision_tower=SimpleNamespace(
            merge_kernel_size=(2, 2),
            patch_embed=SimpleNamespace(
                patch_size=(14, 16),
                proj=SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16)),
            ),
        ),
    )

    item = MoonViTVisionPath.make_image_warmup_items(vision_path)[0]

    _assert_warmup_item(item, Modality.IMAGE, (24, 3, 14, 16), torch.bfloat16)
    assert item.grid_thws.tolist() == [[1, 4, 6]]


def test_qwen_encoder_warmup_items_cover_image_and_video_contracts():
    visual = SimpleNamespace(
        num_grid_per_side=4,
        num_position_embeddings=16,
        spatial_merge_size=2,
        patch_embed=SimpleNamespace(
            in_channels=3,
            temporal_patch_size=2,
            patch_size=16,
        ),
        dtype=torch.bfloat16,
    )
    visual._make_warmup_items = Qwen3VLMoeVisionModel._make_warmup_items.__get__(visual)

    image = Qwen3VLMoeVisionModel.make_image_warmup_items(visual)[0]
    video = Qwen3VLMoeVisionModel.make_video_warmup_items(visual)[0]

    _assert_warmup_item(image, Modality.IMAGE, (16, 1536), torch.bfloat16)
    assert image.image_grid_thw.tolist() == [[1, 4, 4]]
    _assert_warmup_item(video, Modality.VIDEO, (32, 1536), torch.bfloat16)
    assert video.video_grid_thw.tolist() == [[2, 4, 4]]


def test_moonvit_warmup_rejects_misaligned_native_grid():
    model = SimpleNamespace(
        vision_spec=SimpleNamespace(
            init_pos_emb_height=3,
            init_pos_emb_width=4,
        ),
        vision_tower=SimpleNamespace(merge_kernel_size=(2, 2)),
    )

    with pytest.raises(ValueError, match="divisible"):
        MoonViTVisionPath.make_image_warmup_items(model)


def test_qwen_warmup_rejects_non_square_native_position_grid():
    model = SimpleNamespace(
        num_grid_per_side=3,
        num_position_embeddings=10,
    )
    model._make_warmup_items = Qwen3VLMoeVisionModel._make_warmup_items.__get__(model)

    with pytest.raises(ValueError, match="square"):
        Qwen3VLMoeVisionModel.make_image_warmup_items(model)
