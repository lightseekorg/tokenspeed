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
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
from tokenspeed.runtime.models.inkling import (
    InklingAudioTower,
    InklingForConditionalGeneration,
    InklingVisionTower,
)
from tokenspeed.runtime.models.minimax_m3 import (
    MiniMaxM3SparseForConditionalGeneration,
)
from tokenspeed.runtime.models.moonvit import MoonViTVisionPath
from tokenspeed.runtime.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from tokenspeed.runtime.models.qwen3_audio import Qwen3AudioEncoder
from tokenspeed.runtime.models.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
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


def test_minimax_m3_warmup_items_cover_image_and_video_contracts():
    image_encoder = Mock()
    video_encoder = Mock()
    model = SimpleNamespace(
        vl_config=SimpleNamespace(
            image_seq_length=4,
            vision_config=SimpleNamespace(
                num_channels=3,
                temporal_patch_size=3,
                patch_size=2,
                spatial_merge_size=2,
            ),
        ),
        vision_tower=SimpleNamespace(dtype=torch.bfloat16),
        image_encoder=image_encoder,
        video_encoder=video_encoder,
    )
    model._make_vision_warmup_items = (
        MiniMaxM3SparseForConditionalGeneration._make_vision_warmup_items.__get__(model)
    )
    model.make_image_warmup_items = (
        MiniMaxM3SparseForConditionalGeneration.make_image_warmup_items.__get__(model)
    )
    model.make_video_warmup_items = (
        MiniMaxM3SparseForConditionalGeneration.make_video_warmup_items.__get__(model)
    )

    image = MiniMaxM3SparseForConditionalGeneration.make_image_warmup_items(model)[0]
    video = MiniMaxM3SparseForConditionalGeneration.make_video_warmup_items(model)[0]
    specs = MiniMaxM3SparseForConditionalGeneration.get_multimodal_encoder_specs(model)

    _assert_warmup_item(image, Modality.IMAGE, (16, 36), torch.bfloat16)
    assert image.image_grid_thw.tolist() == [[1, 4, 4]]
    _assert_warmup_item(video, Modality.VIDEO, (32, 36), torch.bfloat16)
    assert video.video_grid_thw.tolist() == [[2, 4, 4]]
    assert specs[Modality.IMAGE].fn is image_encoder
    assert specs[Modality.IMAGE].make_warmup_items is model.make_image_warmup_items
    assert specs[Modality.VIDEO].fn is video_encoder
    assert specs[Modality.VIDEO].make_warmup_items is model.make_video_warmup_items


def test_minimax_m3_warmup_rejects_nonpositive_token_count():
    model = SimpleNamespace(
        vl_config=SimpleNamespace(
            image_seq_length=0,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
    )

    with pytest.raises(ValueError, match="positive"):
        MiniMaxM3SparseForConditionalGeneration._make_vision_warmup_items(
            model,
            Modality.IMAGE,
            temporal_patches=1,
            grid_key="image_grid_thw",
        )


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


def test_qwen_audio_warmup_uses_configured_inference_window():
    tower = SimpleNamespace(
        n_window=4,
        n_window_infer=16,
        num_mel_bins=8,
        dtype=torch.bfloat16,
    )

    item = Qwen3AudioEncoder.make_warmup_items(tower)[0]

    _assert_warmup_item(item, Modality.AUDIO, (8, 16), torch.bfloat16)


class _InklingPatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_patch_size = 2
        self.patch_size = 4
        self.n_channels = 3
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))


def test_inkling_warmup_items_use_one_independent_native_unit():
    vision = SimpleNamespace(vision_encoder=_InklingPatchEncoder())
    image = InklingVisionTower.make_warmup_items(vision)[0]
    _assert_warmup_item(image, Modality.IMAGE, (1, 2, 4, 4, 3), torch.bfloat16)

    audio = SimpleNamespace(n_mel_bins=80)
    audio_item = InklingAudioTower.make_warmup_items(audio)[0]
    _assert_warmup_item(audio_item, Modality.AUDIO, (1, 80), torch.long)


def test_audio_and_mixed_models_register_family_warmup_factories():
    audio_factory = Mock()
    audio_encoder = Mock()
    image_factory = Mock()
    video_factory = Mock()
    image_encoder = Mock()
    video_encoder = Mock()

    audio_tower = SimpleNamespace(make_warmup_items=audio_factory)
    asr = SimpleNamespace(audio_tower=audio_tower, audio_encoder=audio_encoder)
    asr_specs = Qwen3ASRForConditionalGeneration.get_multimodal_encoder_specs(asr)
    assert asr_specs[Modality.AUDIO].fn is audio_encoder
    assert asr_specs[Modality.AUDIO].make_warmup_items is audio_factory

    visual = SimpleNamespace(
        make_image_warmup_items=image_factory,
        make_video_warmup_items=video_factory,
    )
    omni = SimpleNamespace(
        visual=visual,
        audio_tower=audio_tower,
        image_encoder=image_encoder,
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
    )
    omni_specs = Qwen3OmniMoeForConditionalGeneration.get_multimodal_encoder_specs(omni)
    assert set(omni_specs) == {Modality.IMAGE, Modality.VIDEO, Modality.AUDIO}
    assert omni_specs[Modality.IMAGE].deepstack
    assert omni_specs[Modality.VIDEO].deepstack
    assert omni_specs[Modality.AUDIO].make_warmup_items is audio_factory
    omni.audio_tower = None
    assert set(
        Qwen3OmniMoeForConditionalGeneration.get_multimodal_encoder_specs(omni)
    ) == {Modality.IMAGE, Modality.VIDEO}

    inkling = SimpleNamespace(
        visual=SimpleNamespace(make_warmup_items=image_factory),
        audio=SimpleNamespace(make_warmup_items=audio_factory),
        image_encoder=image_encoder,
        audio_encoder=audio_encoder,
    )
    inkling_specs = InklingForConditionalGeneration.get_multimodal_encoder_specs(
        inkling
    )
    assert set(inkling_specs) == {Modality.IMAGE, Modality.AUDIO}
    assert inkling_specs[Modality.IMAGE].make_warmup_items is image_factory
    assert inkling_specs[Modality.AUDIO].make_warmup_items is audio_factory
