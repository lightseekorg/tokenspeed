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

import pytest
import torch

from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.multimodal.warmup import (
    install_encoder_cudagraph_wrappers,
    prewarm_multimodal_encoders,
)
from tokenspeed.runtime.utils.env import envs


class _WarmupModel:
    is_multimodal_active = True

    def __init__(self):
        self.mapping = object()
        self.calls = []
        self.image_encoder = self._image_encoder
        self.video_encoder = self._video_encoder

    def make_encoder_warmup_items(self, patches_per_side):
        assert patches_per_side == 4
        return {
            Modality.IMAGE: [MultimodalDataItem(Modality.IMAGE)],
            Modality.VIDEO: [MultimodalDataItem(Modality.VIDEO)],
        }

    def _image_encoder(self, items):
        self.calls.append((Modality.IMAGE, items, torch.is_inference_mode_enabled()))
        return torch.zeros(1)

    def _video_encoder(self, items):
        self.calls.append((Modality.VIDEO, items, torch.is_inference_mode_enabled()))
        return torch.zeros(1)

    def make_encoder_cudagraph_wrappers(self, mapping):
        assert mapping is self.mapping
        wrapper = object()
        return {"image_encoder": wrapper, "video_encoder": wrapper}


def _set_warmup_env(monkeypatch, *, enabled=True, patches_per_side=4):
    monkeypatch.setattr(
        envs.TOKENSPEED_MM_ENABLE_VISION_PREWARM, "get", lambda: enabled
    )
    monkeypatch.setattr(
        envs.TOKENSPEED_MM_VISION_PREWARM_PATCHES_PER_SIDE,
        "get",
        lambda: patches_per_side,
    )


def test_prewarm_dispatches_image_and_video_in_inference_mode(monkeypatch):
    _set_warmup_env(monkeypatch)
    model = _WarmupModel()

    prewarm_multimodal_encoders(
        model, skip_server_warmup=False, device=torch.device("cpu")
    )

    assert [call[0] for call in model.calls] == [Modality.IMAGE, Modality.VIDEO]
    assert all(call[2] for call in model.calls)


@pytest.mark.parametrize(
    ("enabled", "skip_server_warmup", "patches_per_side"),
    [(False, False, 4), (True, True, 4), (True, False, 0)],
)
def test_prewarm_respects_startup_gates(
    monkeypatch, enabled, skip_server_warmup, patches_per_side
):
    _set_warmup_env(monkeypatch, enabled=enabled, patches_per_side=patches_per_side)
    model = _WarmupModel()

    prewarm_multimodal_encoders(
        model,
        skip_server_warmup=skip_server_warmup,
        device=torch.device("cpu"),
    )

    assert model.calls == []


def test_prewarm_propagates_encoder_failure(monkeypatch):
    _set_warmup_env(monkeypatch)
    model = _WarmupModel()

    def fail(_items):
        raise RuntimeError("warmup failed")

    model.image_encoder = fail
    with pytest.raises(RuntimeError, match="warmup failed"):
        prewarm_multimodal_encoders(
            model, skip_server_warmup=False, device=torch.device("cpu")
        )


def test_install_encoder_cudagraph_wrappers_installs_every_modality(monkeypatch):
    monkeypatch.setattr(
        envs.TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH, "get", lambda: True
    )
    model = _WarmupModel()

    wrappers = install_encoder_cudagraph_wrappers(model, "fa4")

    assert wrappers == {
        "image_encoder": model.image_encoder,
        "video_encoder": model.video_encoder,
    }
    assert model.image_encoder is model.video_encoder


def test_install_encoder_cudagraph_wrappers_rejects_cudnn_backend(monkeypatch):
    monkeypatch.setattr(
        envs.TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH, "get", lambda: True
    )
    model = _WarmupModel()

    assert install_encoder_cudagraph_wrappers(model, "flashinfer_cudnn") == {}
    assert callable(model.image_encoder)
