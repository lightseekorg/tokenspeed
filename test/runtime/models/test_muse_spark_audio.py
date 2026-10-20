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

"""The Muse Spark waveform tower: naming, token geometry, causality, wiring."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models.muse_spark import MuseSparkForConditionalGeneration
from tokenspeed.runtime.models.muse_spark_audio import (
    AUDIO_SAMPLES_PER_TOKEN,
    MuseSparkAudioTower,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

HIDDEN = 24


AUDIO_PREFIX = "model.audio_tower."


def _tiny_config(out_hidden_size: int) -> dict:
    return {
        "conv_kernel_size": 7,
        "depths": [2, 2, 2, 2],
        "hidden_act": "gelu",
        "hidden_sizes": [4, 8, 8, 16],
        "mlp_ratio": 4.0,
        "num_hidden_layers": 4,
        "out_hidden_size": out_hidden_size,
        "patch_sizes": [16, 6, 5, 4],
        "rms_norm_eps": 1e-6,
    }


def _tiny_tower() -> MuseSparkAudioTower:
    return MuseSparkAudioTower.from_config(_tiny_config(HIDDEN), HIDDEN)


def _stub_model(audio_tower: MuseSparkAudioTower | None):
    model = MuseSparkForConditionalGeneration.__new__(MuseSparkForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.vision_tower = None
    model.audio_tower = audio_tower
    model._loaded_audio = set()
    model.model = torch.nn.Module()
    model.model.layers = []
    model.is_multimodal_active = audio_tower is not None
    return model


def test_parameter_names_follow_the_checkpoint_layout() -> None:
    names = set(dict(_tiny_tower().named_parameters()))
    block = ("conv", "mlp.fc1", "mlp.fc2")
    expected = {"norm.weight", "proj.weight", "proj.bias", "post_proj_norm.weight"}
    for layer in range(4):
        expected |= {
            f"layers.{layer}.patch_embedder.{n}"
            for n in ("proj.weight", "proj.bias", "norm.weight")
        }
        for b in (0, 1):
            expected |= {f"layers.{layer}.conv_blocks.{b}.norm.weight"}
            expected |= {
                f"layers.{layer}.conv_blocks.{b}.{m}.{p}"
                for m in block
                for p in ("weight", "bias")
            }
    assert names == expected and len(names) == 72


def test_one_token_per_1920_samples_and_items_concatenate() -> None:
    torch.manual_seed(0)
    encoder = _tiny_tower()
    first = torch.randn(3 * AUDIO_SAMPLES_PER_TOKEN + 5)
    second = torch.randn(AUDIO_SAMPLES_PER_TOKEN)
    output = encoder([first, second])
    assert math.ceil(first.numel() / AUDIO_SAMPLES_PER_TOKEN) == 4
    assert output.shape == (5, HIDDEN)
    assert torch.equal(output[:4], encoder([first]))
    assert torch.equal(output[4:], encoder([second]))
    # The partial tail token is encoded as if explicitly zero padded.
    padded = torch.cat([first, torch.zeros(AUDIO_SAMPLES_PER_TOKEN - 5)])
    assert torch.equal(output[:4], encoder([padded]))


def test_encoding_is_causal_per_token() -> None:
    torch.manual_seed(1)
    encoder = _tiny_tower()
    waveform = torch.randn(4 * AUDIO_SAMPLES_PER_TOKEN)
    full = encoder([waveform])
    short = encoder([waveform[: 2 * AUDIO_SAMPLES_PER_TOKEN]])
    torch.testing.assert_close(full[:2], short, rtol=1e-5, atol=1e-5)
    changed = waveform.clone()
    changed[3 * AUDIO_SAMPLES_PER_TOKEN :] += 1.0
    torch.testing.assert_close(encoder([changed])[:3], full[:3], rtol=1e-5, atol=1e-5)


def test_from_config_checks_the_geometry_it_is_handed() -> None:
    tower = MuseSparkAudioTower.from_config(_tiny_config(32), 32)
    assert tower.proj.weight.shape == (32, 16)
    for flaw, message in (
        ({"hidden_act": "silu"}, "GELU"),
        ({"out_hidden_size": 64}, "decoder-width"),
        ({"depths": [2, 2, 2]}, "every layer"),
        ({"patch_sizes": [16, 6, 5, 2]}, "samples per token"),
    ):
        with pytest.raises(ValueError, match=message):
            MuseSparkAudioTower.from_config({**_tiny_config(32), **flaw}, 32)


def test_the_model_registers_the_audio_tower_and_its_warmup() -> None:
    model = _stub_model(_tiny_tower())
    specs = model.get_multimodal_encoder_specs()
    assert set(specs) == {Modality.AUDIO}
    items = specs[Modality.AUDIO].make_warmup_items()
    assert [tuple(item.feature.shape) for item in items] == [(AUDIO_SAMPLES_PER_TOKEN,)]
    assert specs[Modality.AUDIO].fn(items).shape == (1, HIDDEN)
    assert _stub_model(None).get_multimodal_encoder_specs() == {}


def test_wire_shaped_bf16_samples_encode_like_the_flat_waveform() -> None:
    torch.manual_seed(3)
    model = _stub_model(_tiny_tower())
    waveform = torch.randn(2 * AUDIO_SAMPLES_PER_TOKEN + 9)
    wire = MultimodalDataItem(
        modality=Modality.AUDIO, feature=waveform.reshape(-1, 1).to(torch.bfloat16)
    )
    output = model.get_audio_feature([wire])
    assert output.shape == (3, HIDDEN)
    assert torch.equal(output, model.audio_tower([waveform.to(torch.bfloat16).float()]))


def test_checkpoint_audio_tensors_load_into_the_tower() -> None:
    model = _stub_model(_tiny_tower())
    name = AUDIO_PREFIX + "proj.bias"
    loaded = torch.full((HIDDEN,), 0.5)
    # Weights arrive in any number of calls; completeness is judged afterwards.
    model.load_weights([(name, loaded)])
    assert torch.equal(model.audio_tower.proj.bias, loaded)
    with pytest.raises(ValueError, match="missing audio tensors"):
        model.post_quant_warmup()
    with pytest.raises(ValueError, match="no parameter"):
        model.load_weights([(AUDIO_PREFIX + "patch_encoder.weight", loaded)])
    complete = [
        (AUDIO_PREFIX + key, torch.zeros_like(value))
        for key, value in model.audio_tower.state_dict().items()
    ]
    model.load_weights(complete)
    model.post_quant_warmup()
    assert not model.audio_tower.proj.bias.any()
    # A checkpoint that ships no audio tensors at all is refused the same way.
    with pytest.raises(ValueError, match="missing audio tensors"):
        _stub_model(_tiny_tower()).post_quant_warmup()
    # Without the tower, audio tensors (even unknown ones) are skipped, not refused.
    towerless = _stub_model(None)
    towerless.load_weights(complete + [(AUDIO_PREFIX + "patch_encoder.weight", loaded)])
    assert towerless._loaded_audio == set()
    towerless.post_quant_warmup()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
