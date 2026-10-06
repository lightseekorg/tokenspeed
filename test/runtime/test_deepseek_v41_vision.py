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

"""Packed V4.1 image boundaries, batching limits, and encoder equivalence."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.models import deepseek_v41_vision as vision
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem


def _model(backend):
    config = SimpleNamespace(
        vision_config=SimpleNamespace(
            hidden_size=128,
            intermediate_size=192,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=2,
            downsample_ratio=2,
            rope_theta=10000,
        ),
        text_config=SimpleNamespace(hidden_size=32),
    )
    torch.manual_seed(7)
    model = vision.DeepseekV41Vision(config, Mapping(rank=0, world_size=1), backend)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.fill_(1)
            else:
                parameter.normal_(std=0.05)
    return model.eval()


def _item(h, w):
    lh, lw = (h + 1) // 2, (w + 1) // 2
    types = torch.tensor([0] + ([1] * lw + [2]) * lh + [3])
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, len(types) - 1)],
        feature=torch.randn(h * w, 3, 2, 2),
        model_specific_data={
            "vit_grid": torch.tensor([h, w]),
            "llm_grid": torch.tensor([lh, lw]),
            "types": types,
        },
    )


@pytest.mark.parametrize(
    "grids,expected",
    [
        ([(2, 2)] * 9, [9]),
        ([(64, 128), (64, 128), (2, 2)], [2, 1]),
        ([(129, 128), (2, 2)], [1, 1]),
    ],
)
def test_batch_limits_and_order(grids, expected):
    model = _model("triton_attn")
    items = [_item(h, w) for h, w in grids]
    seen = []

    def encode(batch, batch_grids):
        seen.append(len(batch))
        return [
            torch.tensor(
                [[next(i for i, original in enumerate(items) if original is item)]]
            )
            for item in batch
        ]

    with patch.object(model, "_embed_batch", side_effect=encode):
        output = model.embed_media(items)
    assert seen == expected
    assert output.flatten().tolist() == list(range(len(items)))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("vit_grid", [0, 2], "positive"),
        ("vit_grid", [2, 3], "disagree"),
        ("llm_grid", [2, 1], "disagree"),
        ("types", [0, 1, 3, 2], "types"),
    ],
)
def test_invalid_metadata_rejected_before_encoding(field, value, message):
    model = _model("triton_attn")
    item = _item(2, 2)
    item.model_specific_data[field] = torch.tensor(value)
    with patch.object(model, "_embed_batch") as encode:
        with pytest.raises(ValueError, match=message):
            model.embed_media([_item(2, 2), item])
        encode.assert_not_called()


def test_empty_and_invalid_patch_count():
    model = _model("triton_attn")
    with pytest.raises(ValueError, match="at least one"):
        model.embed_media([])
    item = _item(2, 2)
    item.feature = item.feature[:3]
    with pytest.raises(ValueError, match="patch count"):
        model.embed_one(item)


class _CaptureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, cos, sin, cu, lengths, maximum):
        self.calls.append((cos, sin, cu, lengths, maximum))
        return x


def test_packed_metadata_and_rope():
    model = _model("triton_attn")
    tower = model.vision
    capture = _CaptureBlock()
    tower.blocks = nn.ModuleList([capture])
    tower.forward_packed(torch.randn(10, 3, 2, 2), [(2, 2), (3, 2)])
    cos, sin, cu, lengths, maximum = capture.calls[0]
    assert maximum == 6
    assert cu.tolist() == [0, 4, 10]
    assert lengths is None
    for start, end, h, w in [(0, 4, 2, 2), (4, 10, 3, 2)]:
        expected = vision.get_vision_cos_sin(h, w, 32, 10000, torch.device("cpu"))
        torch.testing.assert_close(cos[start:end], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(sin[start:end], expected[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["fa3", "triton_attn", "flashinfer_cudnn"])
@pytest.mark.parametrize("num_images", [3, 9])
@torch.inference_mode()
def test_packed_matches_independent_images_and_isolation(backend, num_images):
    if backend == "fa3" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 requires Hopper")
    model = _model(backend).to(device="cuda", dtype=torch.bfloat16)
    items = [_item(h, w) for h, w in [(4, 4), (3, 5), (2, 6)] * (num_images // 3)]
    # Singleton calls also exercise the original public API.
    reference = torch.cat([model.embed_one(item) for item in items])
    actual = model.embed_media(items)
    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=1e-2)
    length = items[0].model_specific_data["types"].numel()
    items[1].feature.mul_(10)
    changed = model.embed_media(items)
    torch.testing.assert_close(changed[:length], actual[:length], atol=0, rtol=0)
    offset = 0
    for item in items:
        types = item.model_specific_data["types"]
        block = actual[offset : offset + len(types)]
        for tag, sentinel in [
            (0, model.image_start),
            (2, model.image_newline),
            (3, model.image_end),
        ]:
            torch.testing.assert_close(
                block[types == tag],
                sentinel.expand_as(block[types == tag]),
                atol=0,
                rtol=0,
            )
        offset += len(types)


def test_cudnn_uses_singleton_batches():
    model = _model("triton_attn")
    model.vision.mm_attention_backend = "flashinfer_cudnn"
    items = [_item(2, 2), _item(3, 2)]
    with patch.object(
        model, "_embed_batch", return_value=[torch.zeros(1, 32)]
    ) as encode:
        model.embed_media(items)
    assert [len(call.args[0]) for call in encode.call_args_list] == [1, 1]
    with pytest.raises(ValueError, match="singleton"):
        model.vision.forward_packed(torch.randn(10, 3, 2, 2), [(2, 2), (3, 2)])


@pytest.mark.parametrize("budget", [4, 8, 16])
def test_configured_token_budget(monkeypatch, budget):
    monkeypatch.setenv("TOKENSPEED_DEEPSEEK_V41_VISION_MAX_BATCH_TOKENS", str(budget))
    model = _model("triton_attn")
    items = [_item(2, 2) for _ in range(4)]
    with patch.object(
        model,
        "_embed_batch",
        side_effect=lambda batch, grids: [torch.zeros(1, 32) for _ in batch],
    ) as encode:
        model.embed_media(items)
    assert [len(call.args[0]) for call in encode.call_args_list] == [budget // 4] * (
        16 // budget
    )


def test_invalid_token_budget(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DEEPSEEK_V41_VISION_MAX_BATCH_TOKENS", "0")
    with pytest.raises(ValueError, match="positive"):
        _model("triton_attn")
