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

"""Media encodings retain the device lifetime of their last prefill scatter."""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.multimodal.embedder import (  # noqa: E402
    EncoderSpec,
    MultimodalEmbedder,
)
from tokenspeed.runtime.multimodal.inputs import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
    MultimodalInputs,
)


class TinyMediaModel(nn.Module):
    def __init__(self, deepstack):
        super().__init__()
        self.deepstack = deepstack
        self.deepstack_visual_indexes = [0]
        self.projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))

    def encode(self, items):
        encoded = torch.cat([self.projection(item.feature) for item in items])
        return torch.cat((encoded, -encoded), dim=-1) if self.deepstack else encoded

    def separate_deepstack_embeds(self, encoded):
        return encoded[:, :4], encoded[:, 4:]


@pytest.fixture(params=["cpu", "cuda", "npu"])
def device(request):
    name = request.param
    if name == "cpu":
        return torch.device("cpu")
    try:
        module = torch.get_device_module(name)
    except RuntimeError:
        pytest.skip(f"{name} backend is not installed")
    if not module.is_available():
        pytest.skip(f"{name} device is not available")
    return torch.device(f"{name}:0")


@pytest.mark.parametrize("deepstack", [False, True])
def test_last_prefill_chunk_drops_main_and_deepstack_encodings(device, deepstack):
    model = TinyMediaModel(deepstack).to(device)
    text_embedding = nn.Embedding(8, 4).to(device)
    feature = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=19,
        offsets=[(1, 3)],
        feature=feature.clone(),
    )
    inputs = MultimodalInputs(mm_items=[item], im_token_id=7)
    embedder = MultimodalEmbedder()
    encoders = {Modality.IMAGE: EncoderSpec(model.encode, deepstack=deepstack)}

    def apply(prefix, length, ids):
        context = MultimodalForwardContext(
            mm_inputs=[inputs],
            extend_prefix_lens=[prefix],
            extend_seq_lens=[length],
            max_encoder_tokens=8192,
        )
        return embedder.apply(
            torch.tensor(ids, device=device), text_embedding, context, encoders, model
        )

    first, _ = apply(0, 2, [1, 7])
    torch.testing.assert_close(first[1:].cpu(), feature[:1])
    assert item.encoded.device == device
    assert (item.encoded_deepstack is not None) is deepstack

    last, _ = apply(2, 2, [7, 7])
    torch.testing.assert_close(last.cpu(), feature[1:])
    assert item.encoded is None
    assert item.encoded_deepstack is None


@torch.inference_mode()
def test_encoding_from_another_stream_is_consumed_and_released(device):
    if device.type == "cpu":
        pytest.skip("CPU tensors do not use device streams")
    module = torch.get_device_module(device)
    producer = module.Stream(device=device)
    consumer = module.Stream(device=device)
    model = TinyMediaModel(False).to(device)
    text_embedding = nn.Embedding(8, 4).to(device)
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    item = MultimodalDataItem(modality=Modality.IMAGE, hash=23, offsets=[(1, 3)])

    creation_stream = module.current_stream(device)
    producer.wait_stream(creation_stream)
    with module.stream(producer):
        item.feature = expected.to(device)
        item.encoded = model.encode([item])
        item.feature = None

    consumer.wait_stream(creation_stream)
    consumer.wait_stream(producer)
    context = MultimodalForwardContext(
        mm_inputs=[MultimodalInputs(mm_items=[item], im_token_id=7)],
        extend_prefix_lens=[0],
        extend_seq_lens=[4],
        max_encoder_tokens=8192,
    )
    with module.stream(consumer):
        actual, _ = MultimodalEmbedder().apply(
            torch.tensor([1, 7, 7, 7], device=device),
            text_embedding,
            context,
            {Modality.IMAGE: EncoderSpec(model.encode)},
            model,
        )

    consumer.synchronize()
    torch.testing.assert_close(actual[1:].cpu(), expected)
    assert item.encoded is None
    assert item.encoded_deepstack is None
