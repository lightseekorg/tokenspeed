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

from multiprocessing import shared_memory

import pytest
import torch

from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle
from tokenspeed.runtime.utils import common


class _SequentialBroadcast:
    """Capture source broadcasts, then replay them into one receiving rank."""

    def __init__(self) -> None:
        self.values: list[torch.Tensor] = []
        self.read_index = 0

    def source(self, tensor, *, src, group=None, async_op=False):
        assert src == 0
        assert group is None
        assert async_op is False
        self.values.append(tensor.detach().cpu().clone())

    def receiver(self, tensor, *, src, group=None, async_op=False):
        assert src == 0
        assert group is None
        assert async_op is False
        tensor.copy_(self.values[self.read_index].to(tensor.device))
        self.read_index += 1


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_unsigned_request_tensor_round_trip_preserves_dtype_shape_and_values(dtype):
    source = torch.arange(12, dtype=torch.int64).to(dtype).reshape(3, 4)

    restored = common.pickle.loads(common._serialize_request_pyobj([source]))[0]

    assert restored.dtype == dtype
    assert restored.shape == source.shape
    assert torch.equal(restored, source)


def test_broadcast_unsigned_model_metadata_preserves_rank_lifecycles_and_shm(
    monkeypatch,
):
    feature = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    handle = ShmTensorHandle.publish(feature)
    inputs = MultimodalInputs(
        mm_items=[
            MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=17,
                offsets=[(3, 6)],
                feature_shm=handle,
                model_specific_data={
                    "n_llm_h": torch.tensor([10], dtype=torch.uint32),
                    "n_llm_w": torch.tensor([[7, 9]], dtype=torch.uint32),
                },
            )
        ]
    )
    ordinary = torch.tensor([5, -2], dtype=torch.int64)
    source_data = [inputs, {"ordinary": ordinary, "label": "request"}]
    transport = _SequentialBroadcast()

    try:
        monkeypatch.setattr(common.dist, "broadcast", transport.source)
        returned = common.broadcast_pyobj(source_data, rank=0, src=0)
        assert returned is source_data
        assert returned[0].mm_items[0].feature_shm is handle

        monkeypatch.setattr(common.dist, "broadcast", transport.receiver)
        received = common.broadcast_pyobj([], rank=1, src=0)
        assert transport.read_index == 2
        received_item = received[0].mm_items[0]
        assert received_item.model_specific_data["n_llm_h"].dtype == torch.uint32
        assert received_item.model_specific_data["n_llm_h"].shape == (1,)
        assert received_item.model_specific_data["n_llm_h"].item() == 10
        assert received_item.model_specific_data["n_llm_w"].dtype == torch.uint32
        assert received_item.model_specific_data["n_llm_w"].shape == (1, 2)
        assert received_item.model_specific_data["n_llm_w"].tolist() == [[7, 9]]
        assert received[1]["label"] == "request"
        assert torch.equal(received[1]["ordinary"], ordinary)

        received_item.feature_shm.attach()
        copied = received_item.feature_shm.consume()
        assert torch.equal(copied, feature)
        received_item.feature_shm = None
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.shm_name)
    finally:
        handle.release()
