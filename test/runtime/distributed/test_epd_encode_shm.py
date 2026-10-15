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

"""CPU-only regression for a fast encoder rank unlinking a slow peer's pixels."""

import time
from datetime import timedelta
from multiprocessing import shared_memory

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.epd.encode_loop import _submit_encode_requests
from tokenspeed.runtime.epd.encode_worker import EncodeRequest
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle


class _Consumer:
    def __init__(self, discard: bool):
        self.discard = discard

    def submit(self, request):
        for item in request.items:
            handle = item.feature_shm
            handle.attach()
            if self.discard:
                handle.release()
            else:
                result = torch.empty(handle.shape, dtype=handle.dtype)
                handle.copy_into(result)
                torch.testing.assert_close(result, torch.arange(12).reshape(3, 4))


def _consume_on_rank(rank, rendezvous, metadata, discard):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=index,
            offsets=[(0, 0)],
            feature_shm=ShmTensorHandle(name, shape, dtype),
        )
        for index, (name, shape, dtype) in enumerate(metadata)
    ]
    try:
        # Without the pre-submit rendezvous, rank 0 unlinks both segments
        # before rank 1 has a chance to open them.
        if rank == 1:
            time.sleep(0.5)
        request = EncodeRequest(
            request_id="shm-race",
            bootstrap_host="localhost",
            bootstrap_port=1,
            bootstrap_room=1,
            items=items,
        )
        _submit_encode_requests(
            _Consumer(discard),
            [request],
            tp_cpu_group=dist.group.WORLD,
        )
    finally:
        for item in items:
            item.feature_shm.release()
        dist.destroy_process_group()


@pytest.mark.parametrize("discard", [False, True], ids=["cache-miss", "cache-hit"])
def test_encoder_tp_opens_all_shm_before_consuming(tmp_path, discard):
    handles = [
        ShmTensorHandle.publish(torch.arange(12).reshape(3, 4)) for _ in range(2)
    ]
    metadata = [(handle.shm_name, handle.shape, handle.dtype) for handle in handles]
    try:
        mp.spawn(
            _consume_on_rank,
            args=(f"file://{tmp_path / 'rendezvous'}", metadata, discard),
            nprocs=2,
        )
        for handle in handles:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=handle.shm_name)
    finally:
        for handle in handles:
            handle.release()
