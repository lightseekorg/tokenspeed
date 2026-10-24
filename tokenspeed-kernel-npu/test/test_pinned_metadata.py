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

"""NPU staging preserves packed metadata and overlapping upload ownership."""

from types import SimpleNamespace

import numpy as np
import torch
import torch_npu  # noqa: F401

from tokenspeed.runtime.engine.scheduler_utils import (
    packed_block_tables_from_forward_op,
)
from tokenspeed.runtime.utils.tensor import upload_packed


def test_packed_tables_use_pinned_host_storage():
    torch.npu.set_device(0)
    tables = {
        "swa": np.array([[0, 1, -1], [2, 3, 0]], dtype=np.int32),
        "index": np.arange(14, dtype=np.int32).reshape(2, 7),
    }
    for value in tables.values():
        value.flags.writeable = False
    result = packed_block_tables_from_forward_op(
        SimpleNamespace(block_tables_arrays=lambda: tables), "npu", num_reqs=2
    )
    storage = result.tables["swa"].untyped_storage().data_ptr()
    host_storage = result.tables_cpu["swa"].untyped_storage().data_ptr()
    for name, expected in tables.items():
        assert result.tables_cpu[name].is_pinned()
        assert result.tables[name].untyped_storage().data_ptr() == storage
        assert result.tables_cpu[name].untyped_storage().data_ptr() == host_storage
        np.testing.assert_array_equal(result.tables[name].cpu().numpy(), expected)
        np.testing.assert_array_equal(result.tables_cpu[name].numpy(), expected)


def test_mixed_upload_is_pinned_and_exact(monkeypatch):
    torch.npu.set_device(0)
    parts = (
        torch.arange(12).reshape(3, 4).T,
        torch.tensor([True, False]),
        torch.tensor([1.25, -2.5]),
        torch.tensor(19),
        torch.empty(0, dtype=torch.int32),
    )
    original_to = torch.Tensor.to
    uploads = []

    def record(tensor, *args, **kwargs):
        uploads.append((tensor.is_pinned(), kwargs.get("non_blocking")))
        return original_to(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", record)
        output = upload_packed(parts, "npu")
    assert uploads == [(True, True)]
    for actual, expected in zip(output, parts):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_uploads_survive_overlap_and_allocator_reuse():
    torch.npu.set_device(0)
    stream = torch.npu.Stream()
    a = torch.randn((256, 256), device="npu", dtype=torch.bfloat16)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(64):
            torch.mm(a, a)
    expected = []
    outputs = []
    with torch.npu.stream(stream):
        graph.replay()
        for i in range(32):
            host = torch.full((1024,), i, dtype=torch.int64)
            expected.append(host.clone())
            outputs.append(upload_packed((host,), "npu")[0])
            host.fill_(-1)
        # Fresh allocations may recycle completed stages, but must never
        # overwrite a pinned buffer whose transfer is still queued.
        pressure = [
            torch.full((8192,), 77, dtype=torch.uint8, pin_memory=True)
            for _ in range(64)
        ]
    stream.synchronize()
    assert pressure
    for actual, wanted in zip(outputs, expected):
        torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
