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

"""Packed metadata uploads preserve values, alignment and forward ownership."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

# Load this torch-only utility independently of the GPU runtime package, as in
# test_page_table_conversion.py, so the CPU contract is testable without CUDA.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime/utils/tensor.py"
)
_SPEC = importlib.util.spec_from_file_location("packed_upload_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
upload_packed = _MODULE.upload_packed


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for asynchronous upload coverage")
    return torch.device(request.param)


def test_mixed_shapes_and_dtypes_share_one_upload(device, monkeypatch):
    parts = (
        torch.tensor([7], dtype=torch.uint8),
        torch.tensor([[11, 12], [13, 14]], dtype=torch.int64),
        torch.tensor([1.5], dtype=torch.float32),
        torch.tensor(19, dtype=torch.int64),
        torch.arange(12, dtype=torch.int32).reshape(3, 4).T,
        torch.empty((2, 0, 3), dtype=torch.float64),
        torch.tensor([True, False]),
        torch.tensor([2 + 3j], dtype=torch.complex128),
    )
    transfers = []
    original_to = torch.Tensor.to

    def record_upload(tensor, *args, **kwargs):
        transfers.append((tensor.device.type, tensor.dtype, tensor.is_pinned(), kwargs))
        return original_to(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", record_upload)
        result = upload_packed(parts, device)
    assert len(transfers) == 1
    assert transfers[0] == (
        "cpu",
        torch.uint8,
        device.type == "cuda",
        {"device": device, "non_blocking": True},
    )
    assert len(result) == len(parts)
    storage = result[0].untyped_storage().data_ptr()
    for actual, expected in zip(result, parts):
        assert actual.device.type == device.type
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.is_contiguous()
        assert actual.untyped_storage().data_ptr() == storage
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_uploads_own_independent_storage(device):
    part = torch.tensor([1, 2, 3], dtype=torch.int64)
    (first,) = upload_packed((part,), device)
    part.add_(10)
    (second,) = upload_packed((part,), device)
    assert first.data_ptr() != second.data_ptr()
    assert first.cpu().tolist() == [1, 2, 3]
    assert second.cpu().tolist() == [11, 12, 13]


def test_empty_inputs(device):
    assert upload_packed((), device) == ()
    parts = (torch.empty((0, 2), dtype=torch.int32), torch.empty(0, dtype=torch.int64))
    result = upload_packed(parts, device)
    assert len(result) == 2
    for actual, expected in zip(result, parts):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.device.type == device.type


def test_non_cpu_inputs_are_rejected():
    with pytest.raises(ValueError, match="requires CPU tensors"):
        upload_packed((torch.empty(2, device="meta"),), "cpu")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
