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

from types import SimpleNamespace
from unittest.mock import MagicMock

import tokenspeed_kernel.ops.ple as ple
import torch


def test_ple_host_gather_uses_device_visible_table_pointer(monkeypatch) -> None:
    table = torch.empty((4, 8), dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2]], dtype=torch.int64)
    out = torch.empty((2, 8), dtype=torch.bfloat16)
    pointer = MagicMock(return_value=12345)
    kernel = MagicMock()
    monkeypatch.setattr(
        ple,
        "current_platform",
        lambda: SimpleNamespace(device_visible_data_ptr=pointer),
    )
    monkeypatch.setattr(ple, "_ple_host_gather_kernel", kernel)

    assert ple.ple_host_gather(table, ids, out, 0, 4, None, None) is out
    pointer.assert_called_once_with(table)
    args, kwargs = kernel.__getitem__.return_value.call_args
    assert args[0] == 12345
    assert args[1].shape == (2,)
    assert args[4].shape == (2, 8)
    assert kwargs["IS_FP8"] is False


def test_ple_host_gather_empty_ids_skip_pointer_mapping(monkeypatch) -> None:
    table = torch.empty((4, 8), dtype=torch.bfloat16)
    ids = torch.empty(0, dtype=torch.int64)
    out = torch.empty((0, 8), dtype=torch.bfloat16)
    pointer = MagicMock()
    kernel = MagicMock()
    monkeypatch.setattr(
        ple,
        "current_platform",
        lambda: SimpleNamespace(device_visible_data_ptr=pointer),
    )
    monkeypatch.setattr(ple, "_ple_host_gather_kernel", kernel)

    assert ple.ple_host_gather(table, ids, out, 0, 4, None, None) is out
    pointer.assert_not_called()
    kernel.__getitem__.assert_not_called()
