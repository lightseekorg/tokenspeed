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
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.layers.dense import Fp8LinearMethod
from tokenspeed.runtime.layers.linear import RowParallelLinear


def _block_fp8_method() -> Fp8LinearMethod:
    method = object.__new__(Fp8LinearMethod)
    method.block_quant = True
    method.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    return method


def test_block_fp8_linear_forwards_output_buffer() -> None:
    method = _block_fp8_method()
    layer = SimpleNamespace(
        weight=torch.empty(256, 128, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.ones(2, 1),
    )
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    target = torch.empty(2, 256, dtype=torch.bfloat16)

    with mock.patch(
        "tokenspeed.runtime.layers.dense.fp8.tokenspeed_kernel.mm",
        side_effect=lambda *args, out=None, **kwargs: out.fill_(1.0),
    ) as mm:
        actual = method.apply(layer, x, output=target)

    assert actual.data_ptr() == target.data_ptr()
    assert mm.call_args.kwargs["out"].data_ptr() == target.data_ptr()


def test_block_fp8_linear_rejects_output_shape() -> None:
    method = _block_fp8_method()
    layer = SimpleNamespace(
        weight=torch.empty(256, 128, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.ones(2, 1),
    )
    x = torch.randn(2, 128, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="output shape"):
        method.apply(layer, x, output=torch.empty(2, 128, dtype=torch.bfloat16))


def test_row_parallel_linear_forwards_fp8_output() -> None:
    method = _block_fp8_method()
    linear = SimpleNamespace(
        input_is_parallel=True,
        quant_method=method,
        tp_rank=0,
        tp_size=8,
        tp_group=tuple(range(8)),
        skip_bias_add=False,
        bias=None,
        reduce_results=False,
    )
    x = torch.randn(1, 128, dtype=torch.bfloat16)
    target = torch.empty(1, 256, dtype=torch.bfloat16)

    with mock.patch.object(Fp8LinearMethod, "apply", return_value=target) as apply:
        actual, bias = RowParallelLinear.forward(linear, x, output=target)

    assert actual is target
    assert bias is None
    assert apply.call_args.kwargs["output"] is target


def test_row_parallel_linear_rejects_unsupported_output() -> None:
    linear = SimpleNamespace(
        input_is_parallel=True,
        quant_method=object(),
        tp_rank=0,
        tp_size=1,
        skip_bias_add=False,
        bias=None,
        reduce_results=False,
    )

    with pytest.raises(ValueError, match="only for FP8"):
        RowParallelLinear.forward(
            linear,
            torch.randn(1, 128),
            output=torch.empty(1, 256),
        )
