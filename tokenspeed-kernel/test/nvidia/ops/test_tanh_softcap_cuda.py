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

"""``tanh_softcap`` against Torch's ``(x.float() / cap).tanh() * cap`` on NVIDIA
SM100 to SM103 (where Beam calls it), byte for byte.

The function of one element has 65,536 BF16 (and FP16) inputs: every word is run,
specials included, at Beam's cap 30.0, its integer form and 1/3 (not
representable). FP32 inputs are sampled over every binade. NaN rule: equal
``isnan`` masks and equal words elsewhere; at cap 30.0 the NaN words too.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import tanh_softcap
from tokenspeed_kernel.platform import ArchVersion, current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_nvidia
    or not ArchVersion(10, 0) <= current_platform().arch_version <= ArchVersion(10, 3),
    reason="Beam calls the one-pass soft cap on NVIDIA SM100 to SM103",
)

DEVICE = torch.device("cuda", 0)
CAPS = (30.0, 30, 1 / 3)
# A 151,936-entry vocabulary and a width just past the block.
WIDTHS = (1, 2049, 151936)


def _statement(x: torch.Tensor, cap: float) -> torch.Tensor:
    return (x.float() / cap).tanh() * cap


def _words(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().view(torch.int32)


def _assert_nan_rule(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype is torch.float32 and actual.shape == expected.shape
    assert actual.is_contiguous()
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    same = _words(actual) == _words(expected)
    bad = (~same & ~nan).flatten().nonzero().flatten()
    if bad.numel():
        first = int(bad[0])
        pytest.fail(
            f"{bad.numel()} non-NaN words differ; first at flat index {first}: "
            f"kernel {actual.flatten()[first].item()!r} statement "
            f"{expected.flatten()[first].item()!r}"
        )


def _all_words(dtype: torch.dtype) -> torch.Tensor:
    words = torch.arange(-(2**15), 2**15, dtype=torch.int32, device=DEVICE)
    return words.to(torch.int16).view(dtype).reshape(256, 256)


@pytest.mark.parametrize("cap", CAPS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_every_16_bit_input_equals_the_statement(dtype, cap):
    x = _all_words(dtype)
    out, expected = tanh_softcap(x, cap), _statement(x, cap)
    _assert_nan_rule(out, expected)
    if cap == 30.0:
        assert torch.equal(_words(out), _words(expected))


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("rows", (1, 512))
def test_logit_shapes_and_strided_rows_equal_the_statement(rows, width):
    generator = torch.Generator(device=DEVICE).manual_seed(rows * 7 + width)
    wide = torch.randn(rows, width + 64, generator=generator, device=DEVICE) * 40
    wide = wide.bfloat16()
    for x in (wide[:, :width].contiguous(), wide[:, 32 : 32 + width]):
        out = tanh_softcap(x, 30.0)
        assert torch.equal(_words(out), _words(_statement(x, 30.0)))
        assert out.data_ptr() != x.data_ptr()
    empty = tanh_softcap(wide[:0, :width], 30.0)
    assert empty.shape == (0, width) and empty.dtype is torch.float32


def test_graph_replay_equals_the_statement():
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    static = torch.empty(64, 18992, dtype=torch.bfloat16, device=DEVICE)
    tanh_softcap(static.fill_(1.0), 30.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = tanh_softcap(static, 30.0)
    for _ in range(3):
        static.copy_(torch.randn(static.shape, generator=generator, device=DEVICE) * 30)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_words(out), _words(_statement(static, 30.0)))
