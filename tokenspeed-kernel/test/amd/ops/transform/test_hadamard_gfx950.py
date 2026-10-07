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

import pytest
import torch
from tokenspeed_kernel.ops.transform import hadamard_transform
from tokenspeed_kernel_amd.ops.gfx950.transform.hadamard import (
    gluon_hadamard_transform_128_gfx950,
)
from utils import assert_no_triton_compile


@pytest.mark.parametrize("tokens", [1, 17])
def test_gfx950_hadamard_matches_portable_bytes(
    device: str,
    tokens: int,
    require,
) -> None:
    require("transform", "hadamard_transform", "gluon", torch.bfloat16, "x")
    generator = torch.Generator(device=device).manual_seed(tokens)
    x = torch.randn(
        (tokens, 32, 128),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    scale = 128**-0.5

    expected = hadamard_transform(x, scale=scale, solution="triton")
    actual = hadamard_transform(x, scale=scale, solution="gluon")

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_gfx950_hadamard_empty_input(device: str, require) -> None:
    require("transform", "hadamard_transform", "gluon", torch.bfloat16, "x")
    x = torch.empty((0, 32, 128), dtype=torch.bfloat16, device=device)

    actual = hadamard_transform(x, scale=1.0, solution="gluon")

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype


def test_gfx950_hadamard_row_count_reuses_binary(device: str, require) -> None:
    require("transform", "hadamard_transform", "gluon", torch.bfloat16, "x")

    def run(tokens: int) -> None:
        x = torch.empty((tokens, 32, 128), dtype=torch.bfloat16, device=device)
        hadamard_transform(x, scale=128**-0.5, solution="gluon")

    run(1)
    with assert_no_triton_compile(gluon_hadamard_transform_128_gfx950):
        run(2)
        run(17)


def test_gfx950_hadamard_changed_input_graph_replay(device: str, require) -> None:
    require("transform", "hadamard_transform", "gluon", torch.bfloat16, "x")
    generator = torch.Generator(device=device).manual_seed(20260918)
    x = torch.randn(
        (7, 32, 128),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    scale = 128**-0.5

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = hadamard_transform(x, scale=scale, solution="gluon")

    x.copy_(torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator))
    expected = hadamard_transform(x, scale=scale, solution="triton")
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
