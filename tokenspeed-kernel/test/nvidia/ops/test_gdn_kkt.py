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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn._triton.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)


@pytest.mark.parametrize("lengths", [(1024,), (65, 959)])
@pytest.mark.parametrize("beta_value", [0.5, 0.50390625])
def test_chunk_kkt_shared_tile_reuse(lengths: tuple[int, ...], beta_value: float):
    device = "cuda"
    tokens = sum(lengths)
    heads, head_dim, chunk_size = 16, 128, 64
    rng = torch.Generator(device=device).manual_seed(7)
    # Small integers and a dyadic beta make the reference exact in FP32.
    key = torch.randint(
        -3, 4, (1, tokens, heads, head_dim), device=device, generator=rng
    ).to(torch.bfloat16)
    beta = torch.full(
        (1, tokens, heads), beta_value, device=device, dtype=torch.bfloat16
    )
    gate = torch.zeros((1, tokens, heads), device=device, dtype=torch.float32)
    ends = [0]
    for length in lengths:
        ends.append(ends[-1] + length)
    bounds = torch.tensor(ends, device=device, dtype=torch.int32)
    expected = torch.zeros((1, tokens, heads, chunk_size), device=device)
    for start, stop in zip(ends[:-1], ends[1:], strict=True):
        for first in range(start, stop, chunk_size):
            last = min(first + chunk_size, stop)
            tile = key[0, first:last].transpose(0, 1).float()
            gram = (tile @ tile.transpose(-1, -2)) * beta_value
            expected[0, first:last, :, : last - first] = gram.tril(-1).transpose(0, 1)

    # Repeat identical launches to exercise the shared-memory race.
    expected_bits = expected.view(torch.int32)
    matches = torch.stack(
        [
            (
                chunk_scaled_dot_kkt_fwd(key, beta, gate, bounds).view(torch.int32)
                == expected_bits
            ).all()
            for _ in range(128)
        ]
    )
    assert matches.all().item()
