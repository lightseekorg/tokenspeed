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

"""Reference math of TokenSpeed's Qwen3.8-Flash-Next attention prologue and
QSA indexer, shared by the megakernel tests."""

from __future__ import annotations

import torch


def bf(x: torch.Tensor) -> torch.Tensor:
    """``x`` rounded to BF16, as FP32."""
    return x.bfloat16().float()


def cos_sin(max_pos: int) -> torch.Tensor:
    """TokenSpeed's rotary cache: FP32 ``[max_pos, 64]`` = ``[cos | sin]``,
    rotary dim 64 (partial factor 0.25 of 256), theta 1e7."""
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv_freq[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], -1).cuda()


def rotate(n: torch.Tensor, table: torch.Tensor, pos: int) -> torch.Tensor:
    """NeoX rotation of dims 0-63 at ``pos`` (pairs ``(i, i + 32)``)."""
    cos, sin = table[pos, :32], table[pos, 32:]
    n1, n2 = n[:32], n[32:64]
    return torch.cat([n1 * cos - n2 * sin, n2 * cos + n1 * sin, n[64:]])


def gemma(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The attention prologue's RMS norm: ``(1 + w)`` formed in FP32."""
    inv = torch.rsqrt(x.square().sum() / x.numel() + 1e-6)
    return (x * inv) * (w.float() + 1.0)


def index_norm(x: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    """The indexer's RMS norm: ``w1`` is the BF16 ``1 + w``."""
    s = x[:32].square().sum() + x[32:64].square().sum() + x[64:].square().sum()
    return (x * (1.0 / torch.sqrt(s / 128 + 1e-6))) * w1.float()
