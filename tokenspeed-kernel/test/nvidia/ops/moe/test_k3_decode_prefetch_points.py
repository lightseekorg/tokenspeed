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

"""Where a Kimi-K3 decode attention layer forks its L2 prefetches.

KDA and MLA layers alike fork ``attention_moe`` (the o-projection shard and the
MoE layer's non-expert weights) right after the input projection, beside the
attention core, and ``oproj_down`` after the o-projection exit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")


def _layer(forks):
    """A decode attention layer whose projection, exit, and prefetch only record."""
    from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode import (
        KimiK3DecodeAttentionLayer,
    )

    layer = object.__new__(KimiK3DecodeAttentionLayer)
    layer.projection = SimpleNamespace(
        run=lambda x, out: (forks.append(("projection",)), out)[1]
    )
    layer.exit = SimpleNamespace(run=lambda *args: forks.append(("exit",)))
    layer.prefetch = SimpleNamespace(fork=lambda *args: forks.append(args))
    layer.layer_idx, layer.rows = 3, 16
    return layer


def test_the_attention_prefetch_forks_after_the_input_projection():
    forks = []
    layer = _layer(forks)
    projected = layer.project(torch.zeros(8, 32))
    assert projected.shape == (8, 16)
    assert forks == [("projection",), ("attention_moe", 3, 8)]
    out = torch.zeros(8, 32)
    assert layer.leave(torch.zeros(8, 16), None, None, out) is out
    assert forks[2:] == [("exit",), ("oproj_down", 3, 8)]
