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

"""Representative inputs of the Muse Spark TP8 decode monokernels, shared by
their tests (``tokenspeed-kernel/test/amd/gfx950/ops/monokernel``) and their
launch benches (``tokenspeed-kernel/benchmarks/amd/gfx950/muse_spark``).

Seeded random tensors in the model's shapes, one TP8 rank at a time; the
benches time the kernels on them and the tests check the kernels against
references on them. The gfx950 Gluon kernels' own layouts (preprocessed
expert weights, the stand-in for absent ranks) are in
``muse_spark_monokernel_gluon``.
"""

from __future__ import annotations

import torch

WORLD, HIDDEN, POOLED, OWNED, INTER = 8, 8192, 4096, 1024, 512
QKVG, Q = 2304, 1024
# The experts a layer test holds per rank (its router bias leaves the others
# out); the benches route among more.
TEST_EXPERTS = 16


def rope_table(positions: int) -> torch.Tensor:
    """FP32 ``[positions, 64]``: cos of the 32 pairs, then sin (theta 500000)."""
    return rope_rows(0, positions)


def rope_rows(first: int, count: int) -> torch.Tensor:
    """Rows ``first .. first + count`` of the RoPE table (:func:`rope_table`)."""
    inv = 1.0 / 500000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    t = torch.arange(first, first + count, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([t.cos(), t.sin()], dim=1)


def layer_shared(seed: int, length: int, rows: int) -> dict[str, torch.Tensor]:
    """Inputs every rank shares: ``a_in`` and ``h`` of each row (``[rows,
    8192]``; row ``t`` at position ``length - 1 + t``), the weights of the
    norms and the router, and the page table. The router bias routes to the
    first :data:`TEST_EXPERTS` experts."""
    g = torch.Generator().manual_seed(seed)

    def gain(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16).float()

    a_in = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
    h = torch.randn(rows, HIDDEN, generator=g) * 2
    gates = [torch.rand(HIDDEN, generator=g) + 0.5 for _ in range(4)]
    router = torch.randn(256, HIDDEN, generator=g) / HIDDEN**0.5
    bias = torch.randn(256, generator=g) * 0.1
    bias[TEST_EXPERTS:] = -1e4
    pages = (length + rows - 1 + 63) // 64
    table = (torch.randperm(pages + 3, generator=g)[:pages] + 1).to(torch.int32)
    return dict(
        a_in=a_in,
        h=h,
        alpha1=gates[0],
        beta1=gates[1],
        alpha2=gates[2],
        beta2=gates[3],
        pre_gain=gain(HIDDEN),
        pre_exp_gain=gain(POOLED),
        pen_gain=gain(POOLED),
        post_gain=gain(HIDDEN),
        in_gain=gain(HIDDEN),
        router=router,
        bias=bias,
        table=table,
    )


def layer_rank_weights(rank: int, length: int) -> dict[str, torch.Tensor]:
    """``rank``'s BF16 weight shards and a KV cache of ``length`` tokens plus
    five pages (``[slots, 2, 64]``)."""
    g = torch.Generator().manual_seed(1000 + rank)

    def bf(*shape, k):
        return (torch.randn(*shape, generator=g) / k**0.5).to(torch.bfloat16)

    slots = ((length + 63) // 64 + 5) * 64
    return dict(
        w_qkvg=bf(QKVG, HIDDEN, k=HIDDEN),
        w_o=bf(HIDDEN, Q, k=Q),
        w_pool=bf(512, HIDDEN, k=HIDDEN),
        w_post=bf(OWNED, POOLED, k=POOLED),
        k_cache=torch.randn(slots, 2, 64, generator=g).to(torch.bfloat16),
        v_cache=torch.randn(slots, 2, 64, generator=g).to(torch.bfloat16),
    )


def raw_experts(rank: int, experts: int):
    """Random MXFP4 expert shards of ``rank``: w13 rows interleave gate / up.

    Returns:
        ``(w13, s13, w2, s2)`` uint8: packed E2M1 pairs and E8M0 scales.
    """
    g = torch.Generator().manual_seed(500 + rank)
    w13 = torch.randint(
        0, 256, (experts, 2 * INTER, POOLED // 2), dtype=torch.uint8, generator=g
    )
    s13 = torch.randint(
        122, 127, (experts, 2 * INTER, POOLED // 32), dtype=torch.uint8, generator=g
    )
    w2 = torch.randint(
        0, 256, (experts, POOLED, INTER // 2), dtype=torch.uint8, generator=g
    )
    s2 = torch.randint(
        118, 123, (experts, POOLED, INTER // 32), dtype=torch.uint8, generator=g
    )
    return w13, s13, w2, s2
