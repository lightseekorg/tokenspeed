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


# The DFlash2 draft (5 layers, head_dim 128, NeoX RoPE, 2-tap block conv).
DRAFT_HEAD, DRAFT_Q_HEADS, DRAFT_QKV = 128, 8, 1280
DRAFT_INTER_RANK = 28672 // WORLD  # gate rows and up rows a rank
DRAFT_TAPS, DRAFT_GROUP_SIZE = 2, 16
DRAFT_COEFFICIENTS = 2 * DRAFT_TAPS * (HIDDEN // DRAFT_GROUP_SIZE)
# The draft head: a vocabulary shard a rank, the candidate selector's rank,
# the candidates a step, the logit scale and softcap.
HEAD_VOCAB, HEAD_SHARD, HEAD_RANK_DIM, HEAD_TOPK = 202048, 25256, 256, 16
# The logit scale is the draft config's output_multiplier, the softcap its
# final_logit_softcapping.
HEAD_SCALE, HEAD_SOFTCAP = 0.1767766952966369, 20.0


def draft_rope_table(positions: int) -> torch.Tensor:
    """FP32 ``[positions, 128]``: cos then sin of the draft's 64 NeoX
    frequencies."""
    inv = 1.0 / 500000.0 ** (
        torch.arange(0, DRAFT_HEAD, 2, dtype=torch.float64) / DRAFT_HEAD
    )
    angles = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=1).float()


def _draft_half(g: torch.Generator, rows: int):
    parts = [
        (torch.randn(rows, HIDDEN, generator=g) * 0.5).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    residual = (torch.randn(rows, HIDDEN, generator=g) * 2).to(torch.bfloat16)
    gain = (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
    coef_w = (torch.randn(DRAFT_COEFFICIENTS, HIDDEN, generator=g) * 0.01).to(
        torch.bfloat16
    )
    base = (0.5 + 0.1 * torch.randn(2, DRAFT_TAPS, HIDDEN, generator=g)).to(
        torch.bfloat16
    )
    return parts, residual, gain, coef_w, base


def draft_attn_shared(prefix: int, rows: int):
    """The draft attention half's replicated inputs: every rank's BF16 MLP
    partial of the previous layer, the residual, the norm gain, the
    coefficient projection, base_kernel, and the q / k norm gains."""
    g = torch.Generator().manual_seed(7100 + prefix)
    parts, residual, gain, coef_w, base = _draft_half(g, rows)
    q_norm = (1 + 0.1 * torch.randn(DRAFT_HEAD, generator=g)).to(torch.bfloat16)
    k_norm = (1 + 0.1 * torch.randn(DRAFT_HEAD, generator=g)).to(torch.bfloat16)
    return parts, residual, gain, coef_w, base, q_norm, k_norm


def draft_attn_rank(prefix: int, rank: int, rows: int):
    """``rank``'s qkv rows and o_proj columns, and its KV head's ``prefix``
    context keys and values."""
    g = torch.Generator().manual_seed(8300 + 10 * prefix + rank)
    qkv = (torch.randn(DRAFT_QKV, HIDDEN, generator=g) * 0.02).to(torch.bfloat16)
    o = (torch.randn(HIDDEN, DRAFT_Q_HEADS * DRAFT_HEAD, generator=g) * 0.03).to(
        torch.bfloat16
    )
    ctx_k = torch.randn(max(prefix, 1), DRAFT_HEAD, generator=g).to(torch.bfloat16)
    ctx_v = torch.randn(max(prefix, 1), DRAFT_HEAD, generator=g).to(torch.bfloat16)
    return qkv, o, ctx_k, ctx_v


def draft_mlp_shared(launch: int, rows: int):
    """The draft MLP half's replicated inputs: every rank's BF16 attention
    partial, the residual, the norm gain, the coefficient projection and
    base_kernel."""
    return _draft_half(torch.Generator().manual_seed(4100 + launch), rows)


def draft_mlp_rank(launch: int, rank: int):
    """``rank``'s gate / up rows and down columns."""
    g = torch.Generator().manual_seed(5200 + 10 * launch + rank)
    gu = (torch.randn(2 * DRAFT_INTER_RANK, HIDDEN, generator=g) * 0.02).to(
        torch.bfloat16
    )
    down = (torch.randn(HIDDEN, DRAFT_INTER_RANK, generator=g) * 0.02).to(
        torch.bfloat16
    )
    return gu, down


def draft_head_shared(seed: int, steps: int, device):
    """The draft head's replicated inputs: the normed hidden rows of the
    ``steps`` draft steps, the selector's projections, and the anchor token."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, std):
        return (torch.randn(*shape, generator=g, device=device) * std).to(
            torch.bfloat16
        )

    x = randn(steps, HIDDEN, std=1.0)
    hidden_proj = randn(HEAD_RANK_DIM, HIDDEN, std=0.01)
    pred = randn(HEAD_VOCAB, HEAD_RANK_DIM, std=0.1)
    succ = randn(HEAD_VOCAB, HEAD_RANK_DIM, std=0.1)
    anchor = torch.randint(0, HEAD_VOCAB, (1,), generator=g, device=device)
    return x, hidden_proj, pred, succ, anchor


def draft_head_shard(seed: int, rank: int, device) -> torch.Tensor:
    """``rank``'s BF16 vocabulary shard of the LM head."""
    g = torch.Generator(device=device).manual_seed(seed * 100 + rank + 1)
    return (torch.randn(HEAD_SHARD, HIDDEN, generator=g, device=device) * 0.05).to(
        torch.bfloat16
    )
