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

import torch

from tokenspeed.runtime.layers.attention.chunk import (
    build_dcp_compact_reconstruction_plan,
)
from tokenspeed.runtime.layers.attention.dcp.a2a import _merge_received_partials
from tokenspeed.runtime.layers.attention.dcp.comm import (
    reconstruct_and_reduce_scatter,
)


def test_compact_dcp_prefix_plan_reconstructs_request_major_rows() -> None:
    starts = [3, 10, 17]
    lengths = [9, 5, 14]
    dcp_size = 4
    plans = [
        build_dcp_compact_reconstruction_plan(starts, lengths, dcp_size, rank)
        for rank in range(dcp_size)
    ]
    padded = plans[0][1]
    assert all(plan[1] == padded for plan in plans)
    assert sum(
        plans[rank][0][req] for rank in range(dcp_size) for req in range(3)
    ) == sum(lengths)

    gathered = [-1] * (dcp_size * padded)
    for rank, (counts, _, _) in enumerate(plans):
        cursor = rank * padded
        for start, length, count in zip(starts, lengths, counts):
            owned = [
                pos for pos in range(start, start + length) if pos % dcp_size == rank
            ]
            assert len(owned) == count
            gathered[cursor : cursor + count] = owned
            cursor += count

    reconstruction = plans[0][2]
    restored = [gathered[index] for index in reconstruction]
    expected = [
        pos
        for start, length in zip(starts, lengths)
        for pos in range(start, start + length)
    ]
    assert restored == expected


def test_reduce_scatter_contract_masks_empty_local_rows_before_fast_path() -> None:
    output = torch.tensor(
        [[[[1.0, 2.0]], [[float("nan"), float("inf")]]]], dtype=torch.float32
    )
    lse = torch.tensor([[[0.0], [0.0]]], dtype=torch.float32)
    nonempty = torch.tensor([[True, False]])

    actual = reconstruct_and_reduce_scatter(
        output,
        lse,
        dcp_rank=0,
        group=(0,),
        local_nonempty=nonempty,
    )

    assert torch.equal(actual[:, 0], output[:, 0])
    assert torch.equal(actual[:, 1], torch.zeros_like(actual[:, 1]))


def test_a2a_received_partials_use_reference_fp32_merge_arithmetic() -> None:
    partials = torch.tensor(
        [
            [[[-2.0, 1.0, 4.0], [float("nan"), 2.0, 3.0]]],
            [[[6.0, 5.0, -3.0], [7.0, 8.0, 9.0]]],
        ],
        dtype=torch.bfloat16,
    )
    lse = torch.tensor(
        [[[0.25, -torch.inf]], [[-1.5, 2.0]]],
        dtype=torch.float32,
    )
    head_dim = partials.shape[-1]
    recv = torch.empty(
        (*partials.shape[:-1], head_dim + 2),
        dtype=torch.bfloat16,
    )
    recv[..., :head_dim].copy_(partials)
    lse_bits = lse.contiguous().view(torch.int32)
    recv_words = recv.view(torch.uint16)
    recv_words[..., head_dim].copy_((lse_bits & 0xFFFF).to(torch.uint16))
    recv_words[..., head_dim + 1].copy_(((lse_bits >> 16) & 0xFFFF).to(torch.uint16))

    for base, exponent in ((torch.e, torch.exp), (2.0, torch.exp2)):
        maximum = lse.amax(dim=0)
        masses = exponent(lse - maximum)
        masses = torch.where(torch.isfinite(lse), masses, torch.zeros_like(masses))
        weights = masses / masses.sum(dim=0).clamp_min(torch.finfo(torch.float32).tiny)
        safe = torch.where(
            torch.isfinite(partials.float()),
            partials.float(),
            torch.zeros_like(partials, dtype=torch.float32),
        )
        expected = (safe * weights.unsqueeze(-1)).sum(dim=0).to(torch.bfloat16)

        actual = _merge_received_partials(
            recv,
            head_dim=head_dim,
            lse_base=float(base),
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
