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

"""Engram preparation and accepted-state updates on Ascend native graphs."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel.ops.metadata.ngram import (
    NGramHashParams,
    commit_ngram_inputs,
    engram_hash_reciprocals,
    prepare_ngram_inputs,
)


def _params(device):
    primes = (torch.arange(72).reshape(3, 3, 8) * 2 + 30_000_001).to(device)
    flat = primes.flatten(1)
    multipliers = (
        torch.randint(
            0,
            ((2**63 - 1) // 128) // 2,
            (3, 4),
            generator=torch.Generator().manual_seed(2041),
        )
        * 2
        + 1
    ).to(device)
    return NGramHashParams(
        torch.arange(128, device=device).flip(0),
        primes,
        engram_hash_reciprocals(primes),
        flat.cumsum(-1) - flat,
        multipliers,
        2,
    )


@pytest.mark.parametrize("bs,length", [(0, 0), (2, 0), (1, 1), (32, 6), (137, 17)])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize(
    "reset_dtype,ids_dtype,ends_dtype",
    [
        (torch.bool, torch.int32, torch.int64),
        (torch.int64, torch.int64, torch.int32),
        (torch.int32, torch.int32, torch.int32),
    ],
)
def test_ngram_prepare_and_commit_replay(
    bs, length, uniform, reset_dtype, ids_dtype, ends_dtype
):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(2041 + bs)
    pool, vocab = bs + 1, 128
    lengths = torch.randint(
        0, length + 1, (bs,), generator=generator, dtype=torch.int32
    )
    if bs:
        lengths[0] = length
    if uniform:
        lengths.fill_(length)
    total = int(lengths.sum())
    capacity = total + 37
    slots = torch.arange(bs).roll(1)
    positions = torch.full((bs,), 100, dtype=torch.int64)
    cache = torch.full((pool,), 100, dtype=torch.int32)
    cache[slots[::2]] += 1
    state = [
        torch.randint(-3, vocab + 3, (bs, 4), generator=generator),
        positions,
        (torch.arange(bs) % 3 == 0).to(reset_dtype),
        slots,
        cache,
        lengths.cumsum(0).to(ends_dtype),
        torch.randint(-3, vocab + 3, (pool, 3), generator=generator),
        torch.arange(pool) % 2 == 0,
        torch.randint(-3, vocab + 3, (capacity,), generator=generator, dtype=ids_dtype),
        torch.full((capacity, 3), -1, dtype=torch.int64),
        torch.zeros(capacity, dtype=torch.bool),
        torch.empty(bs, 3, dtype=torch.int64),
        torch.empty(capacity, 3, 24, dtype=torch.int64),
    ]
    device_state = [t.npu() for t in state]
    device_lengths = lengths.npu()
    accepts = lengths.clamp(max=2)
    num_extends = bs // 3
    accepts[:num_extends] = 0
    device_accepts = accepts.npu()
    cpu_params, device_params = _params("cpu"), _params("npu")

    def invoke(values, input_lengths, accepted, params):
        prepare_ngram_inputs(*values, total, vocab, length if uniform else 0, params)
        commit_ngram_inputs(
            values[3],
            input_lengths,
            accepted,
            values[8],
            values[9],
            values[10],
            values[11],
            values[6],
            values[7],
            values[4],
            num_extends,
            pool - 1,
        )

    invoke(device_state, device_lengths, device_accepts, device_params)
    for target, source in zip(device_state, state):
        target.copy_(source)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke(device_state, device_lengths, device_accepts, device_params)
    for phase in range(3):
        # Changing token IDs and seed choices must be consumed during replay.
        state[8].copy_(torch.randint(-3, vocab + 3, (capacity,), generator=generator))
        state[2].copy_((torch.arange(bs) + phase) % 3 == 0)
        expected = [t.clone() for t in state]
        for target, source in zip(device_state, state):
            target.copy_(source)
        invoke(expected, lengths, accepts, cpu_params)
        graph.replay()
        torch.npu.synchronize()
        for actual, reference in zip(device_state, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
