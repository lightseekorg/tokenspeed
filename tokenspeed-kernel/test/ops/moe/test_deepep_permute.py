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

"""DeepEP normal-mode permutation: scatter must land every routed slot in its
expert block and gather must undo it with the routing weights applied."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe.triton.deepep_permute import (
    deepep_gather,
    deepep_scatter,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

_ALIGNMENT = 128
_BLOCK = 128


def _make_recv(num_recv, hidden, top_k, num_local_experts, seed):
    """Fake a DeepEP normal dispatch result on the receive side."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(
        (num_recv, hidden), device="cuda", dtype=torch.float32, generator=generator
    )
    recv_x = values.to(torch.float8_e4m3fn)
    recv_scale = torch.rand(
        (num_recv, hidden // _BLOCK),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    # Each received token carries top_k slots; slots whose expert lives on
    # another rank come back as -1.
    topk_ids = torch.randint(
        -1,
        num_local_experts,
        (num_recv, top_k),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    # A token never picks the same expert twice.
    for token in range(num_recv):
        seen = set()
        for slot in range(top_k):
            expert = int(topk_ids[token, slot])
            if expert in seen:
                topk_ids[token, slot] = -1
            elif expert >= 0:
                seen.add(expert)
    topk_weights = torch.rand(
        (num_recv, top_k), device="cuda", dtype=torch.float32, generator=generator
    )
    counts = []
    for expert in range(num_local_experts):
        real = int((topk_ids == expert).sum())
        counts.append((real + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT)
    return recv_x, recv_scale, topk_ids, topk_weights, counts


@pytest.mark.parametrize("num_recv", [1, 37, 512])
@pytest.mark.parametrize("hidden,top_k,num_local_experts", [(256, 4, 3), (1024, 8, 8)])
def test_scatter_places_every_routed_slot(num_recv, hidden, top_k, num_local_experts):
    recv_x, recv_scale, topk_ids, _, counts = _make_recv(
        num_recv, hidden, top_k, num_local_experts, seed=num_recv + hidden
    )
    x, x_scale, m_indices, dest_index = deepep_scatter(
        recv_x, recv_scale, topk_ids, counts, expert_alignment=_ALIGNMENT
    )

    total_rows = sum(counts)
    assert x.shape == (total_rows, hidden)
    assert x_scale.shape == (total_rows, hidden // _BLOCK)
    assert m_indices.shape == (total_rows,)

    starts = []
    running = 0
    for count in counts:
        starts.append(running)
        running += count
    # Every row of an expert block is tagged with that expert, padding rows
    # included: DeepGEMM reads the tag once per tile, and tiles never straddle
    # two experts because the blocks are alignment-sized.
    for expert, (start, count) in enumerate(zip(starts, counts, strict=True)):
        assert torch.all(m_indices[start : start + count] == expert)

    ids = topk_ids.cpu()
    dest = dest_index.cpu()
    x_float = x.to(torch.float32).cpu()
    recv_float = recv_x.to(torch.float32).cpu()
    scale_cpu = x_scale.cpu()
    recv_scale_cpu = recv_scale.cpu()
    per_expert_rows: dict[int, set[int]] = {e: set() for e in range(num_local_experts)}
    for token in range(num_recv):
        for slot in range(top_k):
            expert = int(ids[token, slot])
            row = int(dest[token, slot])
            if expert < 0:
                assert row == -1
                continue
            assert starts[expert] <= row < starts[expert] + counts[expert]
            assert row not in per_expert_rows[expert]
            per_expert_rows[expert].add(row)
            torch.testing.assert_close(x_float[row], recv_float[token])
            torch.testing.assert_close(scale_cpu[row], recv_scale_cpu[token])

    # Rows an expert did not receive stay zero-scaled so they contribute
    # nothing but are still tiled by the grouped GEMM.
    for expert in range(num_local_experts):
        used = per_expert_rows[expert]
        for row in range(starts[expert], starts[expert] + counts[expert]):
            if row not in used:
                assert torch.all(scale_cpu[row] == 0)


@pytest.mark.parametrize("num_recv", [1, 37, 512])
@pytest.mark.parametrize("hidden,top_k,num_local_experts", [(256, 4, 3), (1024, 8, 8)])
def test_gather_matches_weighted_reference(num_recv, hidden, top_k, num_local_experts):
    recv_x, recv_scale, topk_ids, topk_weights, counts = _make_recv(
        num_recv, hidden, top_k, num_local_experts, seed=hidden - num_recv
    )
    _, _, _, dest_index = deepep_scatter(
        recv_x, recv_scale, topk_ids, counts, expert_alignment=_ALIGNMENT
    )
    gemm_out = torch.randn((sum(counts), hidden), device="cuda", dtype=torch.bfloat16)

    got = deepep_gather(gemm_out, topk_ids, topk_weights, dest_index)

    reference = torch.zeros((num_recv, hidden), device="cuda", dtype=torch.float32)
    ids = topk_ids.cpu()
    dest = dest_index.cpu()
    for token in range(num_recv):
        for slot in range(top_k):
            if int(ids[token, slot]) < 0:
                continue
            row = int(dest[token, slot])
            reference[token] += (
                gemm_out[row].to(torch.float32) * topk_weights[token, slot]
            )
    # bf16 output: the kernel's fused multiply-add may round one ulp away from
    # the reference's separate multiply and add.
    torch.testing.assert_close(got, reference.to(torch.bfloat16))


def test_scatter_rejects_unaligned_counts():
    recv_x, recv_scale, topk_ids, _, counts = _make_recv(8, 256, 4, 2, seed=0)
    counts[0] += 1
    with pytest.raises(ValueError, match="aligned"):
        deepep_scatter(
            recv_x, recv_scale, topk_ids, counts, expert_alignment=_ALIGNMENT
        )
