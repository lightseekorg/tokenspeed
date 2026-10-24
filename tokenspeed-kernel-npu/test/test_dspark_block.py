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

"""DSpark packed-key identity, independent oracle and mutable native graphs."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.sampling.triton.dspark_block import (
    _block_step_kernel,
    dspark_block_greedy_resolve,
    dspark_block_greedy_step,
)


def old_step(base, step, anchor, candidates, emb, proj, start, valid, partials, out):
    n = base.shape[0]
    tp, _, tiles = candidates.shape
    rank = emb.shape[1]
    _block_step_kernel[(tiles, triton.cdiv(n, 16))](
        base[:, step],
        base.stride(0),
        anchor,
        anchor.stride(0),
        candidates,
        out[:, max(0, step - 1)],
        out.stride(0),
        emb,
        proj,
        partials,
        n,
        valid,
        start,
        emb.shape[0],
        rank,
        max(16, triton.next_power_of_2(rank)),
        16,
        256,
        128,
        tp,
        triton.next_power_of_2(tp),
        tiles,
        triton.next_power_of_2(tiles),
        False,
        num_warps=4,
    )


def pack(values, ids):
    bits = values.contiguous().view(torch.int32)
    order = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).long()
    return (order << 32) | (0xFFFFFFFF - ids.long())


@pytest.mark.parametrize(
    "n,vocab,valid,rank,tp",
    [
        (1, 257, 257, 32, 1),
        (8, 513, 270, 64, 2),
        (32, 8080, 8080, 256, 16),
        (33, 768, 9, 256, 3),
    ],
)
@pytest.mark.parametrize("step", [0, 1, 4])
def test_step_and_graph(n, vocab, valid, rank, tp, step):
    torch.npu.set_device(0)
    torch.manual_seed(2026 + n + step)
    total_vocab = 129280
    start = 4096
    tiles = triton.cdiv(vocab, 256)
    base = torch.randn(n, 5, vocab, device="npu")
    # BF16 operands with exactly representable products/sums for the FP32 oracle.
    emb = (
        torch.randint(-8, 9, (total_vocab, rank), device="npu").to(torch.bfloat16) / 16
    )
    proj = torch.randint(-8, 9, (vocab, rank), device="npu").to(torch.bfloat16) / 16
    anchor_store = torch.randint(
        0, total_vocab, (n, 3), device="npu", dtype=torch.int32
    )
    anchor = anchor_store[:, 1]
    candidates = pack(
        torch.randn(tp, n, tiles, device="npu"),
        torch.randint(0, total_vocab, (tp, n, tiles), device="npu"),
    )
    new = torch.empty(n, tiles, device="npu", dtype=torch.int64)
    old = torch.empty_like(new)
    new_out = torch.empty(n, 5, device="npu", dtype=torch.int32)
    old_out = torch.empty_like(new_out)

    def run():
        dspark_block_greedy_step(
            base, step, anchor, candidates, emb, proj, start, valid, new, new_out
        )

    def check():
        previous = anchor.cpu().long()
        if step:
            best = candidates.cpu().permute(1, 0, 2).reshape(n, -1).max(1).values
            previous = 0xFFFFFFFF - (best & 0xFFFFFFFF)
            assert torch.equal(new_out[:, step - 1].cpu().long(), previous)
        # Feed the CPU-resolved anchor to the old matrix-scoring kernel. Its
        # fused 3D INT64 candidate reduction miscompiled negative-score rows
        # under this Ascend compiler; it is not a correctness oracle there.
        reference_anchor = previous.to(device=anchor.device, dtype=anchor.dtype)
        old_step(
            base,
            step,
            reference_anchor,
            candidates,
            emb,
            proj,
            start,
            valid,
            old,
            old_out,
        )
        torch.npu.synchronize()
        assert torch.equal(new, old)
        score = (
            base[:, step, :valid].cpu()
            + emb.cpu()[previous].float() @ proj[:valid].cpu().float().T
        )
        expected = score.argmax(1) + start
        got = 0xFFFFFFFF - (new.cpu().max(1).values & 0xFFFFFFFF)
        assert torch.equal(got, expected)

    run()
    check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for _ in range(3):
        base.normal_()
        anchor_store.random_(0, total_vocab)
        candidates.copy_(
            pack(
                torch.randn_like(candidates, dtype=torch.float32),
                torch.randint_like(candidates, 0, total_vocab),
            )
        )
        graph.replay()
        check()


def test_signed_zero_ties_invalid_tiles_and_resolve():
    torch.npu.set_device(0)
    n, vocab, rank = 3, 768, 32
    base = torch.zeros(n, 2, vocab, device="npu")
    emb = torch.zeros(4096, rank, device="npu", dtype=torch.bfloat16)
    proj = torch.zeros(vocab, rank, device="npu", dtype=torch.bfloat16)
    candidates = torch.empty(2, n, 3, device="npu", dtype=torch.int64)
    anchor = torch.zeros(n, device="npu", dtype=torch.int64)
    result = torch.empty(n, 3, device="npu", dtype=torch.int64)
    old = torch.empty_like(result)
    out = torch.empty(n, 2, device="npu", dtype=torch.int32)
    old_out = torch.empty_like(out)
    base[1, :, 2] = float("inf")
    base[2, :, :17] = -float("inf")
    dspark_block_greedy_step(
        base, 0, anchor, candidates, emb, proj, 100, 17, result, out
    )
    old_step(base, 0, anchor, candidates, emb, proj, 100, 17, old, old_out)
    torch.npu.synchronize()
    assert torch.equal(result, old)
    candidates[0].copy_(result)
    candidates[1].copy_(result)
    dspark_block_greedy_resolve(candidates, out, 1)
    assert out[:, 1].cpu().tolist() == [100, 102, 100]
    values = (
        torch.tensor([0.0, -0.0, 1.0, 1.0, -float("inf"), -float("inf")])
        .reshape(2, 3, 1)
        .to("npu")
    )
    ids = torch.tensor([7, 2, 90, 3, 4, 5], device="npu").reshape(2, 3, 1)
    keys = pack(values, ids)
    dspark_block_greedy_resolve(keys, out, 0)
    expected = 0xFFFFFFFF - (keys.cpu().max(0).values[:, 0] & 0xFFFFFFFF)
    assert torch.equal(out[:, 0].cpu().long(), expected)
