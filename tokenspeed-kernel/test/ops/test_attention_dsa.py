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
from tokenspeed_kernel import dsa_top_paged, dsa_topk

torch.manual_seed(42)


@pytest.mark.parametrize("solution", ["triton", "deep_gemm"])
def test_dsa_top_paged(device: str, solution: str, require) -> None:
    require("attention", "dsa_top_paged", solution, torch.bfloat16, "q")
    if solution == "deep_gemm":
        pytest.skip("DeepGEMM DSA paged top-k requires packed FP8 cache fixture")

    page_size = 64
    topk = 512
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((3, 2), device=device, dtype=torch.float32)
    index_k = torch.randn((4 * page_size, 128), device=device, dtype=torch.bfloat16)
    seq_lens = torch.tensor([20, 65, 3], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [[1, 3], [0, 2], [2, 1]], device=device, dtype=torch.int32
    )
    out = torch.empty((3, topk), device=device, dtype=torch.int32)
    lens_out = torch.empty((3,), device=device, dtype=torch.int32)

    topk_slots, topk_lens = dsa_top_paged(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_cache=index_k,
        out=out,
        lens_out=lens_out,
        solution=solution,
    )

    expected = torch.full_like(topk_slots, -1)
    expected_lens = torch.minimum(seq_lens, torch.full_like(seq_lens, topk))
    for token in range(q.shape[0]):
        scores = []
        slots = []
        for offset in range(int(seq_lens[token].item())):
            page = int(block_table[token, offset // page_size].item())
            slot = page * page_size + offset % page_size
            per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
            scores.append((per_head * weights[token]).sum() * (128**-0.5))
            slots.append(slot)
        local = torch.topk(
            torch.stack(scores), int(expected_lens[token].item())
        ).indices
        expected[token, : local.numel()] = torch.tensor(
            [slots[int(i)] for i in local.tolist()], device=device, dtype=torch.int32
        )

    assert topk_slots.data_ptr() == out.data_ptr()
    assert topk_lens.data_ptr() == lens_out.data_ptr()
    torch.testing.assert_close(topk_lens.cpu(), expected_lens.cpu())
    torch.testing.assert_close(topk_slots[:, :65].cpu(), expected[:, :65].cpu())
    assert (topk_slots[0, int(expected_lens[0].item()) :] == -1).all()


@pytest.mark.parametrize("solution", ["triton", "deep_gemm"])
def test_dsa_topk(device: str, solution: str, require) -> None:
    require("attention", "dsa_topk", solution, torch.bfloat16, "q")
    if solution == "deep_gemm":
        pytest.skip("DeepGEMM DSA workspace top-k requires gathered FP8 fixture")

    topk = 512
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((3, 2), device=device, dtype=torch.float32)
    index_k = torch.randn((256, 128), device=device, dtype=torch.bfloat16)
    kv_workspace_slots = torch.arange(85, device=device, dtype=torch.int64) + 17
    row_starts = torch.tensor([0, 10, 70], device=device, dtype=torch.int32)
    row_ends = torch.tensor([20, 75, 85], device=device, dtype=torch.int32)
    out = torch.empty((3, topk), device=device, dtype=torch.int32)
    lens_out = torch.empty((3,), device=device, dtype=torch.int32)

    workspace_indices, topk_lens = dsa_topk(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_cache=index_k,
        out=out,
        lens_out=lens_out,
        solution=solution,
    )

    expected = torch.full_like(workspace_indices, -1)
    expected_lens = torch.minimum(
        row_ends - row_starts, torch.full_like(row_ends, topk)
    )
    for token in range(q.shape[0]):
        scores = []
        rows = []
        for row in range(int(row_starts[token].item()), int(row_ends[token].item())):
            slot = int(kv_workspace_slots[row].item())
            per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
            scores.append((per_head * weights[token]).sum() * (128**-0.5))
            rows.append(row)
        local = torch.topk(
            torch.stack(scores), int(expected_lens[token].item())
        ).indices
        expected[token, : local.numel()] = torch.tensor(
            [rows[int(i)] for i in local.tolist()], device=device, dtype=torch.int32
        )

    assert workspace_indices.data_ptr() == out.data_ptr()
    assert topk_lens.data_ptr() == lens_out.data_ptr()
    torch.testing.assert_close(topk_lens.cpu(), expected_lens.cpu())
    torch.testing.assert_close(workspace_indices[:, :65].cpu(), expected[:, :65].cpu())
    assert (workspace_indices[0, int(expected_lens[0].item()) :] == -1).all()
