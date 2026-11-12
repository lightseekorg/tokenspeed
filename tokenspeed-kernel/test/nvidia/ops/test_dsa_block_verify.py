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

"""Seven- and eight-row DSA verification matches independent causal queries."""

from __future__ import annotations

import pytest
import torch
from ops.test_attention_dsa import _pack_index_k_cache
from tokenspeed_kernel.numerics.torch.attention.dsa import torch_dsa_sparse_attention
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_decode_topk, dsa_plan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell",
)


@pytest.mark.parametrize("width", [7, 8])
def test_deep_gemm_block_verify_matches_expanded_queries(width):
    torch.manual_seed(91)
    page_size, pages, heads, topk = 64, 64, 32, 2048
    requests = 3
    rows = requests * width
    table = torch.arange(requests * pages, device="cuda", dtype=torch.int32).view(
        requests, pages
    )
    lengths = torch.tensor([width, 73, 4095], device="cuda", dtype=torch.int32)
    keys, _ = _pack_index_k_cache(
        torch.randn(
            requests * pages * page_size, 128, device="cuda", dtype=torch.bfloat16
        ),
        page_size,
    )
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(rows, heads, device="cuda")

    def run(grouped):
        causal = (
            lengths[:, None]
            + torch.arange(1 - width, 1, device="cuda", dtype=torch.int32)
        ).reshape(-1)
        plan_lengths = causal[:, None].contiguous()
        return dsa_decode_topk(
            q,
            weights,
            lengths if grouped else causal,
            table if grouped else table.repeat_interleave(width, dim=0),
            page_size=page_size,
            topk=topk,
            softmax_scale=128**-0.5,
            batch_invariant=False,
            q_len_per_req=width if grouped else 1,
            index_k_cache=keys,
            seq_lens_2d=plan_lengths,
            plan=dsa_plan(
                page_size=page_size, seq_lens_2d=plan_lengths, solution="deep_gemm"
            ),
            solution="deep_gemm",
            slot_order="selection",
            simulation_seed=None,
        )

    def check(actual, expected):
        slots, counts = actual
        torch.testing.assert_close(counts, expected[1], rtol=0, atol=0)
        torch.testing.assert_close(
            slots.sort(dim=1).values, expected[0].sort(dim=1).values, rtol=0, atol=0
        )
        for row in range(rows):
            request, offset = divmod(row, width)
            visible = int(lengths[request]) - width + 1 + offset
            selected = slots[row][slots[row] >= 0] - request * pages * page_size
            assert selected.numel() == min(visible, topk)
            assert torch.all((selected >= 0) & (selected < visible))

    check(run(True), run(False))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    lengths.copy_(torch.tensor([width + 1, 65, 3072], device="cuda", dtype=torch.int32))
    q.mul_(0.75)
    graph.replay()
    torch.cuda.synchronize()
    check(captured, run(False))


@pytest.mark.parametrize("width", [7, 8])
@pytest.mark.parametrize("solution", ["flashinfer_trtllm", "triton"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_block_verify_attention_matches_causal_reference(width, solution, dtype):
    torch.manual_seed(117)
    rows, heads, capacity, topk = 2 * width, 8, 4096, 2048
    q = (torch.randn(rows, heads, 576, device="cuda") * 0.1).to(dtype)
    kv = (torch.randn(capacity, 576, device="cuda") * 0.1).to(dtype)
    lengths = torch.tensor([width, 3072], device="cuda", dtype=torch.int32)
    bounds = (
        lengths[:, None] + torch.arange(1 - width, 1, device="cuda", dtype=torch.int32)
    ).reshape(-1)
    counts = bounds.clamp(max=topk)
    slots = torch.full((rows, topk), -1, device="cuda", dtype=torch.int32)
    for row, bound in enumerate(bounds.tolist()):
        selection = torch.randperm(bound, device="cuda")[:topk].to(torch.int32)
        slots[row, : selection.numel()] = selection

    def run(group_width):
        return dsa_decode(
            q=q,
            kv_cache=kv,
            sparse_kv_cache=None,
            topk_slots=slots,
            topk_lens=counts,
            max_seqlen_k=capacity,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=192**-0.5,
            page_size=64,
            q_len_per_req=group_width,
            solution=solution,
            slot_order="selection",
        )

    actual = run(width)
    torch.testing.assert_close(actual, run(1), rtol=0, atol=0)
    # FP8 FlashInfer rounds unlike the FP32 reference, so its oracle is the q_len=1 run.
    if dtype == torch.bfloat16 or solution == "triton":
        reference = torch_dsa_sparse_attention(
            q,
            kv[:, :512],
            kv[:, 512:],
            slots,
            counts,
            192**-0.5,
        )
        torch.testing.assert_close(
            actual.float(), reference.float(), rtol=1e-2, atol=1e-3
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(width)
    q.copy_((torch.randn(rows, heads, 576, device="cuda") * 0.1).to(dtype))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, run(1), rtol=0, atol=0)
