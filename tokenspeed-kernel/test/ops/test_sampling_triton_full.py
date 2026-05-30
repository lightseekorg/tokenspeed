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

import torch
from tokenspeed_kernel.ops.sampling.triton import (
    accumulate_counts_inplace,
    apply_penalties_logit_bias_inplace,
    selected_token_logprobs,
    verify_chain_target_sampled,
)


def test_verify_chain_target_sampled_prefix_compare(device: str) -> None:
    candidates = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=torch.int32,
        device=device,
    )
    target_sampled = torch.tensor(
        [
            [11, 12, 99, 77],
            [99, 21, 22, 23],
            [31, 32, 33, 34],
        ],
        dtype=torch.int32,
        device=device,
    ).reshape(-1)
    bs, n = candidates.shape
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, n), -1, dtype=torch.int32, device=device)
    accept_length = torch.full((bs,), -1, dtype=torch.int32, device=device)

    verify_chain_target_sampled(
        predicts,
        accept_index,
        accept_length,
        candidates,
        target_sampled,
    )

    torch.testing.assert_close(
        accept_length.cpu(), torch.tensor([2, 0, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        accept_index.cpu(),
        torch.tensor(
            [[0, 1, 2, -1], [4, -1, -1, -1], [8, 9, 10, 11]],
            dtype=torch.int32,
        ),
    )
    expected_prefixes = [
        (0, torch.tensor([11, 12, 99], dtype=torch.int32)),
        (4, torch.tensor([99], dtype=torch.int32)),
        (8, torch.tensor([31, 32, 33, 34], dtype=torch.int32)),
    ]
    for start, expected in expected_prefixes:
        torch.testing.assert_close(
            predicts[start : start + expected.numel()].cpu(), expected
        )


def test_apply_penalties_logit_bias_inplace_matches_torch_reference(
    device: str,
) -> None:
    torch.manual_seed(1234)
    bs, n, vocab_size, pool_rows = 2, 3, 257, 5
    rows = bs * n
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device)
    req_pool_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    counts = torch.zeros((pool_rows, vocab_size), dtype=torch.int32, device=device)
    counts[1, torch.tensor([3, 17, 17, 88], device=device)] = torch.tensor(
        [1, 2, 2, 5], dtype=torch.int32, device=device
    )
    counts[3, torch.tensor([5, 99, 101], device=device)] = torch.tensor(
        [4, 1, 3], dtype=torch.int32, device=device
    )
    logit_bias = torch.zeros(
        (pool_rows, vocab_size), dtype=torch.bfloat16, device=device
    )
    logit_bias[1, 3] = 1.25
    logit_bias[1, 99] = -0.5
    logit_bias[3, 5] = -2.0
    logit_bias[3, 101] = 0.75
    freq = torch.tensor(
        [0.0, 0.4, 0.0, -0.25, 0.0], dtype=torch.bfloat16, device=device
    )
    pres = torch.tensor([0.0, 0.2, 0.0, 0.5, 0.0], dtype=torch.bfloat16, device=device)
    rep = torch.tensor([1.0, 1.2, 1.0, 1.5, 1.0], dtype=torch.bfloat16, device=device)

    ref = logits.clone()
    expanded_pool = torch.repeat_interleave(req_pool_indices.long(), n, dim=0)
    counts_sel = counts.index_select(0, expanded_pool)
    active = counts_sel > 0
    counts_f = counts_sel.to(ref.dtype)
    active_f = active.to(ref.dtype)
    rep_sel = rep.index_select(0, expanded_pool).to(ref.dtype).unsqueeze(-1)
    freq_sel = freq.index_select(0, expanded_pool).to(ref.dtype).unsqueeze(-1)
    pres_sel = pres.index_select(0, expanded_pool).to(ref.dtype).unsqueeze(-1)
    scales = torch.where(active, rep_sel.expand_as(ref), torch.ones_like(ref))
    ref = torch.where(ref > 0, ref / scales, ref * scales)
    ref = ref - freq_sel * counts_f - pres_sel * active_f
    ref = ref + logit_bias.index_select(0, expanded_pool).to(ref.dtype)

    out = logits.clone()
    returned = apply_penalties_logit_bias_inplace(
        out,
        req_pool_indices,
        counts,
        logit_bias,
        freq,
        pres,
        rep,
        num_tokens_per_req=n,
        block_size=128,
    )

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


def test_accumulate_counts_inplace_matches_index_put_reference(device: str) -> None:
    counts = torch.zeros((4, 64), dtype=torch.int32, device=device)
    counts[1, 10] = 5
    pool_idx = torch.tensor([1, 1, 2, 2, 2, 3], dtype=torch.int32, device=device)
    tokens = torch.tensor([10, 10, 11, 12, 12, 63], dtype=torch.int32, device=device)
    weights = torch.tensor([1, 2, 1, 0, 3, 4], dtype=torch.int32, device=device)

    ref = counts.clone()
    ref.index_put_((pool_idx, tokens.long()), weights, accumulate=True)
    accumulate_counts_inplace(counts, pool_idx, tokens, weights, block_size=4)

    torch.testing.assert_close(counts, ref)


def test_selected_token_logprobs_matches_torch_reference(device: str) -> None:
    torch.manual_seed(55)
    rows, vocab_size = 5, 513
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device) * 2.0
    logits[2, 17] = float("-inf")
    tokens = torch.tensor([3, 99, 17, 128, 512], dtype=torch.int32, device=device)

    actual = selected_token_logprobs(logits, tokens, block_size=128)
    expected = (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, tokens.long().unsqueeze(-1))
        .squeeze(-1)
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
