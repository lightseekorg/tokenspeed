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

"""SWA prefill must exclude old keys even when replayed in separate chunks."""

from itertools import accumulate

import pytest
import torch
from tokenspeed_kernel.ops.attention import attn_merge_state
from tokenspeed_kernel.ops.attention.mla import mla_prefill


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("window_left", [0, 512])
@pytest.mark.parametrize("solution", [None, "triton"])
def test_swa_prefill_ragged_suffix(device, dtype, window_left, solution):
    torch.manual_seed(23)
    q_lens, kv_lens = (1, 33, 514), (1, 1050, 514)
    cu_q = torch.tensor([0, *accumulate(q_lens)], device=device, dtype=torch.int32)
    cu_kv = torch.tensor([0, *accumulate(kv_lens)], device=device, dtype=torch.int32)
    q = torch.randn(sum(q_lens), 2, 256, device=device, dtype=torch.bfloat16).to(dtype)
    k = torch.randn(sum(kv_lens), 2, 256, device=device, dtype=torch.bfloat16).to(dtype)
    v = torch.randn(sum(kv_lens), 2, 128, device=device, dtype=torch.bfloat16).to(dtype)
    out, lse = mla_prefill(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        max(q_lens),
        max(kv_lens),
        1 / 16,
        is_causal=True,
        window_left=window_left,
        return_lse=True,
        solution=solution,
    )
    for i, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        qs, ks = sum(q_lens[:i]), sum(kv_lens[:i])
        qi, ki, vi = (
            q[qs : qs + q_len].float(),
            k[ks : ks + kv_len].float(),
            v[ks : ks + kv_len].float(),
        )
        scores = torch.einsum("qhd,khd->hqk", qi, ki) / 16
        q_pos = torch.arange(q_len, device=device) + kv_len - q_len
        distance = q_pos[:, None] - torch.arange(kv_len, device=device)[None, :]
        scores.masked_fill_(
            ~((distance >= 0) & (distance <= window_left))[None], -torch.inf
        )
        expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vi)
        # FP8 P@V rounds probabilities to FP8, as in the full-history path.
        tolerance = 0.04 if dtype == torch.float8_e4m3fn else 0.008
        torch.testing.assert_close(
            out[qs : qs + q_len].float(), expected, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            lse[qs : qs + q_len], scores.logsumexp(-1).T, atol=0.008, rtol=0.008
        )


def test_swa_prefix_chunks_keep_absolute_query_positions(device):
    torch.manual_seed(24)
    prefix, length = 1050, 67
    q = torch.randn(length, 2, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(prefix + length, 2, 256, device=device, dtype=torch.bfloat16)
    v = torch.randn(prefix + length, 2, 128, device=device, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, length], device=device, dtype=torch.int32)

    def attend(start, end, causal):
        cu_kv = torch.tensor([0, end - start], device=device, dtype=torch.int32)
        offsets = torch.tensor([prefix - start], device=device, dtype=torch.int32)
        return mla_prefill(
            q,
            k[start:end],
            v[start:end],
            cu_q,
            cu_kv,
            length,
            end - start,
            1 / 16,
            is_causal=causal,
            window_left=512,
            query_start_offsets=offsets,
            return_lse=True,
            solution="triton",
        )

    expected, expected_lse = attend(0, prefix + length, True)
    actual, actual_lse = attend(prefix, prefix + length, True)
    for start, end in ((0, 256), (256, 512), (512, 768), (768, prefix)):
        partial, partial_lse = attend(start, end, False)
        if end <= prefix - 512:
            assert torch.count_nonzero(partial).item() == 0
            assert torch.isneginf(partial_lse).all()
        attn_merge_state(actual, actual_lse, partial, partial_lse, inplace=True)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0.003, rtol=0.003)
