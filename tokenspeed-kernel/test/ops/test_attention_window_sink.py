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

"""``window_sink``: leading tokens kept visible outside the sliding window.

Checked against a dense reference mask rather than another kernel, so the test
fails only when the visibility set itself is wrong. ``sink=0`` is the
no-regression case: the same kernel must still serve plain sliding-window
attention.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

HEADS, KV_HEADS, HEAD_DIM = 8, 2, 128
PAGE_SIZE, NUM_PAGES = 16, 32
# Longer than every window below, so the sink names rows the window dropped.
CACHE_LEN = 300


def _reference(q, k, v, scale, cache_len, window_left, sink):
    """Dense attention over the rows a sink + window mask keeps."""
    heads = q.shape[0]
    repeats = heads // k.shape[1]
    keys = k.repeat_interleave(repeats, dim=1)[:cache_len]
    values = v.repeat_interleave(repeats, dim=1)[:cache_len]
    positions = torch.arange(cache_len, device=q.device)
    if window_left >= 0:
        keep = positions >= (cache_len - 1 - window_left)
    else:
        keep = torch.ones_like(positions, dtype=torch.bool)
    if sink > 0:
        keep = keep | (positions < sink)
    logits = (q.float().unsqueeze(0) * keys.float()).sum(-1) * scale
    logits = logits.masked_fill(~keep.unsqueeze(1), float("-inf"))
    weights = logits.softmax(0)
    return (weights.unsqueeze(-1) * values.float()).sum(0)


@pytest.mark.parametrize("sink", [0, 4, 16])
@pytest.mark.parametrize("window_left", [31, 63])
def test_window_sink_keeps_exactly_the_leading_tokens(sink, window_left):
    torch.manual_seed(7)
    device = "cuda"
    q = torch.randn(1, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(
        NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    page_table = torch.arange(NUM_PAGES, device=device, dtype=torch.int32).unsqueeze(0)
    cache_seqlens = torch.tensor([CACHE_LEN], device=device, dtype=torch.int32)
    scale = HEAD_DIM**-0.5

    result = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=NUM_PAGES * PAGE_SIZE,
        max_seqlen_q=1,
        window_left=window_left,
        window_sink=sink,
        softmax_scale=scale,
        solution="triton",
    )
    actual = getattr(result, "result", result).reshape(HEADS, HEAD_DIM).float()

    expected = _reference(
        q[0],
        k_cache.reshape(NUM_PAGES * PAGE_SIZE, KV_HEADS, HEAD_DIM),
        v_cache.reshape(NUM_PAGES * PAGE_SIZE, KV_HEADS, HEAD_DIM),
        scale,
        CACHE_LEN,
        window_left,
        sink,
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)


def test_a_sink_request_selects_a_leaf_that_takes_it():
    """Without ``solution=``, the plan must not hand ``window_sink`` to a leaf
    whose signature lacks it.

    The parametrized case above pins the arithmetic with ``solution="triton"``,
    which bypasses selection: it cannot see a plan that picks FA3 and then
    fails with "unexpected keyword argument 'window_sink'".
    """
    torch.manual_seed(7)
    device = "cuda"
    q = torch.randn(1, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(
        NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        NUM_PAGES, PAGE_SIZE, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    page_table = torch.arange(NUM_PAGES, device=device, dtype=torch.int32).unsqueeze(0)
    cache_seqlens = torch.tensor([CACHE_LEN], device=device, dtype=torch.int32)

    result = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=NUM_PAGES * PAGE_SIZE,
        max_seqlen_q=1,
        window_left=31,
        window_sink=16,
        softmax_scale=HEAD_DIM**-0.5,
    )
    actual = getattr(result, "result", result).reshape(HEADS, HEAD_DIM).float()
    expected = _reference(
        q[0],
        k_cache.reshape(NUM_PAGES * PAGE_SIZE, KV_HEADS, HEAD_DIM),
        v_cache.reshape(NUM_PAGES * PAGE_SIZE, KV_HEADS, HEAD_DIM),
        HEAD_DIM**-0.5,
        CACHE_LEN,
        31,
        16,
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
