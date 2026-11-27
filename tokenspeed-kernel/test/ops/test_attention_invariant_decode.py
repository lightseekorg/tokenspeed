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

"""The batch-invariant MHA decode leaf.

Checked against ``torch_dense_attention`` over the keys a row sees, and for
execution invariance: a request's bytes stay fixed while only the requests
beside it change. Geometry follows a GQA block draft (32 query heads over 8 KV
heads, head_dim 128, sliding window plus a retained sink, block width 8).
"""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache
from tokenspeed_kernel.ops.attention.mha._triton.invariant_decode import (
    _invariant_decode_kernel,
)
from tokenspeed_kernel.profiling import ShapeCapture
from tokenspeed_kernel.selection import NoKernelFoundError
from utils import assert_no_triton_compile

H_Q, H_KV, HEAD_DIM, PAGE = 32, 8, 128, 128
SEQ, WINDOW_LEFT, WINDOW_SINK = 2048, 1024, 16
BLOCK = 8
MAX_BS = 16
PAGES_PER = SEQ // PAGE
# One BF16 rounding of the output, and P rounded to BF16 before P @ V.
TOLERANCE = dict(atol=2**-8, rtol=2**-7)
LEAF = "triton_invariant_mha_decode_with_kvcache"


@pytest.fixture(scope="module")
def cache():
    """One cache and one query set, sliced by each batch size.

    Built once at the largest size: CUDA ``randn`` values depend on the
    tensor's element count, so generating per batch size would change
    request 0's bytes along with its co-batch.
    """
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.bfloat16)

    pages = MAX_BS * PAGES_PER + 1
    table = torch.arange(1, pages, device="cuda", dtype=torch.int32)
    return (
        randn(pages, PAGE, H_KV, HEAD_DIM),
        randn(pages, PAGE, H_KV, HEAD_DIM),
        table.view(MAX_BS, PAGES_PER),
        randn(MAX_BS * BLOCK, H_Q, HEAD_DIM),
    )


def decode(cache, bs, *, width=1, block_draft=False, lens=None, **options):
    """``width`` query rows per request. A block draft expands each request
    to ``BLOCK`` one-row requests sharing its page table and length, as the
    attention backend does; otherwise ``width`` rows are one causal request."""
    k, v, table, q_all = cache
    rows_per_request = BLOCK if block_draft else width
    q = q_all[: bs * rows_per_request].contiguous()
    lens = (
        torch.full((bs,), SEQ, device="cuda", dtype=torch.int32)
        if lens is None
        else lens
    )
    page_table = table[:bs]
    if block_draft:
        page_table = page_table.repeat_interleave(BLOCK, dim=0)
        lens = lens.repeat_interleave(BLOCK)
    kwargs = dict(
        window_left=WINDOW_LEFT,
        window_sink=WINDOW_SINK,
        softmax_scale=HEAD_DIM**-0.5,
        batch_invariant=True,
    )
    kwargs.update(options)
    return mha_decode_with_kvcache(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=page_table,
        cache_seqlens=lens,
        max_seqlen_k=SEQ,
        max_seqlen_q=1 if block_draft else width,
        solution="triton",
        block_draft=block_draft,
        uniform_row_groups=BLOCK if block_draft else 1,
        **kwargs,
    )


def reference(cache, row, end, *, window_left=WINDOW_LEFT, sinks=None):
    """Attention of query row ``row`` of request 0 over the keys it sees: the
    ``window_left + 1`` before ``end``, and the leading ``WINDOW_SINK`` ones
    when a window applies. Gathers them and runs dense attention, independent
    of the kernel's tiling."""
    k, v, table, q_all = cache
    pages = table[0]
    keys = k[pages].reshape(-1, H_KV, HEAD_DIM)
    values = v[pages].reshape(-1, H_KV, HEAD_DIM)
    position = torch.arange(SEQ, device="cuda")
    visible = position < end
    if window_left >= 0:
        visible &= (position >= end - (window_left + 1)) | (position < WINDOW_SINK)
    return torch_dense_attention(
        q_all[row : row + 1],
        keys[visible],
        values[visible],
        scale=HEAD_DIM**-0.5,
        causal_offset=None,
        window_left=None,
        sinks=sinks,
    )


def test_a_bitwise_call_selects_the_invariant_leaf_and_only_it(cache, tmp_path):
    shapes = tmp_path / "shapes.json"
    ShapeCapture.reset()
    ShapeCapture.get().enabled = True
    try:
        decode(cache, 2, block_draft=True)
        decode(cache, 2, block_draft=True, batch_invariant=False)
        ShapeCapture.get().dump(shapes)
    finally:
        ShapeCapture.reset()
    names = [record["kernel_name"] for record in json.loads(shapes.read_text())]
    assert names[0] == LEAF
    assert names[1] != LEAF


@pytest.mark.parametrize("bs", [2, 3, 8, 16])
@pytest.mark.parametrize(
    "shape",
    [dict(block_draft=True), dict(width=1), dict(width=4)],
    ids=["block-draft", "decode", "verify-width-4"],
)
def test_rows_do_not_move_with_the_batch(cache, bs, shape):
    """Request 0's rows come back bit-identical whatever rides with them."""
    rows = BLOCK if shape.get("block_draft") else shape["width"]
    alone = decode(cache, 1, **shape)
    batched = decode(cache, bs, **shape)
    assert torch.equal(batched[:rows], alone)


@pytest.mark.parametrize(
    "shape",
    [dict(block_draft=True), dict(width=1), dict(width=4)],
    ids=["block-draft", "decode", "verify-width-4"],
)
@pytest.mark.parametrize(
    "window", [dict(), dict(window_left=-1, window_sink=0)], ids=["window", "full"]
)
def test_matches_dense_attention(cache, shape, window):
    out = decode(cache, 1, **shape, **window)
    width = BLOCK if shape.get("block_draft") else shape["width"]
    for row in range(width):
        # A block row sees the whole request; row t of a causal verify sees
        # the keys before the last (width - 1 - t).
        end = SEQ if shape.get("block_draft") else SEQ - (width - 1 - row)
        expected, _ = reference(
            cache, row, end, window_left=window.get("window_left", WINDOW_LEFT)
        )
        torch.testing.assert_close(out[row].float(), expected[0], **TOLERANCE)


def test_matches_dense_attention_with_sinks_and_lse(cache):
    g = torch.Generator(device="cuda").manual_seed(5)
    sinks = torch.randn(H_Q, generator=g, device="cuda", dtype=torch.float32)
    out = decode(cache, 1, width=1, sinks=sinks)
    expected, _ = reference(cache, 0, SEQ, sinks=sinks)
    torch.testing.assert_close(out[0].float(), expected[0], **TOLERANCE)

    out, lse = decode(cache, 1, width=1, return_lse=True)
    expected, expected_lse = reference(cache, 0, SEQ)
    torch.testing.assert_close(out[0].float(), expected[0], **TOLERANCE)
    torch.testing.assert_close(lse[0], expected_lse[0], atol=2**-7, rtol=2**-7)


def test_a_logit_cap_has_no_invariant_leaf(cache):
    with pytest.raises(NoKernelFoundError):
        decode(cache, 1, width=1, logit_cap=30.0)


def test_run_invariance(cache):
    assert torch.equal(
        decode(cache, 4, block_draft=True), decode(cache, 4, block_draft=True)
    )


def test_ragged_batches_do_not_recompile(cache):
    """Batch size, lengths and page-table width are runtime arguments."""

    def run(bs, base):
        lens = base + 7 * torch.arange(bs, device="cuda", dtype=torch.int32)
        return decode(cache, bs, width=1, lens=lens)

    run(2, 300)
    run(3, 1500)
    with assert_no_triton_compile(_invariant_decode_kernel):
        for bs, base in ((1, 100), (5, 777), (16, 1025), (7, 2000)):
            run(bs, base)


def test_pages_past_2_pow_31_elements():
    """K and V offsets of the far end of a cache whose page index times the
    page stride passes 2**31 elements; only its last two pages are touched."""
    heads, dim, page = 1, 64, 16
    pages = 2**31 // (page * heads * dim) + 8
    cache_bytes = 2 * pages * page * heads * dim * 2
    if torch.cuda.mem_get_info()[0] < cache_bytes + 2**30:
        pytest.skip(f"needs {cache_bytes >> 30} GiB of free device memory")
    g = torch.Generator(device="cuda").manual_seed(1)
    k = torch.empty(pages, page, heads, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.empty_like(k)
    for tail in (k[-2:], v[-2:]):
        tail.copy_(torch.randn(tail.shape, generator=g, device="cuda").bfloat16())
    q = torch.randn(1, 4, dim, generator=g, device="cuda").bfloat16()
    table = torch.tensor([[pages - 2, pages - 1]], dtype=torch.int32, device="cuda")
    out = mha_decode_with_kvcache(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=table,
        cache_seqlens=torch.tensor([2 * page], dtype=torch.int32, device="cuda"),
        max_seqlen_k=2 * page,
        max_seqlen_q=1,
        solution="triton",
        block_draft=False,
        batch_invariant=True,
        softmax_scale=dim**-0.5,
    )
    expected, _ = torch_dense_attention(
        q,
        k[-2:].reshape(-1, heads, dim),
        v[-2:].reshape(-1, heads, dim),
        scale=dim**-0.5,
        causal_offset=None,
        window_left=None,
        sinks=None,
    )
    torch.testing.assert_close(out.float(), expected, **TOLERANCE)
