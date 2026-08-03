# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""store_sf_interleaved: MXFP8 scale scatter into the FA4 atom layout.

Bit-exactness is pinned two ways:
  1. Against a pure-torch scatter that applies the documented mapping
     (token at page offset t -> packed-u32 position (t%32)*4 + t//32).
  2. Against the FA4 fork's own ``interleave_sf`` (the layout's source of
     truth) for a full contiguous page, when the fork is installed.
"""

from __future__ import annotations

import pytest
import torch

triton_mod = pytest.importorskip("tokenspeed_kernel.ops.kvcache.triton")
store_sf_interleaved = triton_mod.store_sf_interleaved

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

NHEADS = 4
SF_DIM = 4  # head_dim 128 / sf_vec_size 32
PAGE = 128


def _ref_scatter(sf_in: torch.Tensor, sf_out: torch.Tensor, loc: torch.Tensor):
    """Documented mapping, applied one token at a time in torch."""
    out_u32 = sf_out.view(torch.uint8).reshape(-1, 4).view(torch.int32).reshape(-1)
    in_u32 = (
        sf_in.view(torch.uint8)
        .reshape(sf_in.shape[0], NHEADS, 4)
        .contiguous()
        .view(torch.int32)
        .reshape(sf_in.shape[0], NHEADS)
    )
    page_stride = NHEADS * 128
    for t in range(sf_in.shape[0]):
        slot = int(loc[t])
        page, off = divmod(slot, PAGE)
        pos = (off % 32) * 4 + (off // 32)
        for h in range(NHEADS):
            out_u32[page * page_stride + h * 128 + pos] = in_u32[t, h]


@needs_cuda
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 300])
def test_matches_reference_scatter(num_tokens: int):
    torch.manual_seed(3)
    num_pages = 8
    sf_in = torch.randint(
        1, 255, (num_tokens, NHEADS, SF_DIM), dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)
    loc = torch.randperm(num_pages * PAGE, device="cuda")[:num_tokens].to(torch.int64)

    out = torch.zeros(
        num_pages, NHEADS, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda"
    )
    ref = torch.zeros_like(out)

    store_sf_interleaved(sf_in, out, loc)
    _ref_scatter(sf_in, ref, loc)

    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


@needs_cuda
def test_matches_fork_interleave_sf():
    """A full contiguous page written token-by-token must equal the fork's
    bulk interleave of the same [1, 128, H, 4] slab."""
    try:
        from flash_attn.cute.blockscaled_utils import interleave_sf
    except ImportError:
        pytest.skip("fa4 fork with blockscaled_utils not installed")

    torch.manual_seed(5)
    sf_tok = torch.randint(
        1, 255, (PAGE, NHEADS, SF_DIM), dtype=torch.uint8, device="cuda"
    )
    out = torch.zeros(1, NHEADS, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda")
    loc = torch.arange(PAGE, dtype=torch.int64, device="cuda")
    store_sf_interleaved(sf_tok.view(torch.float8_e8m0fnu), out, loc)

    # fork: (batch=1, seqlen=128, nheads, sf_k) -> (1, nheads, REST_M=1, REST_K=1, 32, 4, 4)
    ref = interleave_sf(sf_tok.unsqueeze(0).view(torch.float8_e8m0fnu), 32)
    ref = ref.view(torch.uint8).reshape(1, NHEADS, 32, 4, 4)

    assert torch.equal(out.view(torch.uint8), ref)


@needs_cuda
@pytest.mark.parametrize("num_tokens", [1, 300, 1024])
def test_k_chunk_pages_match_chunkwise_scatter(num_tokens: int):
    """page_size = 2*128 (hetero full group): the 6D layout must equal the
    page-128 scatter of the same tokens re-based onto per-128-chunk pages,
    transposed into (pages, heads, chunk, atom) order."""
    torch.manual_seed(11)
    num_pages, k = 4, 2
    page_tokens = k * PAGE
    sf_in = torch.randint(
        1, 255, (num_tokens, NHEADS, SF_DIM), dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)
    loc = torch.randperm(num_pages * page_tokens, device="cuda")[:num_tokens].to(
        torch.int64
    )

    out = torch.zeros(
        num_pages, NHEADS, k, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda"
    )
    store_sf_interleaved(sf_in, out, loc, page_size=page_tokens)

    chunks = torch.zeros(
        num_pages * k, NHEADS, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda"
    )
    store_sf_interleaved(sf_in, chunks, loc)  # same slots at 128-page grain
    ref = (
        chunks.view(torch.uint8)
        .reshape(num_pages, k, NHEADS, 32, 4, 4)
        .transpose(1, 2)
        .contiguous()
    )
    assert torch.equal(out.view(torch.uint8), ref)


@needs_cuda
def test_zero_tokens_noop():
    out = torch.zeros(2, NHEADS, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda")
    sf_in = torch.zeros(0, NHEADS, SF_DIM, dtype=torch.float8_e8m0fnu, device="cuda")
    loc = torch.zeros(0, dtype=torch.int64, device="cuda")
    store_sf_interleaved(sf_in, out, loc)  # must not raise


@needs_cuda
def test_rejects_wrong_page_size():
    sf_in = torch.zeros(4, NHEADS, SF_DIM, dtype=torch.float8_e8m0fnu, device="cuda")
    out = torch.zeros(1, NHEADS, 32, 4, 4, dtype=torch.float8_e8m0fnu, device="cuda")
    loc = torch.arange(4, dtype=torch.int64, device="cuda")
    with pytest.raises(AssertionError):
        store_sf_interleaved(sf_in, out, loc, page_size=64)
    with pytest.raises(AssertionError):
        store_sf_interleaved(sf_in, out, loc, page_size=192)
