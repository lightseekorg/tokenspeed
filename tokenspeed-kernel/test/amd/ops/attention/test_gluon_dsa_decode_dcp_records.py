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

"""GFX950 KVP decode records: the dense matrix core on a rank-major query
gather, its split reduction writing the DCP partials exchange's records."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD GFX950 is required for the Gluon DSA decode records",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.attention.dsa import (  # isort: skip
    dsa_decode,
    dsa_decode_dcp_records,
    dsa_decode_dcp_records_supported,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (  # isort: skip
    dispatch_dcp_attention_partials,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import (  # isort: skip
    attention as gfx950_attention,
)

# K3.1 under KVP: 576-wide FP8 latent rows on 64-token pages, a 2048-slot
# selection of which a rank owns about 256 live entries per row.
PAGES = 64
TOPK = 2048
SCALE = 192**-0.5


def _inputs(tokens, degree, local_heads, *, seed):
    """Every rank's FP8 query heads ``[degree, tokens, local_heads, 576]``, a
    paged FP8 cache and one rank's compacted selection: an empty row, an
    all-invalid live row, a full-width row, a NaN query head, ``-1`` holes
    inside live prefixes and stale slots past every length."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cache = torch.randn(
        (PAGES, 64, 1, 576), device="cuda", dtype=torch.bfloat16, generator=generator
    ).to(torch.float8_e4m3fn)
    q = torch.randn(
        (degree, tokens, local_heads, 576),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    slots = torch.randint(
        0,
        PAGES * 64,
        (tokens, TOPK),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    lengths = torch.randint(
        0, 320, (tokens,), device="cuda", dtype=torch.int32, generator=generator
    )
    special = torch.tensor([0, 5, TOPK, 64], device="cuda", dtype=torch.int32)
    lengths[: min(tokens, 4)] = special[: min(tokens, 4)]
    if tokens > 1:
        slots[1, :5] = -1
    if tokens > 3:
        # FP8 E4M3 NaN: every key of that head scores NaN.
        q.view(torch.uint8)[degree - 1, 3, local_heads - 1, 7] = 0x7F
    slots[4::7, ::5] = -1
    return q, cache, slots, lengths


def _token_major(q):
    """The rank-major gather as ``gather_query_heads`` lays it out: head
    ``r * local_heads + h`` of every token. Copied as bytes."""
    degree, tokens, local_heads, dim = q.shape
    return (
        q.view(torch.uint8)
        .permute(1, 0, 2, 3)
        .reshape(tokens, degree * local_heads, dim)
        .view(q.dtype)
    )


def _kwargs(cache, slots, lengths):
    return dict(
        kv_cache=cache,
        topk_slots=slots,
        topk_lens=lengths,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=SCALE,
        page_size=64,
        q_len_per_req=4,
        solution="gluon",
        slot_order="selection",
    )


def _supported(tokens, heads):
    return dsa_decode_dcp_records_supported(
        tokens=tokens,
        heads=heads,
        topk=TOPK,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )


@pytest.mark.parametrize(
    "tokens,degree,local_heads",
    [
        # K3.1 at TP8 / KVP8: the drafter's 16 rows, C12 and C16 verify rows.
        (16, 8, 12),
        (48, 8, 12),
        (64, 8, 12),
        (4, 8, 12),
        (64, 2, 8),
        (20, 4, 16),
    ],
)
def test_decode_records_match_the_dispatched_decode(tokens, degree, local_heads):
    q, cache, slots, lengths = _inputs(
        tokens, degree, local_heads, seed=tokens + 31 * degree + local_heads
    )
    assert _supported(tokens, degree * local_heads)
    kwargs = _kwargs(cache, slots, lengths)
    # Today's head-sharded decode: the token-major gather, the core with its
    # LSE, then the repack into the records the all-to-all sends.
    out, lse = dsa_decode(
        q=_token_major(q),
        sparse_kv_cache=None,
        max_seqlen_k=PAGES * 64,
        return_lse=True,
        **kwargs,
    )
    expected = dispatch_dcp_attention_partials(out, lse, degree=degree)
    records = dsa_decode_dcp_records(q=q, **kwargs)
    assert records.dtype == torch.int32 and records.is_contiguous()
    assert records.shape == (degree, tokens, local_heads, 257)
    assert torch.equal(records, expected)
    # The empty and all-invalid rows are zero output with a -inf LSE, as the
    # combine expects of an empty partial.
    empty = records[:, : min(tokens, 2)]
    assert not empty[..., :256].any()
    neg_inf_bits = torch.tensor(-torch.inf).view(torch.int32).item()
    assert (empty[..., 256] == neg_inf_bits).all()


def test_decode_records_refuse_unsplit_launches():
    # Past 512 programs (two waves of 256 CUs) the core runs one split and
    # writes its output itself, not through the reduction.
    assert _supported(85, 96)
    assert not _supported(86, 96)
    assert not _supported(0, 96)
    assert not dsa_decode_dcp_records_supported(
        tokens=16,
        heads=96,
        topk=TOPK,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    q, cache, slots, lengths = _inputs(86, 8, 12, seed=5)
    with pytest.raises(ValueError, match="split keys"):
        dsa_decode_dcp_records(q=q, **_kwargs(cache, slots, lengths))


def test_decode_records_replay_in_a_graph_across_lengths():
    degree, local_heads, tokens = 8, 12, 64
    q, cache, slots, lengths = _inputs(tokens, degree, local_heads, seed=11)
    kwargs = _kwargs(cache, slots, lengths)
    dsa_decode_dcp_records(q=q, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dsa_decode_dcp_records(q=q, **kwargs)
    for visible in (0, 13, 257, TOPK):
        lengths.fill_(visible)
        slots.copy_(torch.roll(slots, 1, dims=1))
        graph.replay()
        out, lse = dsa_decode(
            q=_token_major(q),
            sparse_kv_cache=None,
            max_seqlen_k=PAGES * 64,
            return_lse=True,
            **kwargs,
        )
        expected = dispatch_dcp_attention_partials(out, lse, degree=degree)
        torch.cuda.synchronize()
        assert torch.equal(captured, expected)


def test_decode_records_compile_once_across_tokens():
    degree, local_heads, most = 8, 12, 64
    q, cache, slots, lengths = _inputs(most, degree, local_heads, seed=7)

    def run(tokens):
        rows = q.view(torch.uint8)[:, :tokens].contiguous().view(q.dtype)
        kwargs = _kwargs(cache, slots[:tokens], lengths[:tokens])
        dsa_decode_dcp_records(q=rows, **kwargs)
        dsa_decode(
            q=_token_major(rows),
            sparse_kv_cache=None,
            max_seqlen_k=PAGES * 64,
            return_lse=True,
            **kwargs,
        )

    # The rank stride follows the token count as a runtime argument; the
    # layout (the heads per rank, a model constant) is compiled, so each
    # layout has one binary for every decode size.
    run(64)
    with assert_no_triton_compile(
        gfx950_attention._dsa_dense_mfma_kv_kernel,
        gfx950_attention._dsa_dense_mfma_reduce_kernel,
    ):
        for tokens in (4, 16, 20, 48):
            run(tokens)
    torch.cuda.synchronize()
