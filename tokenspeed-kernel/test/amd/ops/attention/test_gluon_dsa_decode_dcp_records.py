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
gather, it or its split reduction writing the DCP partials exchange's
records."""

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
    dsa_decode_dcp_records_workspace_bytes,
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
    degree, tokens, local_heads, _ = q.shape
    return (
        q.view(torch.uint8)
        .permute(1, 0, 2, 3)
        .reshape(tokens, degree * local_heads, -1)
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


def _dispatched(q, kwargs):
    """Today's head-sharded decode: the token-major gather, the core with its
    LSE, then the repack into the records the all-to-all sends."""
    out, lse = dsa_decode(
        q=_token_major(q),
        sparse_kv_cache=None,
        max_seqlen_k=PAGES * 64,
        return_lse=True,
        **kwargs,
    )
    return dispatch_dcp_attention_partials(out, lse, degree=q.shape[0])


def _assert_records_attend(records, q, cache, slots, lengths):
    """Each head's record against FP32 attention over its row's live slots:
    the output within what the core's FP8 probabilities cost the
    token-major decode too, the natural-log LSE of the scaled scores within
    FP32 rounding. The NaN query head is left out."""
    degree, tokens, local_heads, _ = q.shape
    keys = cache.view(-1, 576).float()
    out = torch.zeros((degree, tokens, local_heads, 512), device="cuda")
    lse = torch.full((degree, tokens, local_heads), -torch.inf, device="cuda")
    for row in range(tokens):
        live = slots[row, : int(lengths[row])]
        row_keys = keys[live[live >= 0].long()]
        if row_keys.shape[0]:
            scores = torch.einsum("rhd,kd->rhk", q[:, row].float(), row_keys) * SCALE
            lse[:, row] = torch.logsumexp(scores, dim=-1)
            out[:, row] = torch.softmax(scores, dim=-1) @ row_keys[:, :512]
    kept = ~lse.isnan()
    error = (records[..., :256].contiguous().view(torch.bfloat16).float() - out)[kept]
    assert error.abs().max() < 0.15
    assert error.square().mean().sqrt() <= 0.02 * out[kept].square().mean().sqrt()
    torch.testing.assert_close(
        records[..., 256].contiguous().view(torch.float32)[kept],
        lse[kept],
        atol=2e-4,
        rtol=1e-5,
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
    expected = _dispatched(q, kwargs)
    records = dsa_decode_dcp_records(q=q, **kwargs)
    assert records.dtype == torch.int32 and records.is_contiguous()
    assert records.shape == (degree, tokens, local_heads, 257)
    # Rows near a rank's share of the selection split less than the
    # token-major decode; the full-width row splits as it does: the same LSE,
    # and outputs within one BF16 rounding (that decode's merge unrolls its
    # splits).
    _assert_records_attend(records, q, cache, slots, lengths)
    _assert_records_attend(expected, q, cache, slots, lengths)
    if tokens > 2:
        assert torch.equal(records[:, 2, :, 256], expected[:, 2, :, 256])
        bits = [
            r[:, 2, :, :256].contiguous().view(torch.int16).int()
            for r in (records, expected)
        ]
        assert (bits[0] - bits[1]).abs().max() <= 1
    # The empty and all-invalid rows are zero output with a -inf LSE, as the
    # combine expects of an empty partial.
    empty = records[:, : min(tokens, 2)]
    assert not empty[..., :256].any()
    neg_inf_bits = torch.tensor(-torch.inf).view(torch.int32).item()
    assert (empty[..., 256] == neg_inf_bits).all()


@pytest.mark.parametrize("tokens", [4, 16, 48, 64])
def test_decode_records_of_a_row_follow_only_its_own_length(tokens):
    # A row's split count comes from its own live length, so its record
    # does not move when the other rows' lengths do.
    degree, local_heads = 8, 12
    q, cache, slots, lengths = _inputs(tokens, degree, local_heads, seed=tokens + 5)
    kwargs = _kwargs(cache, slots, lengths)
    before = dsa_decode_dcp_records(q=q, **kwargs)
    row = tokens - 1
    for other in (0, 256, 1000, TOPK):
        lengths[:row] = other
        after = dsa_decode_dcp_records(q=q, **kwargs)
        assert torch.equal(after[:, row], before[:, row])
        _assert_records_attend(after, q, cache, slots, lengths)


@pytest.mark.parametrize("tokens", [4, 48, 64])
def test_decode_records_read_a_query_riding_in_a_gathered_chunk(tokens):
    # One all-gather of every rank's index scores and query lands rank r's
    # query after its scores: each rank's rows contiguous, the ranks one
    # whole chunk apart.
    degree, local_heads = 8, 12
    q, cache, slots, lengths = _inputs(tokens, degree, local_heads, seed=tokens + 3)
    kwargs = _kwargs(cache, slots, lengths)
    expected = dsa_decode_dcp_records(q=q, **kwargs)
    scores = tokens * TOPK * 4
    chunks = torch.full(
        (degree, scores + tokens * local_heads * 576),
        0x7F,
        device="cuda",
        dtype=torch.uint8,
    )
    rider = chunks[:, scores:].view(q.dtype).view(q.shape)
    rider.view(torch.uint8).copy_(q.view(torch.uint8))
    assert not rider.is_contiguous()
    with assert_no_triton_compile(
        gfx950_attention._dsa_dense_mfma_kv_kernel,
        gfx950_attention._dsa_dense_mfma_records_reduce_kernel,
    ):
        assert torch.equal(dsa_decode_dcp_records(q=rider, **kwargs), expected)
    # A rank stride the query's copy to LDS cannot take is copied first.
    odd = torch.zeros(
        (degree, tokens * local_heads * 576 + 8), device="cuda", dtype=torch.uint8
    )
    shifted = odd[:, : tokens * local_heads * 576].view(q.dtype).view(q.shape)
    shifted.view(torch.uint8).copy_(q.view(torch.uint8))
    assert torch.equal(dsa_decode_dcp_records(q=shifted, **kwargs), expected)


@pytest.mark.parametrize("tokens", [1, 4, 48, 64])
def test_decode_records_lead_a_given_workspace(tokens):
    # An exchange that reads the records in place lends their storage: they
    # lead it, bit for bit what the core writes into its own.
    degree, local_heads = 8, 12
    q, cache, slots, lengths = _inputs(tokens, degree, local_heads, seed=tokens + 9)
    kwargs = _kwargs(cache, slots, lengths)
    expected = dsa_decode_dcp_records(q=q, **kwargs)
    words = (
        dsa_decode_dcp_records_workspace_bytes(
            tokens=tokens,
            heads=degree * local_heads,
            topk=TOPK,
            q_dtype=q.dtype,
            kv_dtype=cache.dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        // 4
    )
    assert words >= expected.numel()
    # Stale words in the lent storage, and room past the splits.
    workspace = torch.full((words + 64,), -7, device="cuda", dtype=torch.int32)
    records = dsa_decode_dcp_records(q=q, workspace=workspace, **kwargs)
    assert records.data_ptr() == workspace.data_ptr()
    assert torch.equal(records, expected)
    for bad in (workspace[: words - 1], workspace.view(torch.uint8), workspace[::2]):
        with pytest.raises(ValueError, match="workspace"):
            dsa_decode_dcp_records(q=q, workspace=bad, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_decode_records_serve_unsplit_launches(dtype):
    # Past 512 programs (two waves of 256 CUs), and for a BF16 query and
    # cache at any size, the core runs one split and writes the records
    # itself, as the token-major decode writes its output.
    tokens = 86 if dtype == torch.float8_e4m3fn else 16
    q, cache, slots, lengths = _inputs(tokens, 8, 12, seed=5)
    q, cache = q.to(dtype), cache.to(dtype)
    assert dsa_decode_dcp_records_supported(
        tokens=tokens,
        heads=96,
        topk=TOPK,
        q_dtype=dtype,
        kv_dtype=dtype,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    kwargs = _kwargs(cache, slots, lengths)
    assert torch.equal(dsa_decode_dcp_records(q=q, **kwargs), _dispatched(q, kwargs))
    assert not _supported(0, 96)


def test_decode_records_replay_in_a_graph_across_lengths():
    degree, local_heads, tokens = 8, 12, 64
    q, cache, slots, lengths = _inputs(tokens, degree, local_heads, seed=11)
    kwargs = _kwargs(cache, slots, lengths)
    dsa_decode_dcp_records(q=q, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dsa_decode_dcp_records(q=q, **kwargs)
    for visible in (0, 13, 256, 257, 700, TOPK):
        lengths.fill_(visible)
        slots.copy_(torch.roll(slots, 1, dims=1))
        graph.replay()
        expected = dsa_decode_dcp_records(q=q, **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(captured, expected)
        _assert_records_attend(captured, q, cache, slots, lengths)


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
        gfx950_attention._dsa_dense_mfma_records_reduce_kernel,
    ):
        for tokens in (4, 16, 20, 48):
            run(tokens)
    torch.cuda.synchronize()
