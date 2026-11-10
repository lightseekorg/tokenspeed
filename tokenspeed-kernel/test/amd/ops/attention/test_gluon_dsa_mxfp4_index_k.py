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

"""GFX950 DSA scoring of mxfp4 queries over mxfp4 index-K pages.

Each page holds 64 rows of 64 nibble-packed E2M1 code bytes, then each row's
four UE8M0 scales as one 4-byte word: DSV4.1's 68-byte index rows, page
planar. The writer and the query quantizer must match DSV4.1's ``cache_pack``
byte for byte; the decode, prefill and owned-page scorers must score as FP32
over the ``cache_unpack`` queries and keys (E2M1 times a power of two is
exact, so only the summation order differs); a query must meet only its own
format's pages; and the facades must reach them only when the caller names
the format, a bare uint8 plane staying fp8_scaled.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsa as dsa_ops
from tokenspeed_kernel.ops.attention.dsv41.triton import cache_pack, cache_unpack
from tokenspeed_kernel.ops.kvcache.triton import index_k_layernorm_fp8_scatter
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.selection import NoKernelFoundError
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD GFX950 is required for the mxfp4 multi-row Gluon scorers",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import sparse_mla  # isort: skip
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (  # isort: skip
    launch_gluon_dsa_index_candidates_gfx950,
)

_PAGE, _DIM, _SCALE = 64, 128, 0.0123


def _mxfp4_plane(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[slots, 128]`` keys as a packed ``[slots, 68]`` page-planar plane,
    and the keys it dequantizes to."""
    packed = cache_pack(keys, "index", None).view(-1, _PAGE, 68)
    plane = torch.cat(
        (
            packed[:, :, :64].reshape(-1, _PAGE * 64),
            packed[:, :, 64:].reshape(-1, _PAGE * 4),
        ),
        dim=1,
    ).view(-1, 68)
    return plane.contiguous(), cache_unpack(packed.view(-1, 68), "index", None).float()


def _mxfp4_query(
    tokens: int, heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mxfp4 query codes and scale words, and the queries they dequantize to."""
    q = torch.randn(tokens, heads, _DIM, device="cuda", dtype=torch.bfloat16) * 3
    codes, scales = dsa_ops.dsa_quantize_index_query_mxfp4(q)
    packed = cache_pack(q.reshape(-1, _DIM), "index", None)
    values = cache_unpack(packed, "index", None).float().view(tokens, heads, _DIM)
    return codes, scales, values


def _fp8_query(tokens: int, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.randn(tokens, heads, _DIM, device="cuda", dtype=torch.bfloat16)
    codes, scales = quantize_fp8(
        q.view(tokens, -1), granularity="token_group", group_size=_DIM
    )
    return codes.view(tokens, heads, _DIM), scales.view(tokens, heads).contiguous()


def _reference(values, weights, keys):
    dots = torch.einsum("thd,nd->thn", values, keys)
    return _SCALE * (torch.relu(dots) * weights[:, :, None]).sum(1)


def _assert_scores(got: torch.Tensor, want: torch.Tensor) -> None:
    assert (got - want).abs().max().item() <= 1e-5 * want.abs().max().item() + 1e-6


@pytest.fixture
def captured_logits(monkeypatch):
    logits = []
    select = sparse_mla._dsa_topk_indices

    def spy(scores, *args, **kwargs):
        logits.append(scores.clone())
        return select(scores, *args, **kwargs)

    monkeypatch.setattr(sparse_mla, "_dsa_topk_indices", spy)
    return logits


@pytest.mark.parametrize("masked", [False, True])
def test_writer_matches_dsv41_codec(masked):
    torch.manual_seed(3)
    pages, tokens = 12, 500
    x = torch.randn(tokens, 160, device="cuda", dtype=torch.bfloat16)[:, :_DIM] * 3
    weight = torch.rand(_DIM, device="cuda") + 0.5
    bias = torch.randn(_DIM, device="cuda") * 0.1
    loc = torch.randperm(pages * _PAGE, device="cuda")[:tokens]
    mask = torch.rand(tokens, device="cuda") > 0.3 if masked else None
    buf = torch.full((pages * _PAGE, 68), 0xA5, device="cuda", dtype=torch.uint8)
    index_k_layernorm_fp8_scatter(
        buf,
        x,
        weight,
        bias,
        loc,
        eps=1e-6,
        page_size=_PAGE,
        group_size=32,
        write_mask=mask,
        index_k_format="mxfp4",
    )
    normed = torch.nn.functional.layer_norm(x.float(), (_DIM,), weight, bias, 1e-6)
    rows = torch.full_like(buf, 0xA5)
    keep = slice(None) if mask is None else mask
    rows[loc[keep]] = cache_pack(normed.bfloat16()[keep].float(), "index", None)
    rows = rows.view(pages, _PAGE, 68)
    want = torch.cat(
        (rows[:, :, :64].reshape(pages, -1), rows[:, :, 64:].reshape(pages, -1)), 1
    )
    assert torch.equal(buf.view(pages, -1), want)


def test_query_quantizer_matches_dsv41_codec():
    torch.manual_seed(5)
    q = torch.randn(37, 64, 160, device="cuda", dtype=torch.bfloat16)[..., :_DIM]
    q = q * torch.logspace(-3, 3, 64, device="cuda").to(torch.bfloat16)[:, None]
    codes, scales = dsa_ops.dsa_quantize_index_query_mxfp4(q)
    packed = cache_pack(q.reshape(-1, _DIM).float(), "index", None).view(37, 64, 68)
    assert codes.shape == (37, 64, 64) and codes.dtype == torch.uint8
    assert scales.shape == (37, 64) and scales.dtype == torch.int32
    assert torch.equal(codes, packed[..., :64])
    assert torch.equal(scales.view(torch.uint8).view(37, 64, 4), packed[..., 64:])


def test_queries_meet_only_their_own_pages():
    plane, _ = _mxfp4_plane(
        torch.randn(4 * _PAGE, _DIM, device="cuda", dtype=torch.bfloat16)
    )
    fp8_plane = torch.zeros(4 * _PAGE, 132, device="cuda", dtype=torch.uint8)
    seq_lens = torch.tensor([200], device="cuda", dtype=torch.int32)
    table = torch.arange(4, device="cuda", dtype=torch.int32)[None]
    weights = torch.rand(1, 32, device="cuda")
    fp8_q, fp8_scales = _fp8_query(1, 32)
    codes, scales, _ = _mxfp4_query(1, 32)
    common = dict(page_size=_PAGE, topk=2048, softmax_scale=_SCALE)
    with pytest.raises(TypeError, match="mxfp4"):
        sparse_mla.launch_gluon_dsa_decode_topk_fp8_rows_gfx950(
            fp8_q,
            weights,
            seq_lens,
            table,
            index_k_cache=plane,
            q_scales=fp8_scales,
            index_k_format="mxfp4",
            **common,
        )
    with pytest.raises(TypeError, match="mxfp4"):
        sparse_mla.launch_gluon_dsa_decode_topk_fp8_rows_gfx950(
            codes,
            weights,
            seq_lens,
            table,
            index_k_cache=fp8_plane,
            q_scales=scales,
            **common,
        )


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("q_len", [1, 4])
def test_decode_scores(captured_logits, heads, q_len):
    torch.manual_seed(0)
    pages = 96
    plane, keys = _mxfp4_plane(
        torch.randn(pages * _PAGE, _DIM, device="cuda", dtype=torch.bfloat16)
    )
    seq_lens = torch.tensor([3000, 777, 4100, 64], device="cuda", dtype=torch.int32)
    table = torch.stack(
        [torch.randperm(pages, device="cuda")[:66] for _ in range(4)]
    ).to(torch.int32)
    q, q_scales, values = _mxfp4_query(4 * q_len, heads)
    weights = torch.rand(4 * q_len, heads, device="cuda")
    sparse_mla.launch_gluon_dsa_decode_topk_fp8_rows_gfx950(
        q,
        weights,
        seq_lens,
        table,
        page_size=_PAGE,
        topk=2048,
        softmax_scale=_SCALE,
        q_len_per_req=q_len,
        index_k_cache=plane,
        q_scales=q_scales,
        index_k_format="mxfp4",
    )
    (logits,) = captured_logits
    offsets = torch.arange(_PAGE, device="cuda")
    for row in range(4 * q_len):
        req = row // q_len
        visible = int(seq_lens[req]) - (q_len - 1) + row % q_len
        slots = (table[req].long()[:, None] * _PAGE + offsets).reshape(-1)[:visible]
        want = _reference(values[row : row + 1], weights[row : row + 1], keys[slots])
        _assert_scores(logits[row, :visible], want[0])


@pytest.mark.parametrize("heads", [32, 64])
def test_prefill_scores(captured_logits, heads):
    torch.manual_seed(1)
    pages, rows, tokens = 80, 3000, 300
    plane, keys = _mxfp4_plane(
        torch.randn(pages * _PAGE, _DIM, device="cuda", dtype=torch.bfloat16)
    )
    workspace = torch.randperm(pages * _PAGE, device="cuda")[:rows]
    q, q_scales, values = _mxfp4_query(tokens, heads)
    weights = torch.rand(tokens, heads, device="cuda")
    ends = torch.randint(1, rows + 1, (tokens,), device="cuda", dtype=torch.int32)
    starts = (ends // 3).to(torch.int32)
    sparse_mla.launch_gluon_dsa_prefill_topk_fp8_rows_gfx950(
        q,
        weights,
        workspace,
        starts,
        ends,
        topk=2048,
        softmax_scale=_SCALE,
        index_k_cache=plane,
        page_size=_PAGE,
        q_scales=q_scales,
        index_k_format="mxfp4",
    )
    (logits,) = captured_logits
    columns = torch.arange(rows, device="cuda")[None]
    live = (columns >= starts[:, None]) & (columns < ends[:, None])
    want = _reference(values, weights, keys[workspace])
    _assert_scores(torch.where(live, logits, 0.0), torch.where(live, want, 0.0))


@pytest.mark.parametrize("rows,requests", [(8, None), (512, 2)])
def test_owned_page_scores(rows, requests):
    torch.manual_seed(2)
    pages, owned_pages = 64, 20
    plane, keys = _mxfp4_plane(
        torch.randn(pages * _PAGE, _DIM, device="cuda", dtype=torch.bfloat16)
    )
    q, q_scales, values = _mxfp4_query(rows, 32)
    weights = torch.rand(rows, 32, device="cuda")
    owned = torch.randperm(pages, device="cuda")[:owned_pages].to(torch.int32)
    table = owned[None].expand(rows, -1).contiguous()
    lengths = torch.sort(torch.randint(1, owned_pages * _PAGE, (rows,), device="cuda"))[
        0
    ].to(torch.int32)
    query_requests = None
    if requests:
        query_requests = (torch.arange(rows, device="cuda") * requests // rows).to(
            torch.int32
        )
    logits = launch_gluon_dsa_index_candidates_gfx950(
        q,
        q_scales,
        weights,
        plane,
        table,
        lengths,
        page_size=_PAGE,
        softmax_scale=_SCALE,
        query_group_size=1,
        query_requests=query_requests,
        request_count=requests,
        index_k_format="mxfp4",
    )
    slots = (
        owned.long()[:, None] * _PAGE + torch.arange(_PAGE, device="cuda")
    ).reshape(-1)
    want = _reference(values, weights, keys[slots])
    live = torch.arange(want.shape[1], device="cuda")[None] < lengths[:, None]
    _assert_scores(torch.where(live, logits, 0.0), torch.where(live, want, 0.0))


def test_facades_select_mxfp4_only_when_named():
    torch.manual_seed(4)
    pages, reqs = 80, 2
    plane, keys = _mxfp4_plane(
        torch.randn(pages * _PAGE, _DIM, device="cuda", dtype=torch.bfloat16)
    )
    q, q_scales, values = _mxfp4_query(reqs, 32)
    weights = torch.rand(reqs, 32, device="cuda")
    seq_lens = torch.tensor([5000, 3100], device="cuda", dtype=torch.int32)
    table = torch.stack(
        [torch.randperm(pages, device="cuda")[:79] for _ in range(reqs)]
    ).to(torch.int32)
    common = dict(
        page_size=_PAGE,
        topk=2048,
        softmax_scale=_SCALE,
        batch_invariant=False,
        index_k_cache=plane,
        q_scales=q_scales,
        slot_order="selection",
    )
    indices, lens = dsa_ops.dsa_decode_topk(
        q, weights, seq_lens, table, index_k_format="mxfp4", **common
    )
    offsets = torch.arange(_PAGE, device="cuda")
    for req in range(reqs):
        slots = (table[req].long()[:, None] * _PAGE + offsets).reshape(-1)
        slots = slots[: int(seq_lens[req])]
        scores = _reference(values[req : req + 1], weights[req : req + 1], keys[slots])[
            0
        ]
        kth = scores.topk(2048).values[-1]
        assert int(lens[req]) == 2048
        chosen = indices[req].long()
        rank_of = torch.full((pages * _PAGE,), -1, device="cuda", dtype=torch.long)
        rank_of[slots] = torch.arange(slots.numel(), device="cuda")
        picked = scores[rank_of[chosen]]
        # Near ties at the k-th score may swap; nothing far below it is picked.
        assert picked.min() >= kth - 1e-5 * scores.abs().max()
    # The same bytes read as fp8_scaled have a different row width.
    with pytest.raises(NoKernelFoundError):
        dsa_ops.dsa_decode_topk(q, weights, seq_lens, table, **common)
