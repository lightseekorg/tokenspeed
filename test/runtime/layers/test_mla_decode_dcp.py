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

"""Masking options of the MLA decode kernel: DCP and the sliding window.

Under DCP each rank holds a strided ``1/cp_world`` slice of the KV context
(local key ``c`` is global position ``c*cp_world + rank``). The kernel gained
``return_lse`` + ``causal_seqs`` + ``cp_world`` so a caller can run the kernel on
each rank's slice and merge the partials into the full-context result.

This test asserts that property end to end: splitting the context into strided
slices, running the kernel per slice (``return_lse=True``), and merging the
partials via their LSE weights reproduces the ``cp_world=1`` full-context output.
A wrong strided mask makes a slice's partial wrong; a wrong LSE makes the merge
weights wrong -- either breaks the reconstruction. Keys are skewed by parity so
the merge weight is far from 0.5, which makes the (log2) LSE scaling load-bearing.

``window_left`` is the other bound the kernel takes: it moves the low edge of
each block row's visible range, and the kernel both masks it and starts its
KV walk there. Wrong by a tile and a draft silently attends to keys its
training mask never showed it, so the test asserts against an explicit
per-row slice rather than against a second kernel configuration.
"""

from __future__ import annotations

import pytest
import torch

_HAS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
pytestmark = pytest.mark.skipif(
    not _HAS_SM100, reason="TokenSpeed MLA decode kernel requires Blackwell SM100"
)

KV_LORA, QK_ROPE = 512, 64
D = KV_LORA + QK_ROPE
H, Q = 128, 4
PAGE = 64


def _workspace():
    from tokenspeed_mla import get_num_sm

    n = get_num_sm(torch.device("cuda")) * H * 8 * (KV_LORA + 1) * 4
    return torch.empty(n, dtype=torch.int8, device="cuda")


def _paged(key_rows, dtype):
    """Pack [n, D] key rows into a [pages, PAGE, D] cache + [1, pages] block table."""
    n = key_rows.shape[0]
    pages = (n + PAGE - 1) // PAGE
    cache = torch.zeros(pages, PAGE, D, device="cuda", dtype=dtype)
    cache.view(-1, D)[:n] = key_rows
    bt = torch.arange(pages, device="cuda", dtype=torch.int32).view(1, pages)
    return cache, bt


def _decode(
    query,
    cache,
    bt,
    n,
    ws,
    dtype,
    *,
    causal_seqs=None,
    causal_mask=True,
    window_left=-1,
    cp_world=1,
    cp_rank=0,
    lse=False,
):
    from tokenspeed_mla import tokenspeed_mla_decode

    return tokenspeed_mla_decode(
        query=query,
        kv_cache=cache,
        workspace_buffer=ws,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=QK_ROPE,
        block_tables=bt,
        seq_lens=torch.tensor([n], device="cuda", dtype=torch.int32),
        max_seq_len=n,
        softmax_scale=1.0 / (D**0.5),
        output_scale=1.0,
        return_lse=lse,
        causal_mask=causal_mask,
        window_left=window_left,
        causal_seqs=causal_seqs,
        cp_world=cp_world,
        cp_rank=cp_rank,
    )


def test_dcp_strided_slices_merge_to_full_context():
    # DCP (strided global-coordinate masking) is implemented on the fp8 kernel.
    dtype, tol = torch.float8_e4m3fn, 6e-2
    torch.manual_seed(0)
    L, W = 128, 2  # global context length, cp_world
    ws = _workspace()

    # Skew keys by parity (even keys aligned with the query direction, odd keys
    # anti-aligned) so the two strided slices carry very different attention mass
    # and the LSE-derived merge weight is far from 0.5.
    direction = torch.randn(D, device="cuda")
    query = (
        direction.view(1, 1, 1, D) + 0.2 * torch.randn(1, Q, H, D, device="cuda")
    ).to(dtype)
    keys = 0.2 * torch.randn(L, D, device="cuda")
    keys[0::2] += 0.6 * direction
    keys[1::2] -= 0.6 * direction
    keys = keys.to(dtype)

    cache, bt = _paged(keys, dtype)
    o_full = _decode(query, cache, bt, L, ws, dtype).float()

    parts = []
    for r in range(W):
        rk = keys[r::W]
        c, b = _paged(rk, dtype)
        # Clean API: pass the SAME global bound L on every rank; the wrapper folds
        # in cp_rank=r. (Equivalent to pre-subtracting and passing L - r.)
        o, lse = _decode(
            query,
            c,
            b,
            rk.shape[0],
            ws,
            dtype,
            causal_seqs=torch.tensor([L], device="cuda", dtype=torch.int32),
            cp_world=W,
            cp_rank=r,
            lse=True,
        )
        parts.append((o.float(), lse.float()))

    # Kernel stores LSE in log2 units (global_lse = lse_max + log2(sum exp2(...))),
    # so the natural-softmax normalizer of a slice is Z = 2**lse. Merge accordingly.
    z = torch.stack([torch.exp2(p[1]) for p in parts], 0)  # [W, B, Q, H]
    w = (z / z.sum(0)).unsqueeze(-1)  # [W, B, Q, H, 1]
    o_merge = sum(w[i] * parts[i][0] for i in range(W))

    rel = (o_merge - o_full).abs().max() / o_full.abs().max().clamp_min(1e-6)
    w0 = w[0].flatten()
    assert w0.min() < 0.4 or w0.max() > 0.6, (
        f"merge weight not skewed (range [{w0.min():.3f},{w0.max():.3f}]); "
        "LSE scaling would be untested"
    )
    assert (
        rel < tol
    ), f"DCP strided merge != full context: rel max|Δ|={rel:.3e} (tol {tol})"


def test_cp_rank_folds_into_causal_seqs():
    """Passing the global bound + cp_rank must be identical to pre-subtracting the
    rank (causal_seqs = global - rank, cp_rank=0). Guards the documented contract
    that callers pass the SAME global causal_seqs on every rank."""
    dtype = torch.float8_e4m3fn
    torch.manual_seed(4)
    L, W, r = 128, 2, 1
    ws = _workspace()
    query = torch.randn(1, Q, H, D, device="cuda").to(dtype)
    rk = (torch.randn(L, D, device="cuda") * 0.3).to(dtype)[r::W]
    c, b = _paged(rk, dtype)

    def call(cs, rank):
        return _decode(
            query,
            c,
            b,
            rk.shape[0],
            ws,
            dtype,
            causal_seqs=torch.tensor([cs], device="cuda", dtype=torch.int32),
            cp_world=W,
            cp_rank=rank,
            lse=True,
        )

    o_global, lse_global = call(L, r)  # clean API: global bound + cp_rank
    o_pre, lse_pre = call(L - r, 0)  # pre-subtracted bound
    torch.testing.assert_close(o_global, o_pre, rtol=0, atol=0)
    torch.testing.assert_close(lse_global, lse_pre, rtol=0, atol=0)


def test_return_lse_toggles_output_shape():
    """return_lse=False keeps the bare-tensor API; True returns (output, lse)."""
    torch.manual_seed(1)
    dtype, L = torch.float8_e4m3fn, 64
    ws = _workspace()
    query = torch.randn(1, Q, H, D, device="cuda").to(dtype)
    cache, bt = _paged(torch.randn(L, D, device="cuda").to(dtype), dtype)

    out = _decode(query, cache, bt, L, ws, dtype)  # defaults: cp_world=1, no LSE
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, Q, H, KV_LORA)

    o, lse = _decode(query, cache, bt, L, ws, dtype, lse=True)
    assert o.shape == (1, Q, H, KV_LORA)
    assert lse.shape == (1, Q, H)
    torch.testing.assert_close(o, out, rtol=0, atol=0)  # LSE path must not alter output


def test_dcp_rejects_non_fp8_dtype():
    """DCP args on a non-fp8 query must raise a clear error in the wrapper rather
    than silently ignore them or fail deep in the kernel. The guard runs before
    the kernel is invoked, so this does not depend on the (fp8-only) decode path."""
    torch.manual_seed(2)
    dtype, L = torch.bfloat16, 64
    ws = _workspace()
    query = torch.randn(1, Q, H, D, device="cuda", dtype=dtype)
    cache, bt = _paged(torch.randn(L, D, device="cuda", dtype=dtype), dtype)

    with pytest.raises(ValueError, match="fp8"):
        _decode(
            query,
            cache,
            bt,
            L,
            ws,
            dtype,
            causal_seqs=torch.tensor([L], device="cuda", dtype=torch.int32),
            cp_world=2,
        )


def test_dcp_requires_causal_seqs():
    """cp_world>1 without causal_seqs must raise: the kernel divides the bound by
    cp_world, so falling back to rank-local seq_lens would mask each rank to
    ~1/cp_world of its slice and silently produce a wrong partial."""
    torch.manual_seed(3)
    dtype, L = torch.float8_e4m3fn, 64
    ws = _workspace()
    query = torch.randn(1, Q, H, D, device="cuda").to(dtype)
    cache, bt = _paged(torch.randn(L, D, device="cuda").to(dtype), dtype)

    with pytest.raises(ValueError, match="causal_seqs"):
        _decode(query, cache, bt, L, ws, dtype, cp_world=2)


def _windowed_reference(query, keys, n, window_left, causal_mask):
    """Per-row slice of the visible keys, softmaxed in fp32."""
    q, k = query.float()[0], keys.float()[:n]
    out = torch.empty(Q, H, KV_LORA, device="cuda")
    for row in range(Q):
        low = 0 if window_left < 0 else max(0, n - Q - window_left + row)
        high = n - (Q - 1) + row if causal_mask else n
        weights = torch.softmax((q[row] @ k[low:high].T) / (D**0.5), dim=-1)
        out[row] = weights @ k[low:high, :KV_LORA]
    return out.unsqueeze(0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("causal_mask", [False, True])
# 32 leaves only the last KV tiles alive, 320 keeps most of the cache with the
# low edge inside it, and 4096 opens the window past the cache entirely.
@pytest.mark.parametrize("window_left", [-1, 32, 320, 4096])
def test_sliding_window_matches_a_per_row_masked_reference(
    dtype, causal_mask, window_left
):
    torch.manual_seed(6)
    n, tol = 600, 6e-2 if dtype == torch.float8_e4m3fn else 2e-2
    ws = _workspace()
    query = (torch.randn(1, Q, H, D, device="cuda") / 8).to(dtype)
    keys = (torch.randn(n, D, device="cuda") / 8).to(dtype)
    cache, bt = _paged(keys, dtype)

    out = _decode(
        query,
        cache,
        bt,
        n,
        ws,
        dtype,
        causal_mask=causal_mask,
        window_left=window_left,
    ).float()

    expected = _windowed_reference(query, keys, n, window_left, causal_mask)
    assert (out - expected).abs().max() < tol * expected.abs().max()
