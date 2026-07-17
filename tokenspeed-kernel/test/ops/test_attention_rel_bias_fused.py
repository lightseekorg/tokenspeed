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

"""Fused rel_bias (ShearingBias) path parity inside the rel_mha family.

The registered rel_mha kernels route to the fused sheared-bias path when the
installed flash-attn build ships it and the call satisfies its constraints
(extent multiple of 128; with a sliding window the window length must equal
the extent — the kernel slices the table to it). These tests pin the fused
route against both the torch reference and the score_mod gather route on
identical inputs, plus the fallback guards.

Skipped when the installed flash-attn lacks ``rel_bias`` (stock wheels).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_attention_rel_mha import (  # noqa: E402
    DTYPE,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    _build_paged,
    _cu,
    _ref_rel_attn,
    _require_fa4,
)
from tokenspeed_kernel import (  # noqa: E402
    rel_mha_decode_with_kvcache,
    rel_mha_prefill,
)
from tokenspeed_kernel.ops.attention import flash_attn as fa_mod  # noqa: E402

torch.manual_seed(11)

TOL = 2e-2

needs_fused = pytest.mark.skipif(
    not getattr(fa_mod, "_FA4_HAS_FUSED_REL_BIAS", False),
    reason="installed flash-attn build has no fused rel_bias path",
)


def _prefill_case(device, seq_lens, rel_extent):
    total = sum(seq_lens)
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = torch.randn(total, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.1
    return q, k, v, rel


@needs_fused
@pytest.mark.parametrize(
    "rel_extent,window_left",
    [(128, -1), (256, -1), (256, 127)],
    ids=["full-e128", "full-e256", "swa-127-sliced-from-e256"],
)
def test_rel_mha_prefill_fused_matches_reference_and_scoremod(
    device: str, require, rel_extent: int, window_left: int
) -> None:
    _require_fa4(require)
    seq_lens = [192, 384]
    q, k, v, rel = _prefill_case(device, seq_lens, rel_extent)
    cu = _cu(seq_lens, device)
    scale = 1.0 / HEAD_DIM

    kwargs = dict(
        rel_logits=rel,
        cu_seqlens=cu,
        cu_seqlens_cpu=[0] + list(torch.tensor(seq_lens).cumsum(0)),
        max_seqlen=max(seq_lens),
        window_left=window_left,
        softmax_scale=scale,
    )
    out_fused = rel_mha_prefill(q=q, k=k, v=v, **kwargs)

    # same inputs through the score_mod gather route
    orig = fa_mod._FA4_HAS_FUSED_REL_BIAS
    fa_mod._FA4_HAS_FUSED_REL_BIAS = False
    try:
        out_scoremod = rel_mha_prefill(q=q, k=k, v=v, **kwargs)
    finally:
        fa_mod._FA4_HAS_FUSED_REL_BIAS = orig

    torch.testing.assert_close(out_fused, out_scoremod, atol=TOL, rtol=TOL)

    off = 0
    for length in seq_lens:
        ref = _ref_rel_attn(
            q[off : off + length],
            k[off : off + length],
            v[off : off + length],
            rel[off : off + length],
            rel_extent,
            window_left,
            scale,
        )
        torch.testing.assert_close(
            out_fused[off : off + length], ref, atol=TOL, rtol=TOL
        )
        off += length


@needs_fused
@pytest.mark.parametrize(
    "rel_extent,window_left",
    [(128, -1), (256, 127)],
    ids=["full-e128", "swa-127-sliced-from-e256"],
)
def test_rel_mha_decode_fused_matches_reference_and_scoremod(
    device: str, require, rel_extent: int, window_left: int
) -> None:
    _require_fa4(require)
    kv_lens = [200, 513, 64]
    batch = len(kv_lens)
    k_cache, v_cache, page_table, ks, vs = _build_paged(kv_lens, device)
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel = torch.randn(batch, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.1
    cache_seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    cu_q = torch.arange(batch + 1, device=device, dtype=torch.int32)
    scale = 1.0 / HEAD_DIM

    kwargs = dict(
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max(kv_lens),
        rel_logits=rel,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        window_left=window_left,
        softmax_scale=scale,
    )
    out_fused = rel_mha_decode_with_kvcache(
        q=q, k_cache=k_cache, v_cache=v_cache, **kwargs
    )

    orig = fa_mod._FA4_HAS_FUSED_REL_BIAS
    fa_mod._FA4_HAS_FUSED_REL_BIAS = False
    try:
        out_scoremod = rel_mha_decode_with_kvcache(
            q=q, k_cache=k_cache, v_cache=v_cache, **kwargs
        )
    finally:
        fa_mod._FA4_HAS_FUSED_REL_BIAS = orig

    torch.testing.assert_close(out_fused, out_scoremod, atol=TOL, rtol=TOL)

    for i in range(batch):
        ref = _ref_rel_attn(
            q[i : i + 1],
            ks[i],
            vs[i],
            rel[i : i + 1],
            rel_extent,
            window_left,
            scale,
        )
        torch.testing.assert_close(out_fused[i : i + 1], ref, atol=TOL, rtol=TOL)


@needs_fused
def test_fused_falls_back_on_unaligned_extent(device: str, require) -> None:
    """Extents violating the fused-path constraints must route to score_mod."""
    _require_fa4(require)
    rel_96 = torch.zeros(4, NUM_Q_HEADS, 96, device=device, dtype=DTYPE)
    extra, _ = fa_mod._rel_attention_kwargs(rel_96, -1)
    assert "score_mod" in extra and "rel_bias" not in extra

    # window not matching a 128-multiple effective extent
    rel_256 = torch.zeros(4, NUM_Q_HEADS, 256, device=device, dtype=DTYPE)
    extra, _ = fa_mod._rel_attention_kwargs(rel_256, 96)
    assert "score_mod" in extra and "rel_bias" not in extra

    # aligned full-causal extent routes fused
    extra, _ = fa_mod._rel_attention_kwargs(rel_256, -1)
    assert "rel_bias" in extra
