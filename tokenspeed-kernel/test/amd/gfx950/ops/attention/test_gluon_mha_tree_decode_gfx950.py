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

"""The gfx950 Gluon draft-tree decode against an fp32 reference.

Each request's last ``window`` keys are its tree window: query row ``r`` sees
every key before it (or the sliding window ending there) and window key ``j``
when bit ``j`` of its mask words is set. Covered: target verify rows (16 over
a 16-key window, head dim 256, GQA 6) and EAGLE-4 drafting lanes (4 rows over
16 and 44 keys, two mask words past 64, head dim 128, sliding), over BF16 and
FP8 E4M3 caches under BF16 and FP8 queries, prefixes shorter than a page,
padding requests without keys, and non-finite V in window slots no row sees.
"""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.tree import (  # noqa: E402
    tree_decode_attention,
    tree_decode_query_dtype,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.tree import (  # noqa: E402
    launch_gluon_mha_tree_decode_gfx950,
)

PAGE = 64
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16


def _problem(
    prefixes,
    rows,
    window,
    q_heads,
    kv_heads,
    dim,
    kv_dtype,
    q_dtype,
    seed,
    *,
    window_left=-1,
    unseen_nan=False,
):
    """Paged inputs and the fp32 reference over the stored (rounded) Q/K/V.

    A request with prefix ``None`` is padding: zero keys, its rows see nothing
    and must come out zero."""
    gen = torch.Generator().manual_seed(seed)
    bs = len(prefixes)
    seq_lens = [0 if p is None else p + window for p in prefixes]
    max_pages = max(1, (max(seq_lens) + PAGE - 1) // PAGE)
    table = torch.zeros(bs, max_pages + 3, dtype=torch.int32)
    num_pages = bs * max_pages + 1
    table[:, :max_pages] = (torch.randperm(bs * max_pages, generator=gen) + 1).view(
        bs, max_pages
    )
    k = torch.randn(num_pages, PAGE, kv_heads, dim, generator=gen).to(kv_dtype)
    v = torch.randn(num_pages, PAGE, kv_heads, dim, generator=gen).to(kv_dtype)
    # Scale the queries so the softmax is far from uniform.
    q = (2 * torch.randn(bs * rows, q_heads, dim, generator=gen)).to(q_dtype)
    words = (window + 63) // 64
    row_bits = []
    for _ in range(bs * rows):
        bits = int(torch.randint(0, 1 << 62, (1,), generator=gen)) | (
            int(torch.randint(0, 1 << 62, (1,), generator=gen)) << 62
        )
        bits |= int(torch.randint(0, 1 << 62, (1,), generator=gen)) << 124
        if unseen_nan:
            # Rows see only step 1's slots, as before later lanes are drafted.
            bits &= (1 << rows) - 1
        row_bits.append(bits & ((1 << window) - 1))
    mask_words = [
        ((b >> (64 * w)) & ((1 << 64) - 1)) for b in row_bits for w in range(words)
    ]
    mask = torch.tensor(
        [x - (1 << 64) if x >= 1 << 63 else x for x in mask_words], dtype=torch.int64
    )

    flat_k = k.view(-1, kv_heads, dim)
    flat_v = v.view(-1, kv_heads, dim)
    group = q_heads // kv_heads
    ref = torch.zeros(bs * rows, q_heads, dim)
    for b, length in enumerate(seq_lens):
        if length == 0:
            continue
        pos = torch.arange(length)
        slots = (table[b, pos // PAGE].long() * PAGE + pos % PAGE).tolist()
        kk = flat_k[slots].float().repeat_interleave(group, 1)
        vv = flat_v[slots].float().repeat_interleave(group, 1)
        prefix = length - window
        first = 0 if window_left < 0 else max(prefix - 1 - window_left, 0)
        for r in range(rows):
            bits = row_bits[b * rows + r]
            vis = torch.tensor(
                [
                    first <= j < prefix
                    or (j >= prefix and bool((bits >> (j - prefix)) & 1))
                    for j in range(length)
                ]
            )
            s = torch.einsum("hd,xhd->hx", q[b * rows + r].float(), kk) * dim**-0.5
            s = s.masked_fill(~vis[None], float("-inf"))
            p = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
            ref[b * rows + r] = torch.einsum("hx,xhd->hd", p, vv)
    if unseen_nan:
        for b, length in enumerate(seq_lens):
            if length == 0:
                continue
            seen = 0
            for r in range(rows):
                seen |= row_bits[b * rows + r]
            prefix = length - window
            for j in range(window):
                if not (seen >> j) & 1:
                    pos = prefix + j
                    flat_v[int(table[b, pos // PAGE]) * PAGE + pos % PAGE] = float(
                        "nan"
                    )
    return (
        q.cuda(),
        k.cuda(),
        v.cuda(),
        table.cuda(),
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        mask.cuda(),
        ref,
    )


def _check(out, ref, q_dtype):
    assert out.dtype == BF16
    out = out.float().cpu()
    assert torch.isfinite(out).all()
    # An FP8 query also rounds its probabilities to FP8.
    tol = 6e-2 if q_dtype == FP8 else 2e-2
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


DTYPES = [(BF16, BF16), (BF16, FP8), (FP8, FP8)]
DTYPE_IDS = ["bf16", "w8a16", "w8a8"]


@pytest.mark.parametrize("q_dtype,kv_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("bs", [1, 5, 8, 32])
def test_verify_rows(q_dtype, kv_dtype, bs):
    """Target verify: 16 rows over a 16-key window, head dim 256, 24/4 heads.
    From 8 requests (32 requests x KV heads) the launcher takes the wide
    tiles of its cache dtype."""
    prefixes = ([700, 3, 0, None, 4100, 65, 129, 1500] * 4)[:bs] if bs > 1 else [1000]
    q, k, v, table, seq_lens, mask, ref = _problem(
        prefixes, 16, 16, 24, 4, 256, kv_dtype, q_dtype, seed=bs
    )
    out = tree_decode_attention(
        q, k, v, table, seq_lens, mask,
        rows_per_req=16, window=16, max_seqlen_k=8192, sm_scale=256**-0.5,
        solution="gluon",
    )  # fmt: skip
    _check(out, ref, q_dtype)


@pytest.mark.parametrize("q_dtype,kv_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("window", [16, 44, 80])
@pytest.mark.parametrize("window_left", [-1, 40, 4095])
def test_lane_rows(q_dtype, kv_dtype, window, window_left):
    """EAGLE-4 lanes: 4 rows over (steps - 1) * 4 window keys; 80 keys take a
    second mask word. A sliding draft layer sees the window ending at the
    prefix end."""
    prefixes = [700, 41, 3, None, 130, 5000]
    q, k, v, table, seq_lens, mask, ref = _problem(
        prefixes, 4, window, 40, 40, 128, kv_dtype, q_dtype,
        seed=window + window_left, window_left=window_left,
    )  # fmt: skip
    out = tree_decode_attention(
        q, k, v, table, seq_lens, mask,
        rows_per_req=4, window=window, max_seqlen_k=8192,
        window_left=window_left, sm_scale=128**-0.5, solution="gluon",
    )  # fmt: skip
    _check(out, ref, q_dtype)


@pytest.mark.parametrize("kv_dtype", [BF16, FP8])
def test_unseen_window_slots_stay_out(kv_dtype):
    """Window slots no row sees hold NaN V (lanes not drafted yet); a zero
    probability must not turn them into NaN outputs."""
    q, k, v, table, seq_lens, mask, ref = _problem(
        [700, 2, 130], 4, 44, 40, 40, 128, kv_dtype, BF16, seed=3,
        window_left=4095, unseen_nan=True,
    )  # fmt: skip
    out = launch_gluon_mha_tree_decode_gfx950(
        q, k, v, table, seq_lens, mask,
        rows_per_req=4, window=44, max_seqlen_k=8192, window_left=4095,
    )  # fmt: skip
    _check(out, ref, BF16)


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_fp8_cache_takes_a_16_bit_query(sliding, head_dim):
    """Over an FP8 cache the tree rows keep the activation dtype."""
    assert (
        tree_decode_query_dtype(
            kv_cache_dtype=FP8,
            activation_dtype=BF16,
            head_dim=head_dim,
            page_size=PAGE,
            sliding_window=sliding,
        )
        == BF16
    )


def test_rejects_mismatched_inputs():
    q, k, v, table, seq_lens, mask, _ = _problem(
        [100], 4, 16, 8, 8, 128, BF16, BF16, seed=0
    )
    with pytest.raises(ValueError, match="mask words"):
        launch_gluon_mha_tree_decode_gfx950(
            q, k, v, table, seq_lens, mask,
            rows_per_req=4, window=130, max_seqlen_k=1024,
        )  # fmt: skip
    with pytest.raises(ValueError, match="int64 words"):
        launch_gluon_mha_tree_decode_gfx950(
            q, k, v, table, seq_lens, mask[:2],
            rows_per_req=4, window=16, max_seqlen_k=1024,
        )  # fmt: skip
    with pytest.raises(ValueError, match="matching dtypes"):
        launch_gluon_mha_tree_decode_gfx950(
            q.to(FP8), k, v, table, seq_lens, mask,
            rows_per_req=4, window=16, max_seqlen_k=1024,
        )  # fmt: skip
