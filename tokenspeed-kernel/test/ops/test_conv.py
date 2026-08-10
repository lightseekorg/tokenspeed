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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.conv import (
    PAD_SLOT_ID,
    inkling_ring_sconv,
    inkling_ring_sconv_update,
    seq_idx_from_cu_seqlens,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sconv tests require a CUDA GPU."
)

W = 4
R = 9  # current-config ring: (W-1) + K(4) + lookback(2); kernels read it from the cache shape
DTYPE = torch.bfloat16
ATOL = 1e-2
RTOL = 1e-2


def ref_sconv(
    x: torch.Tensor,
    weight: torch.Tensor,
    prefix: torch.Tensor,
    use_residual: bool = True,
) -> torch.Tensor:
    """Torch reference: causal FIR over [prefix ++ x] with optional residual."""
    xp = torch.cat([prefix, x]).float()
    y = sum(xp[w : w + len(x)] * weight[:, w].float() for w in range(weight.shape[1]))
    return (y + x.float() if use_residual else y).to(x.dtype)


def seed_ring_prefix(ring, slot, pre_len, prefix):
    """Place the last W-1 pre-chunk rows at their positions' ring rows."""
    for j in range(W - 1):
        pos = pre_len - (W - 1) + j
        if pos >= 0:
            ring[slot, pos % ring.shape[1]] = prefix[j]


def ring_rows_at(ring, slot, positions):
    return torch.stack([ring[slot, p % ring.shape[1]] for p in positions])


def unified_decode(x, weight, conv_cache, cache_indices, seq_lens):
    """T=1-per-request call of the unified kernel."""
    B = x.shape[0]
    device = x.device
    qsl = torch.arange(B + 1, dtype=torch.int32, device=device)
    return inkling_ring_sconv(
        x,
        weight,
        conv_cache,
        qsl,
        qsl[:B],
        cache_indices,
        torch.ones(B, dtype=torch.bool, device=device),
        seq_lens,
    )


def _make_cu_seqlens(lens: list[int], device: str) -> torch.Tensor:
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(
        torch.tensor(lens, dtype=torch.int64, device=device), dim=0
    ).to(torch.int32)
    return cu


def _make_prefill_inputs(
    lens: list[int],
    D: int,
    device: str,
    *,
    num_slots: int = 8,
    ring: int = R,
    seed: int = 0,
):
    torch.manual_seed(seed)
    T = sum(lens)
    x = torch.randn(T, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    conv_cache = torch.randn(num_slots, ring, D, device=device, dtype=DTYPE)
    cu_seqlens = _make_cu_seqlens(lens, device)
    seq_idx = seq_idx_from_cu_seqlens(cu_seqlens, T)
    return x, weight, conv_cache, cu_seqlens, seq_idx


@pytest.mark.parametrize("D", [2048, 6144])
@pytest.mark.parametrize("use_residual", [True, False])
def test_sconv_prefill_varlen(D: int, use_residual: bool, device: str) -> None:
    lens = [3, 850, 1]
    pre_lens = [128, 0, 256]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, seed=0
    )
    cache_indices = torch.tensor([2, 5, PAD_SLOT_ID], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, False, True], device=device)
    seq_lens = torch.tensor(
        [p + n for p, n in zip(pre_lens, lens)], dtype=torch.int32, device=device
    )
    prefixes = [
        torch.randn(W - 1, D, device=device, dtype=DTYPE) for _ in range(len(lens))
    ]
    seed_ring_prefix(conv_cache, 2, pre_lens[0], prefixes[0])
    cache_snapshot = conv_cache.clone()

    y = inkling_ring_sconv(
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        seq_lens,
        use_residual=use_residual,
    )

    zeros = torch.zeros(W - 1, D, device=device, dtype=DTYPE)
    cu = cu_seqlens.tolist()
    for i, prefix in enumerate((prefixes[0], zeros, zeros)):
        s, e = cu[i], cu[i + 1]
        ref = ref_sconv(x[s:e], weight, prefix, use_residual=use_residual)
        torch.testing.assert_close(y[s:e], ref, atol=ATOL, rtol=RTOL)

    # Short chunks are persisted in-kernel at their positions' ring rows;
    # the long chunk (850 > R - (W-1)) and the PAD row leave the ring alone.
    expected = cache_snapshot.clone()
    for j in range(lens[0]):
        expected[2, (pre_lens[0] + j) % R] = x[cu[0] + j]
    assert torch.equal(conv_cache, expected)

    # The follow-up pass refills the full ring depth from the long chunk.
    inkling_ring_sconv_update(
        x, conv_cache, cu_seqlens, seq_lens, cache_indices, kernel_width=W
    )
    for j in range(R):
        pos = seq_lens[1].item() - R + j
        expected[5, pos % R] = x[cu[2] - R + j]
    assert torch.equal(conv_cache, expected)


def test_ring_update_bounds_and_pad(device: str) -> None:
    D = 2048
    lens = [850, 4, 7, 5]
    pre_lens = [100, 40, 60, 30]
    x, _, conv_cache, cu_seqlens, _ = _make_prefill_inputs(
        lens, D, device, num_slots=10, seed=1
    )
    cache_indices = torch.tensor(
        [2, 5, 7, PAD_SLOT_ID], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor(
        [p + n for p, n in zip(pre_lens, lens)], dtype=torch.int32, device=device
    )
    old = conv_cache.clone()

    inkling_ring_sconv_update(
        x, conv_cache, cu_seqlens, seq_lens, cache_indices, kernel_width=W
    )

    expected = old.clone()
    cu = cu_seqlens.tolist()
    # Only chunks longer than R - (W-1) = 6 are written: lens 850 and 7. Each
    # persists its last min(chunk_len, R) rows (9 and 7 respectively).
    for i in (0, 2):
        for j in range(R):
            if lens[i] < R - j:
                continue
            pos = int(seq_lens[i]) - R + j
            expected[int(cache_indices[i]), pos % R] = x[cu[i + 1] - R + j]
    assert torch.equal(conv_cache, expected)


def test_ring_update_small_ring_borrows_nothing(device: str) -> None:
    """Non-spec ring (R=4): a 2-token chunk writes only in-chunk source rows;
    the pre-chunk position's ring row is left as is (position addressing
    needs no shift borrow)."""
    D = 2048
    small_r = W  # non-spec ring: (W-1) + 1
    lens = [2]
    pre_len = 10
    x, _, conv_cache, cu_seqlens, _ = _make_prefill_inputs(
        lens, D, device, ring=small_r, seed=2
    )
    cache_indices = torch.tensor([3], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([pre_len + lens[0]], dtype=torch.int32, device=device)
    old = conv_cache.clone()

    inkling_ring_sconv_update(
        x, conv_cache, cu_seqlens, seq_lens, cache_indices, kernel_width=W
    )

    expected = old.clone()
    # chunk_len 2 > R - (W-1) = 1, so the update runs; source rows exist only
    # for the last two ring positions (j = 2, 3).
    expected[3, (pre_len + 0) % small_r] = x[0]
    expected[3, (pre_len + 1) % small_r] = x[1]
    assert torch.equal(conv_cache, expected)


def test_sconv_prefill_then_lookback_window(device: str) -> None:
    """Round-1 draft lookback after a long prefill: the window starts at
    ``through - lookback`` and its first taps read down to ``through - 5``,
    deeper than W-1 — the ring_update pass must have persisted those rows."""
    D = 2048
    lb, k = 2, 4
    prefill_len = 850
    torch.manual_seed(7)
    x_full = torch.randn(prefill_len + k, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    conv_cache = torch.randn(4, R, D, device=device, dtype=DTYPE)
    cache_indices = torch.tensor([1], dtype=torch.int32, device=device)

    x_pre = x_full[:prefill_len].contiguous()
    cu_pre = _make_cu_seqlens([prefill_len], device)
    pre_seq_lens = torch.tensor([prefill_len], dtype=torch.int32, device=device)
    inkling_ring_sconv(
        x_pre,
        weight,
        conv_cache,
        cu_pre,
        seq_idx_from_cu_seqlens(cu_pre, prefill_len),
        cache_indices,
        torch.tensor([False], device=device),
        pre_seq_lens,
    )
    inkling_ring_sconv_update(
        x_pre, conv_cache, cu_pre, pre_seq_lens, cache_indices, kernel_width=W
    )

    # Lookback window: lb committed rows rewritten + k fresh rows, positions
    # prefill_len - lb .. prefill_len + k - lb - 1.
    start = prefill_len - lb
    x_win = x_full[start : start + lb + k].contiguous()
    cu_win = _make_cu_seqlens([lb + k], device)
    win_seq_lens = torch.tensor([start + lb + k], dtype=torch.int32, device=device)
    y_win = inkling_ring_sconv(
        x_win,
        weight,
        conv_cache,
        cu_win,
        seq_idx_from_cu_seqlens(cu_win, lb + k),
        cache_indices,
        torch.tensor([True], device=device),
        win_seq_lens,
    )

    ref = ref_sconv(x_win, weight, x_full[start - (W - 1) : start])
    torch.testing.assert_close(y_win, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("D", [2048, 6144])
@pytest.mark.parametrize("B", [1, 64, 300])
def test_sconv_decode(D: int, B: int, device: str) -> None:
    torch.manual_seed(4)
    num_slots = max(2 * B, 8)
    x = torch.randn(B, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    conv_cache = torch.randn(num_slots, R, D, device=device, dtype=DTYPE)
    cache_indices = torch.randperm(num_slots, device=device)[:B].to(torch.int32)
    seq_lens = torch.randint(
        W, 500, (B,), dtype=torch.int32, device=device
    )  # length INCLUDING the current token
    pad_rows: list[int] = []
    if B >= 2:
        pad_rows = [0, B - 1]
        cache_indices[pad_rows] = PAD_SLOT_ID
    old_cache = conv_cache.clone()

    y = unified_decode(x, weight, conv_cache, cache_indices, seq_lens)

    expected = old_cache.clone()
    for i in range(B):
        ci = int(cache_indices[i])
        L = int(seq_lens[i])
        if ci != PAD_SLOT_ID:
            prefix = ring_rows_at(old_cache, ci, range(L - W, L - 1))
            expected[ci, (L - 1) % R] = x[i]
        else:
            prefix = torch.zeros(W - 1, D, device=device, dtype=DTYPE)
        ref = ref_sconv(x[i : i + 1], weight, prefix)
        torch.testing.assert_close(y[i : i + 1], ref, atol=ATOL, rtol=RTOL)

    # Decode persists its own row in-kernel; everything else is untouched.
    assert torch.equal(conv_cache, expected)


def test_sconv_verify_overwrite_after_rejection(device: str) -> None:
    """Speculate-and-overwrite: a verify round writes all K rows; the next
    round at the accepted frontier overwrites the rejected positions and
    reads only accepted history."""
    torch.manual_seed(7)
    D, K, B = 512, 4, 2
    committed_len = 20
    steps = 3
    x_all = torch.randn(B, committed_len + K * (steps + 1), D, device=device).to(DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    conv_cache = torch.zeros(6, R, D, device=device, dtype=DTYPE)
    cache_indices = torch.tensor([1, 4], dtype=torch.int32, device=device)
    for b in range(B):
        seed_ring_prefix(
            conv_cache,
            int(cache_indices[b]),
            committed_len,
            x_all[b, committed_len - (W - 1) : committed_len],
        )

    qsl = torch.arange(0, B * K + 1, K, dtype=torch.int32, device=device)
    seq_idx = seq_idx_from_cu_seqlens(qsl, B * K)
    ones = torch.ones(B, dtype=torch.bool, device=device)
    accepts = [2, 1, 4]
    frontier = [committed_len] * B
    for step in range(steps):
        # Each request's verify window starts at its frontier; positions
        # beyond this round's accept carry REJECTED content that later
        # rounds must overwrite.
        chunks = []
        for b in range(B):
            chunk = x_all[b, frontier[b] : frontier[b] + K].clone()
            chunk[accepts[step] :] += 100.0
            chunks.append(chunk)
        xs = torch.cat(chunks)
        seq_lens = torch.tensor(
            [frontier[b] + K for b in range(B)], dtype=torch.int32, device=device
        )
        y = inkling_ring_sconv(
            xs, weight, conv_cache, qsl, seq_idx, cache_indices, ones, seq_lens
        )
        for b in range(B):
            ref = ref_sconv(
                chunks[b],
                weight,
                x_all[b, frontier[b] - (W - 1) : frontier[b]],
            )
            torch.testing.assert_close(
                y[b * K : (b + 1) * K], ref, atol=ATOL, rtol=RTOL
            )
        for b in range(B):
            x_all[b, frontier[b] : frontier[b] + accepts[step]] = chunks[b][
                : accepts[step]
            ]
            frontier[b] += accepts[step]

    # After all rounds the ring rows at the last (W-1) + lookback-capacity
    # accepted positions hold the committed inputs, not rejected leftovers.
    for b in range(B):
        ci = int(cache_indices[b])
        for pos in range(frontier[b] - 6, frontier[b]):
            torch.testing.assert_close(
                conv_cache[ci, pos % R], x_all[b, pos], atol=0, rtol=0
            )


def test_sconv_chained_prefill_update_decode(device: str) -> None:
    """Full prefill == partial prefill + ring_update + 3 decode steps."""
    D = 2048
    num_decode = 3
    lens = [12, 16]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, seed=5
    )
    cache_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([False, False], device=device)
    full_seq_lens = cu_seqlens.diff().to(torch.int32)

    # Reference: one prefill over the full sequences (no initial state).
    y_full = inkling_ring_sconv(
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        full_seq_lens,
    )

    # Chained: prefill over lens - 3 tokens, persist, then 3 decode steps.
    part_lens = [n - num_decode for n in lens]
    cu_part = _make_cu_seqlens(part_lens, device)
    T_part = sum(part_lens)
    seq_idx_part = seq_idx_from_cu_seqlens(cu_part, T_part)
    cu = cu_seqlens.tolist()
    x_part = torch.cat(
        [x[cu[i] : cu[i] + part_lens[i]] for i in range(len(lens))]
    ).contiguous()
    part_seq_lens = torch.tensor(part_lens, dtype=torch.int32, device=device)

    y_part = inkling_ring_sconv(
        x_part,
        weight,
        conv_cache,
        cu_part,
        seq_idx_part,
        cache_indices,
        has_initial_state,
        part_seq_lens,
    )
    inkling_ring_sconv_update(
        x_part, conv_cache, cu_part, part_seq_lens, cache_indices, kernel_width=W
    )

    y_decode = []
    for j in range(num_decode):
        x_step = torch.stack(
            [x[cu[i] + part_lens[i] + j] for i in range(len(lens))]
        ).contiguous()
        step_seq_lens = torch.tensor(
            [part_lens[i] + j + 1 for i in range(len(lens))],
            dtype=torch.int32,
            device=device,
        )
        y_decode.append(
            unified_decode(x_step, weight, conv_cache, cache_indices, step_seq_lens)
        )

    cu_p = cu_part.tolist()
    for i in range(len(lens)):
        s, e = cu[i], cu[i + 1]
        torch.testing.assert_close(
            y_full[s : s + part_lens[i]],
            y_part[cu_p[i] : cu_p[i + 1]],
            atol=ATOL,
            rtol=RTOL,
        )
        for j in range(num_decode):
            torch.testing.assert_close(
                y_full[s + part_lens[i] + j],
                y_decode[j][i],
                atol=ATOL,
                rtol=RTOL,
            )


def test_sconv_channel_sliced_cache_view(device: str) -> None:
    """Both ops must work on a channel-sliced view of a wider ring."""
    torch.manual_seed(6)
    D, off = 2048, 64
    D_total = D + 3 * off
    num_slots = 8
    lens = [5, 2]
    pre_lens = [50, 31]
    B = len(lens)
    T = sum(lens)

    x = torch.randn(T, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    cache_full = torch.randn(num_slots, R, D_total, device=device, dtype=DTYPE)
    cache_view = cache_full[:, :, off : off + D]
    assert not cache_view.is_contiguous()

    cu_seqlens = _make_cu_seqlens(lens, device)
    seq_idx = seq_idx_from_cu_seqlens(cu_seqlens, T)
    cache_indices = torch.tensor([0, 4], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, True], device=device)
    seq_lens = torch.tensor(
        [p + n for p, n in zip(pre_lens, lens)], dtype=torch.int32, device=device
    )
    snapshot = cache_full.clone()

    outside = torch.ones(D_total, dtype=torch.bool, device=device)
    outside[off : off + D] = False

    y = inkling_ring_sconv(
        x,
        weight,
        cache_view,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        seq_lens,
    )
    cu = cu_seqlens.tolist()
    for i in range(B):
        s, e = cu[i], cu[i + 1]
        ci = int(cache_indices[i])
        prefix = ring_rows_at(
            snapshot[:, :, off : off + D], ci, range(pre_lens[i] - (W - 1), pre_lens[i])
        )
        ref = ref_sconv(x[s:e], weight, prefix)
        torch.testing.assert_close(y[s:e], ref, atol=ATOL, rtol=RTOL)

    # Both chunks are short (<= R - (W-1)): persisted in-kernel, in-slice only.
    expected_view = snapshot[:, :, off : off + D].clone()
    for i in range(B):
        ci = int(cache_indices[i])
        for j in range(lens[i]):
            expected_view[ci, (pre_lens[i] + j) % R] = x[cu[i] + j]
    assert torch.equal(cache_view, expected_view)
    assert torch.equal(cache_full[:, :, outside], snapshot[:, :, outside])

    # ring_update on the view: no request exceeds the bound -> no-op.
    before = cache_full.clone()
    inkling_ring_sconv_update(
        x, cache_view, cu_seqlens, seq_lens, cache_indices, kernel_width=W
    )
    assert torch.equal(cache_full, before)
