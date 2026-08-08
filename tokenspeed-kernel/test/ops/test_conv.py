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
    inkling_sconv,
    sconv_cache_update,
    seq_idx_from_cu_seqlens,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sconv tests require a CUDA GPU."
)

W = 4
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


def unified_decode(x, weight, conv_cache, cache_indices):
    """T=1-per-request call of the unified kernel (the former sconv_decode)."""
    B = x.shape[0]
    device = x.device
    qsl = torch.arange(B + 1, dtype=torch.int32, device=device)
    return inkling_sconv(
        x,
        weight,
        conv_cache,
        qsl,
        qsl[:B],
        cache_indices,
        torch.ones(B, dtype=torch.bool, device=device),
    )


def ref_cache_row(
    x_seq: torch.Tensor, old_row: torch.Tensor, has_state: bool
) -> torch.Tensor:
    """Expected cache row after update: last W-1 tokens of [prev ++ x_seq]."""
    prev = old_row if has_state else torch.zeros_like(old_row)
    return torch.cat([prev, x_seq])[-(W - 1) :]


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
    seed: int = 0,
):
    torch.manual_seed(seed)
    T = sum(lens)
    x = torch.randn(T, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    conv_cache = torch.randn(num_slots, W - 1, D, device=device, dtype=DTYPE)
    cu_seqlens = _make_cu_seqlens(lens, device)
    seq_idx = seq_idx_from_cu_seqlens(cu_seqlens, T)
    return x, weight, conv_cache, cu_seqlens, seq_idx


def _ref_prefix(
    conv_cache: torch.Tensor, cache_index: int, has_state: bool, D: int
) -> torch.Tensor:
    if has_state and cache_index != PAD_SLOT_ID:
        return conv_cache[cache_index]
    return torch.zeros(W - 1, D, device=conv_cache.device, dtype=conv_cache.dtype)


@pytest.mark.parametrize("D", [2048, 6144])
@pytest.mark.parametrize("use_residual", [True, False])
def test_sconv_prefill_varlen(D: int, use_residual: bool, device: str) -> None:
    lens = [3, 850, 1]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, seed=0
    )
    cache_indices = torch.tensor([2, 5, PAD_SLOT_ID], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, False, True], device=device)
    cache_snapshot = conv_cache.clone()

    y = inkling_sconv(
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        use_residual=use_residual,
    )

    cu = cu_seqlens.tolist()
    for i in range(len(lens)):
        s, e = cu[i], cu[i + 1]
        prefix = _ref_prefix(
            cache_snapshot, int(cache_indices[i]), bool(has_initial_state[i]), D
        )
        ref = ref_sconv(x[s:e], weight, prefix, use_residual=use_residual)
        torch.testing.assert_close(y[s:e], ref, atol=ATOL, rtol=RTOL)

    # Prefill must not modify the cache.
    assert torch.equal(conv_cache, cache_snapshot)


@pytest.mark.parametrize("D", [2048, 6144])
def test_sconv_cache_update_long_sequences(D: int, device: str) -> None:
    lens = [3, 850, 7]
    x, _, conv_cache, cu_seqlens, _ = _make_prefill_inputs(lens, D, device, seed=1)
    cache_indices = torch.tensor([2, 5, 7], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, False, True], device=device)
    old_cache = conv_cache.clone()

    sconv_cache_update(x, conv_cache, cu_seqlens, cache_indices, has_initial_state)

    cu = cu_seqlens.tolist()
    for i in range(len(lens)):
        s, e = cu[i], cu[i + 1]
        ci = int(cache_indices[i])
        expected = ref_cache_row(x[s:e], old_cache[ci], bool(has_initial_state[i]))
        assert torch.equal(conv_cache[ci], expected)

    # Untouched slots keep their old content.
    for slot in range(conv_cache.shape[0]):
        if slot not in (2, 5, 7):
            assert torch.equal(conv_cache[slot], old_cache[slot])


@pytest.mark.parametrize("query_len", [1, 2])
@pytest.mark.parametrize("has_state", [True, False])
def test_sconv_cache_update_short_sequences(
    query_len: int, has_state: bool, device: str
) -> None:
    D = 2048
    lens = [query_len, query_len]
    x, _, conv_cache, cu_seqlens, _ = _make_prefill_inputs(lens, D, device, seed=2)
    cache_indices = torch.tensor([1, 4], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([has_state, has_state], device=device)
    old_cache = conv_cache.clone()

    sconv_cache_update(x, conv_cache, cu_seqlens, cache_indices, has_initial_state)

    cu = cu_seqlens.tolist()
    for i in range(len(lens)):
        s, e = cu[i], cu[i + 1]
        ci = int(cache_indices[i])
        expected = ref_cache_row(x[s:e], old_cache[ci], has_state)
        assert torch.equal(conv_cache[ci], expected)


def test_sconv_cache_update_pad_row_does_not_clobber_slot_zero(device: str) -> None:
    """Regression test: PAD rows must be fully masked out, not clamped to slot 0.

    The TML reference clamped cache_indices == PAD_SLOT_ID to slot 0 and wrote
    unconditionally, racing against (and clobbering) the real occupant of
    slot 0.
    """
    D = 2048
    lens = [5, 5]
    x, _, conv_cache, cu_seqlens, _ = _make_prefill_inputs(lens, D, device, seed=3)
    # Slot 0 holds a real request's state; the batch has a PAD row.
    cache_indices = torch.tensor([PAD_SLOT_ID, 3], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, True], device=device)
    old_cache = conv_cache.clone()

    sconv_cache_update(x, conv_cache, cu_seqlens, cache_indices, has_initial_state)

    # Slot 0 (and every slot other than 3) is untouched by the PAD row.
    assert torch.equal(conv_cache[0], old_cache[0])
    cu = cu_seqlens.tolist()
    expected = ref_cache_row(x[cu[1] : cu[2]], old_cache[3], True)
    assert torch.equal(conv_cache[3], expected)
    for slot in range(conv_cache.shape[0]):
        if slot != 3:
            assert torch.equal(conv_cache[slot], old_cache[slot])


@pytest.mark.parametrize("D", [2048, 6144])
@pytest.mark.parametrize("B", [1, 64, 300])
def test_sconv_decode(D: int, B: int, device: str) -> None:
    torch.manual_seed(4)
    num_slots = max(2 * B, 8)
    x = torch.randn(B, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    conv_cache = torch.randn(num_slots, W - 1, D, device=device, dtype=DTYPE)
    cache_indices = torch.randperm(num_slots, device=device)[:B].to(torch.int32)
    pad_rows: list[int] = []
    if B >= 2:
        pad_rows = [0, B - 1]
        cache_indices[pad_rows] = PAD_SLOT_ID
    old_cache = conv_cache.clone()

    y = unified_decode(x, weight, conv_cache, cache_indices)

    # Decode is read-only on the cache; persistence is a separate step.
    assert torch.equal(conv_cache, old_cache)
    qsl = torch.arange(B + 1, dtype=torch.int32, device=device)
    sconv_cache_update(
        x,
        conv_cache,
        qsl,
        cache_indices,
        torch.ones(B, dtype=torch.bool, device=device),
    )

    zeros = torch.zeros(W - 1, D, device=device, dtype=DTYPE)
    for i in range(B):
        ci = int(cache_indices[i])
        prefix = old_cache[ci] if ci != PAD_SLOT_ID else zeros
        ref = ref_sconv(x[i : i + 1], weight, prefix)
        torch.testing.assert_close(y[i : i + 1], ref, atol=ATOL, rtol=RTOL)
        if ci != PAD_SLOT_ID:
            expected_row = torch.cat([old_cache[ci][1:], x[i : i + 1]])
            assert torch.equal(conv_cache[ci], expected_row)

    # Slots not referenced by any valid row (incl. PAD rows) are untouched.
    valid = {int(c) for c in cache_indices.tolist() if c != PAD_SLOT_ID}
    for slot in range(num_slots):
        if slot not in valid:
            assert torch.equal(conv_cache[slot], old_cache[slot])


def test_sconv_chained_prefill_update_decode(device: str) -> None:
    """Full prefill == partial prefill + cache_update + 3 decode steps."""
    D = 2048
    num_decode = 3
    lens = [8, 12]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, seed=5
    )
    cache_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([False, False], device=device)

    # Reference: one prefill over the full sequences (no initial state).
    y_full = inkling_sconv(
        x, weight, conv_cache, cu_seqlens, seq_idx, cache_indices, has_initial_state
    )

    # Chained: prefill over lens - 3 tokens, write back, then 3 decode steps.
    part_lens = [n - num_decode for n in lens]
    cu_part = _make_cu_seqlens(part_lens, device)
    T_part = sum(part_lens)
    seq_idx_part = seq_idx_from_cu_seqlens(cu_part, T_part)
    cu = cu_seqlens.tolist()
    x_part = torch.cat(
        [x[cu[i] : cu[i] + part_lens[i]] for i in range(len(lens))]
    ).contiguous()

    y_part = inkling_sconv(
        x_part,
        weight,
        conv_cache,
        cu_part,
        seq_idx_part,
        cache_indices,
        has_initial_state,
    )
    sconv_cache_update(x_part, conv_cache, cu_part, cache_indices, has_initial_state)

    y_decode = []
    qsl_step = torch.arange(len(lens) + 1, dtype=torch.int32, device=device)
    ones_step = torch.ones(len(lens), dtype=torch.bool, device=device)
    for j in range(num_decode):
        x_step = torch.stack(
            [x[cu[i] + part_lens[i] + j] for i in range(len(lens))]
        ).contiguous()
        y_decode.append(unified_decode(x_step, weight, conv_cache, cache_indices))
        sconv_cache_update(x_step, conv_cache, qsl_step, cache_indices, ones_step)

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
    """All three ops must work on a channel-sliced view of a wider cache."""
    torch.manual_seed(6)
    D, off = 2048, 64
    D_total = D + 3 * off
    num_slots = 8
    lens = [6, 2]
    B = len(lens)
    T = sum(lens)

    x = torch.randn(T, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    cache_full = torch.randn(num_slots, W - 1, D_total, device=device, dtype=DTYPE)
    cache_view = cache_full[:, :, off : off + D]
    assert not cache_view.is_contiguous()

    cu_seqlens = _make_cu_seqlens(lens, device)
    seq_idx = seq_idx_from_cu_seqlens(cu_seqlens, T)
    cache_indices = torch.tensor([0, 4], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, True], device=device)
    snapshot = cache_full.clone()

    outside = torch.ones(D_total, dtype=torch.bool, device=device)
    outside[off : off + D] = False

    # Prefill on the view: correct output, cache untouched.
    y = inkling_sconv(
        x,
        weight,
        cache_view,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
    )
    cu = cu_seqlens.tolist()
    for i in range(B):
        s, e = cu[i], cu[i + 1]
        ref = ref_sconv(x[s:e], weight, snapshot[cache_indices[i], :, off : off + D])
        torch.testing.assert_close(y[s:e], ref, atol=ATOL, rtol=RTOL)
    assert torch.equal(cache_full, snapshot)

    # Cache update on the view: in-slice rows updated, outside channels intact.
    sconv_cache_update(x, cache_view, cu_seqlens, cache_indices, has_initial_state)
    for i in range(B):
        s, e = cu[i], cu[i + 1]
        ci = int(cache_indices[i])
        expected = ref_cache_row(x[s:e], snapshot[ci, :, off : off + D], True)
        assert torch.equal(cache_view[ci], expected)
    assert torch.equal(cache_full[:, :, outside], snapshot[:, :, outside])

    # Decode on the view.
    snapshot = cache_full.clone()
    x_dec = torch.randn(B, D, device=device, dtype=DTYPE)
    y_dec = unified_decode(x_dec, weight, cache_view, cache_indices)
    assert torch.equal(cache_full, snapshot)  # decode is read-only now
    sconv_cache_update(
        x_dec,
        cache_view,
        torch.arange(B + 1, dtype=torch.int32, device=device),
        cache_indices,
        torch.ones(B, dtype=torch.bool, device=device),
    )
    for i in range(B):
        ci = int(cache_indices[i])
        ref = ref_sconv(x_dec[i : i + 1], weight, snapshot[ci, :, off : off + D])
        torch.testing.assert_close(y_dec[i : i + 1], ref, atol=ATOL, rtol=RTOL)
        expected_row = torch.cat([snapshot[ci, 1:, off : off + D], x_dec[i : i + 1]])
        assert torch.equal(cache_view[ci], expected_row)
    assert torch.equal(cache_full[:, :, outside], snapshot[:, :, outside])
