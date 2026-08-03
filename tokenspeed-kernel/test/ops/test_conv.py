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
    sconv_cache_update,
    sconv_decode,
    sconv_prefill,
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

    y = sconv_prefill(
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

    y = sconv_decode(x, weight, conv_cache, cache_indices)

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
    y_full = sconv_prefill(
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

    y_part = sconv_prefill(
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
    for j in range(num_decode):
        x_step = torch.stack(
            [x[cu[i] + part_lens[i] + j] for i in range(len(lens))]
        ).contiguous()
        y_decode.append(sconv_decode(x_step, weight, conv_cache, cache_indices))

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
    y = sconv_prefill(
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
    y_dec = sconv_decode(x_dec, weight, cache_view, cache_indices)
    for i in range(B):
        ci = int(cache_indices[i])
        ref = ref_sconv(x_dec[i : i + 1], weight, snapshot[ci, :, off : off + D])
        torch.testing.assert_close(y_dec[i : i + 1], ref, atol=ATOL, rtol=RTOL)
        expected_row = torch.cat([snapshot[ci, 1:, off : off + D], x_dec[i : i + 1]])
        assert torch.equal(cache_view[ci], expected_row)
    assert torch.equal(cache_full[:, :, outside], snapshot[:, :, outside])


class TestSconvPaged:
    """Paged sconv (per-token input columns, SWA semantics) vs fp32 reference."""

    @pytest.mark.parametrize(
        "BT,slack",
        # 4 / 64 are the MXFP8 conv blocks (bf16 columns over halved fp8
        # byte slots): hiddenconv 4, kvconv 64.
        [
            (4, 0),
            (4, 2048),
            (8, 2048),
            (16, 0),
            (16, 96),
            (64, 0),
            (64, 96),
            (128, 0),
            (128, 256),
        ],
    )
    @pytest.mark.parametrize(
        "prefix_lens,chunk_lens",
        [
            ([0, 0], [16, 40]),
            ([17, 33, 5], [15, 3, 31]),  # non-aligned prefix restores
        ],
    )
    def test_prefill_and_decode_parity(self, prefix_lens, chunk_lens, BT, slack):
        import torch
        from tokenspeed_kernel.ops.conv import (
            sconv_decode_paged,
            sconv_prefill_paged,
        )

        torch.manual_seed(0)
        dev = "cuda"
        D, W = 512, 4
        weight = torch.randn(D, W, device=dev, dtype=torch.bfloat16) * 0.3
        B = len(prefix_lens)
        max_blocks = 64
        # Slot-level slack mimics the serving layout: conv columns are a
        # leading view of larger byte slots (hetero full layers), so the
        # slot stride exceeds BT * D and the view is non-contiguous.
        slot_elems = BT * D + slack
        slab = torch.zeros(max_blocks + 1, slot_elems, device=dev, dtype=torch.bfloat16)
        pool = slab[:, : BT * D].view(max_blocks + 1, BT, D)
        perm = torch.randperm(max_blocks, device=dev, dtype=torch.int32) + 1
        pt = torch.stack(
            [perm.roll(i)[: max_blocks // B] for i in range(B)]
        ).contiguous()
        full = [
            torch.randn(p + c, D, device=dev, dtype=torch.bfloat16)
            for p, c in zip(prefix_lens, chunk_lens)
        ]

        def ref(seq):
            pad = torch.zeros(W - 1, D, device=dev, dtype=torch.float32)
            s = torch.cat([pad, seq.float()])
            out = sum(s[j : j + len(seq)] * weight[:, j].float() for j in range(W))
            return out + seq.float()

        for b in range(B):
            p, c = prefix_lens[b], chunk_lens[b]
            if p:
                si = torch.zeros(p, device=dev, dtype=torch.int32)
                cu = torch.tensor([0, p], device=dev, dtype=torch.int32)
                pl = torch.tensor([0], device=dev, dtype=torch.int32)
                sconv_prefill_paged(
                    full[b][:p],
                    weight,
                    pool,
                    pt[b : b + 1],
                    si,
                    cu,
                    pl,
                    block_tokens=BT,
                )
            si = torch.zeros(c, device=dev, dtype=torch.int32)
            cu = torch.tensor([0, c], device=dev, dtype=torch.int32)
            pl = torch.tensor([p], device=dev, dtype=torch.int32)
            y = sconv_prefill_paged(
                full[b][p:],
                weight,
                pool,
                pt[b : b + 1],
                si,
                cu,
                pl,
                block_tokens=BT,
            )
            want = ref(full[b])[p : p + c]
            assert (y.float() - want).abs().max().item() < 0.02
            for t in range(3):
                xt = torch.randn(1, D, device=dev, dtype=torch.bfloat16)
                full[b] = torch.cat([full[b], xt])
                sl = torch.tensor([p + c + t + 1], device=dev, dtype=torch.int32)
                yd = sconv_decode_paged(
                    xt, weight, pool, pt[b : b + 1], sl, block_tokens=BT
                )
                want_t = ref(full[b])[-1:]
                assert (yd.float() - want_t).abs().max().item() < 0.02

    def test_holes_between_cached_blocks(self):
        """Punched (-1) table entries read as zero: a hole OUTSIDE the tap
        window changes nothing; a hole INSIDE it equals zeroed inputs."""
        import torch
        from tokenspeed_kernel.ops.conv import (
            sconv_decode_paged,
            sconv_prefill_paged,
        )

        torch.manual_seed(1)
        dev = "cuda"
        D, W, BT = 512, 4, 16
        weight = torch.randn(D, W, device=dev, dtype=torch.bfloat16) * 0.3
        pool = torch.zeros(9, BT, D, device=dev, dtype=torch.bfloat16)
        pt = torch.arange(1, 9, device=dev, dtype=torch.int32).unsqueeze(0)
        p, c = 40, 8  # prefix 40 (blocks 0-2), chunk crosses into block 3
        full = torch.randn(p + c, D, device=dev, dtype=torch.bfloat16)

        def ref(seq):
            pad = torch.zeros(W - 1, D, device=dev, dtype=torch.float32)
            s = torch.cat([pad, seq.float()])
            out = sum(s[j : j + len(seq)] * weight[:, j].float() for j in range(W))
            return out + seq.float()

        def run_chunk(table):
            si = torch.zeros(c, device=dev, dtype=torch.int32)
            cu = torch.tensor([0, c], device=dev, dtype=torch.int32)
            pl = torch.tensor([p], device=dev, dtype=torch.int32)
            return sconv_prefill_paged(
                full[p:], weight, pool, table, si, cu, pl, block_tokens=BT
            )

        # Persist the whole prefix once with a fully live table.
        si = torch.zeros(p, device=dev, dtype=torch.int32)
        cu = torch.tensor([0, p], device=dev, dtype=torch.int32)
        pl = torch.tensor([0], device=dev, dtype=torch.int32)
        sconv_prefill_paged(full[:p], weight, pool, pt, si, cu, pl, block_tokens=BT)

        # Hole OUTSIDE the tap window (block 0 = positions 0-15; the chunk's
        # taps reach back only to position 37): output identical to full.
        pt_outside = pt.clone()
        pt_outside[0, 0] = -1
        y = run_chunk(pt_outside)
        want = ref(full)[p:]
        assert (y.float() - want).abs().max().item() < 0.02

        # Hole INSIDE the tap window (block 2 = positions 32-47 holds taps
        # 37-39): exactly equals zeroing the holed inputs the chunk can see.
        pt_inside = pt.clone()
        pt_inside[0, 2] = -1
        y2 = run_chunk(pt_inside)
        full_zeroed = full.clone()
        full_zeroed[32:40] = 0
        want2 = ref(full_zeroed)[p:]
        assert (y2.float() - want2).abs().max().item() < 0.02

    def test_prefill_pad_to_max_request(self):
        """Breakable-prefill-graph padding: tokens past the real count carry
        seq_idx == the PAD request row (empty chunk closed at the real token
        count, prefix 0, all -1 table row). Real rows must be bit-identical
        to the unpadded call and the pool must not see a single write from
        the pad tokens — including via the stale rows between bs and PAD."""
        import torch
        from tokenspeed_kernel.ops.conv import sconv_prefill_paged

        torch.manual_seed(2)
        dev = "cuda"
        D, W, BT = 256, 4, 8
        MAX_BS = 4  # PAD row index; static tables carry MAX_BS + 1 rows
        weight = torch.randn(D, W, device=dev, dtype=torch.bfloat16) * 0.3
        pool = torch.randn(12, BT, D, device=dev, dtype=torch.bfloat16)
        prefix = torch.tensor([5, 0], device=dev, dtype=torch.int32)
        bs, T, bucket = 2, 10, 32  # chunks 6 + 4, padded to a 32-token bucket
        table = torch.full((MAX_BS + 1, 6), -1, device=dev, dtype=torch.int32)
        table[0, :2] = torch.tensor([3, 7], device=dev, dtype=torch.int32)
        table[1, :2] = torch.tensor([5, 9], device=dev, dtype=torch.int32)
        # Stale rows between bs and PAD point at a live block: any read of
        # them would corrupt slot 11 and fail the pool comparison.
        table[bs:MAX_BS, :] = 11
        x = torch.randn(bucket, D, device=dev, dtype=torch.bfloat16)

        cu_live = torch.tensor([0, 6, 10], device=dev, dtype=torch.int32)
        si_live = torch.tensor([0] * 6 + [1] * 4, device=dev, dtype=torch.int32)
        pool_ref = pool.clone()
        y_ref = sconv_prefill_paged(
            x[:T],
            weight,
            pool_ref,
            table[:bs],
            si_live,
            cu_live,
            prefix,
            block_tokens=BT,
        )

        # The padded static metadata exactly as the serving wrapper lands it.
        cu_pad = torch.zeros(MAX_BS + 2, device=dev, dtype=torch.int32)
        cu_pad[: bs + 1] = cu_live
        cu_pad[MAX_BS:] = T
        si_pad = torch.full((bucket,), MAX_BS, device=dev, dtype=torch.int32)
        si_pad[:T] = si_live
        pl_pad = torch.zeros(MAX_BS + 1, device=dev, dtype=torch.int32)
        pl_pad[:bs] = prefix
        pool_pad = pool.clone()
        y_pad = sconv_prefill_paged(
            x, weight, pool_pad, table, si_pad, cu_pad, pl_pad, block_tokens=BT
        )
        assert torch.equal(y_pad[:T], y_ref)
        assert torch.equal(pool_pad, pool_ref)

    @pytest.mark.parametrize("BT,ch", [(128, 512), (128, 256), (8, 512)])
    def test_dual_pool_fused_matches_two_calls(self, BT, ch):
        """One dual-pool K+V call == two single-pool calls, bit for bit."""
        import torch
        from tokenspeed_kernel.ops.conv import (
            sconv_decode_paged,
            sconv_prefill_paged,
        )

        torch.manual_seed(3)
        dev = "cuda"
        W, D = 4, 2 * ch
        weight = torch.randn(D, W, device=dev, dtype=torch.bfloat16) * 0.3
        n_slots = 40
        slot_elems = BT * ch + 64  # slack slots, like the serving views

        def pools():
            k = torch.zeros(n_slots, slot_elems, device=dev, dtype=torch.bfloat16)
            v = torch.zeros(n_slots, slot_elems, device=dev, dtype=torch.bfloat16)
            return (
                k[:, : BT * ch].view(n_slots, BT, ch),
                v[:, : BT * ch].view(n_slots, BT, ch),
            )

        p, c = 3 * BT + 5, 17
        full = torch.randn(p + c, D, device=dev, dtype=torch.bfloat16)
        pt = (torch.randperm(n_slots - 1, device=dev, dtype=torch.int32) + 1)[
            : (p + c + BT) // BT + 1
        ].unsqueeze(0)
        si = torch.zeros(p + c, device=dev, dtype=torch.int32)

        def run(fused):
            kp, vp = pools()
            cu = torch.tensor([0, p], device=dev, dtype=torch.int32)
            pl = torch.tensor([0], device=dev, dtype=torch.int32)
            if fused:
                sconv_prefill_paged(
                    full[:p],
                    weight,
                    kp,
                    pt,
                    si[:p],
                    cu,
                    pl,
                    block_tokens=BT,
                    col_pool2=vp,
                    half_d=ch,
                )
            else:
                sconv_prefill_paged(
                    full[:p, :ch],
                    weight[:ch],
                    kp,
                    pt,
                    si[:p],
                    cu,
                    pl,
                    block_tokens=BT,
                )
                sconv_prefill_paged(
                    full[:p, ch:],
                    weight[ch:],
                    vp,
                    pt,
                    si[:p],
                    cu,
                    pl,
                    block_tokens=BT,
                )
            cu2 = torch.tensor([0, c], device=dev, dtype=torch.int32)
            pl2 = torch.tensor([p], device=dev, dtype=torch.int32)
            if fused:
                y = sconv_prefill_paged(
                    full[p:],
                    weight,
                    kp,
                    pt,
                    si[:c],
                    cu2,
                    pl2,
                    block_tokens=BT,
                    col_pool2=vp,
                    half_d=ch,
                )
            else:
                y = torch.cat(
                    [
                        sconv_prefill_paged(
                            full[p:, :ch],
                            weight[:ch],
                            kp,
                            pt,
                            si[:c],
                            cu2,
                            pl2,
                            block_tokens=BT,
                        ),
                        sconv_prefill_paged(
                            full[p:, ch:],
                            weight[ch:],
                            vp,
                            pt,
                            si[:c],
                            cu2,
                            pl2,
                            block_tokens=BT,
                        ),
                    ],
                    dim=1,
                )
            xt = full[-1:].clone()
            sl = torch.tensor([p + c + 1], device=dev, dtype=torch.int32)
            if fused:
                yd = sconv_decode_paged(
                    xt,
                    weight,
                    kp,
                    pt,
                    sl,
                    block_tokens=BT,
                    col_pool2=vp,
                    half_d=ch,
                )
            else:
                yd = torch.cat(
                    [
                        sconv_decode_paged(
                            xt[:, :ch], weight[:ch], kp, pt, sl, block_tokens=BT
                        ),
                        sconv_decode_paged(
                            xt[:, ch:], weight[ch:], vp, pt, sl, block_tokens=BT
                        ),
                    ],
                    dim=1,
                )
            return y, yd, kp, vp

        y1, yd1, kp1, vp1 = run(fused=False)
        y2, yd2, kp2, vp2 = run(fused=True)
        assert torch.equal(y1, y2)
        assert torch.equal(yd1, yd2)
        assert torch.equal(kp1, kp2) and torch.equal(vp1, vp2)

    def test_fp8_pool_direct_cast(self):
        """FP8 (e4m3, scale 1.0) column pools on the same kernels: persist is
        the direct cast (triton SATURATES at 448 — torch's cast NaNs), taps
        upcast through f32, and a chunk reading fp8 prefix taps is
        bit-identical to a bf16 pool preloaded with the round-tripped
        values. Compute stays bf16-in/bf16-out."""
        import torch
        from tokenspeed_kernel.ops.conv import (
            sconv_decode_paged,
            sconv_prefill_paged,
        )

        torch.manual_seed(3)
        dev = "cuda"
        D, W, BT = 128, 4, 8
        fp8 = torch.float8_e4m3fn
        weight = torch.randn(D, W, device=dev, dtype=torch.bfloat16) * 0.3
        pt = torch.arange(1, 9, device=dev, dtype=torch.int32)[None, :]
        p, c = 24, 9
        full = torch.randn(p + c, D, device=dev, dtype=torch.bfloat16)

        pool8 = torch.zeros(10, BT, D, device=dev, dtype=fp8)
        pool16 = torch.zeros(10, BT, D, device=dev, dtype=torch.bfloat16)
        si = torch.zeros(p, device=dev, dtype=torch.int32)
        cu = torch.tensor([0, p], device=dev, dtype=torch.int32)
        pl = torch.zeros(1, device=dev, dtype=torch.int32)
        sconv_prefill_paged(full[:p], weight, pool16, pt, si, cu, pl, block_tokens=BT)
        y8p = sconv_prefill_paged(
            full[:p], weight, pool8, pt, si, cu, pl, block_tokens=BT
        )
        # persist == direct cast of the bf16 persist (values in fp8 range)
        assert torch.equal(pool8.to(torch.bfloat16), pool16.to(fp8).to(torch.bfloat16))

        # chunk reading fp8 prefix taps == bf16 pool holding round-tripped taps
        pool16.copy_(pool16.to(fp8).to(torch.bfloat16))
        si2 = torch.zeros(c, device=dev, dtype=torch.int32)
        cu2 = torch.tensor([0, c], device=dev, dtype=torch.int32)
        pl2 = torch.tensor([p], device=dev, dtype=torch.int32)
        y8 = sconv_prefill_paged(
            full[p:], weight, pool8, pt, si2, cu2, pl2, block_tokens=BT
        )
        y16 = sconv_prefill_paged(
            full[p:], weight, pool16, pt, si2, cu2, pl2, block_tokens=BT
        )
        assert torch.equal(y8, y16)
        assert y8.dtype == torch.bfloat16

        # decode: fp8 taps vs round-tripped bf16 taps (chunk tail already
        # round-tripped in BOTH pools after the copy_ above)
        pool16.copy_(pool8.to(torch.bfloat16))
        xd = torch.randn(1, D, device=dev, dtype=torch.bfloat16)
        sl = torch.tensor([p + c + 1], device=dev, dtype=torch.int32)
        d8 = sconv_decode_paged(xd, weight, pool8, pt, sl, block_tokens=BT)
        d16 = sconv_decode_paged(xd, weight, pool16, pt, sl, block_tokens=BT)
        assert torch.equal(d8, d16)

        # saturation: |x| > 448 persists as +-448, never NaN
        big = torch.full((8, D), 1000.0, device=dev, dtype=torch.bfloat16)
        sconv_prefill_paged(
            big,
            weight,
            pool8,
            pt,
            si[:8],
            torch.tensor([0, 8], device=dev, dtype=torch.int32),
            pl,
            block_tokens=BT,
        )
        stored = pool8.to(torch.bfloat16)[1, 0]
        assert (stored == 448.0).all() and not stored.isnan().any()

    def test_fp8_dual_pool_fused(self):
        """FP8 dual-pool (fused K+V halves, HALF_D) matches two single-pool
        fp8 calls bit-exactly."""
        import torch
        from tokenspeed_kernel.ops.conv import sconv_prefill_paged

        torch.manual_seed(4)
        dev = "cuda"
        D, W, BT = 256, 4, 8  # two 128-halves
        fp8 = torch.float8_e4m3fn
        half = D // 2
        weight = torch.randn(D, W, device=dev, dtype=torch.bfloat16) * 0.3
        pt = torch.arange(1, 5, device=dev, dtype=torch.int32)[None, :]
        T = 20
        x = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
        si = torch.zeros(T, device=dev, dtype=torch.int32)
        cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
        pl = torch.zeros(1, device=dev, dtype=torch.int32)

        pk = torch.zeros(6, BT, half, device=dev, dtype=fp8)
        pv = torch.zeros(6, BT, half, device=dev, dtype=fp8)
        y_fused = sconv_prefill_paged(
            x,
            weight,
            pk,
            pt,
            si,
            cu,
            pl,
            block_tokens=BT,
            col_pool2=pv,
            half_d=half,
        )
        pk2 = torch.zeros(6, BT, half, device=dev, dtype=fp8)
        pv2 = torch.zeros(6, BT, half, device=dev, dtype=fp8)
        ya = sconv_prefill_paged(
            x[:, :half], weight[:half], pk2, pt, si, cu, pl, block_tokens=BT
        )
        yb = sconv_prefill_paged(
            x[:, half:], weight[half:], pv2, pt, si, cu, pl, block_tokens=BT
        )
        assert torch.equal(y_fused, torch.cat([ya, yb], dim=1))
        assert torch.equal(pk.to(torch.bfloat16), pk2.to(torch.bfloat16))
        assert torch.equal(pv.to(torch.bfloat16), pv2.to(torch.bfloat16))
