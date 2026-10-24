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
from tokenspeed_kernel.numerics.torch.conv import (
    torch_block_grouped_conv,
    torch_short_conv,
)
from tokenspeed_kernel.ops.conv import (
    PAD_SLOT_ID,
    dflash2_grouped_conv,
    inkling_commit_tables,
    inkling_ring_commit,
    inkling_ring_sconv,
    seq_idx_from_cu_seqlens,
)
from utils import assert_no_triton_compile

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
    return torch_short_conv(x, weight, prefix, use_residual)


def seed_ring_prefix(ring, slot, pre_len, prefix):
    """Place the last W-1 pre-chunk rows at their positions' ring rows."""
    for j in range(W - 1):
        pos = pre_len - (W - 1) + j
        if pos >= 0:
            ring[slot, pos % ring.shape[1]] = prefix[j]


def ring_rows_at(ring, slot, positions):
    return torch.stack([ring[slot, p % ring.shape[1]] for p in positions])


def inert_publish(B: int, D: int, device: str):
    """Publish plumbing that never fires: a hole-only two-slot table."""
    table = torch.zeros(B, 2, dtype=torch.int32, device=device)
    ckpt = torch.zeros(1, W - 1, D, dtype=DTYPE, device=device)
    return table, ckpt


def make_ckpt(
    B: int, D: int, device: str, *, num_pages: int = 8, dtype: torch.dtype = DTYPE
):
    """A checkpoint field plus an all-hole two-slot table for it."""
    table = torch.zeros(B, 2, dtype=torch.int32, device=device)
    ckpt = torch.randn(num_pages, W - 1, D, device=device).to(dtype)
    return table, ckpt


def decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    table: torch.Tensor | None = None,
    ckpt_a: torch.Tensor | None = None,
    ckpt_b: torch.Tensor | None = None,
    publish_live: bool = False,
) -> torch.Tensor:
    """T=1-per-request call of the decode kernel (inert publish by default)."""
    B = x.shape[0]
    device = x.device
    if table is None:
        table, ckpt_a = inert_publish(B, x.shape[1], str(device))
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
        table,
        ckpt_a,
        ckpt_b,
        num_extends=0,
        publish_live=publish_live,
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


def _window(x_full: torch.Tensor, prefix: torch.Tensor, through_in_x: int):
    """The W-1 rows ending at chunk row ``through_in_x`` over [prefix ++ x]."""
    return torch.cat([prefix, x_full])[through_in_x : through_in_x + W - 1]


@pytest.mark.parametrize("D", [2048, 6144])
@pytest.mark.parametrize("use_residual", [True, False])
def test_sconv_prefill_varlen(D: int, use_residual: bool, device: str) -> None:
    """Prefill taps read the input slot's checkpoint (zeros when cold); the
    last min(chunk_len + W-1, R) positions persist to the ring in one launch
    (no epilogue), and each request publishes its chunk-end window to its
    live slot."""
    lens = [3, 850, 1]
    pre_lens = [131, 0, 257]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, seed=0
    )
    cache_indices = torch.tensor([2, 5, PAD_SLOT_ID], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, False, True], device=device)
    seq_lens = torch.tensor(
        [p + n for p, n in zip(pre_lens, lens)], dtype=torch.int32, device=device
    )
    table, ckpt = make_ckpt(len(lens), D, device, num_pages=12)
    # req0 resumes from the checkpoint in page 3 and publishes into page 4;
    # req1 is cold and publishes into page 5; the PAD row's live slot is
    # never written.
    table[0] = torch.tensor([3, 4], device=device)
    table[1] = torch.tensor([0, 5], device=device)
    table[2] = torch.tensor([6, 7], device=device)
    prefix0 = ckpt[3].to(DTYPE)
    page7 = ckpt[7].clone()
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
        table,
        ckpt,
        None,
        num_extends=3,
        publish_live=True,
        use_residual=use_residual,
    )

    zeros = torch.zeros(W - 1, D, device=device, dtype=DTYPE)
    cu = cu_seqlens.tolist()
    for i, prefix in enumerate((prefix0, zeros, zeros)):
        s, e = cu[i], cu[i + 1]
        ref = ref_sconv(x[s:e], weight, prefix, use_residual=use_residual)
        torch.testing.assert_close(y[s:e], ref, atol=ATOL, rtol=RTOL)

    # Ring: req0 gets its 3 chunk rows plus the restored window (3 + 3 <= R);
    # req1 gets its last R rows in-kernel; the PAD row leaves the ring alone.
    expected = cache_snapshot.clone()
    for j in range(W - 1):
        expected[2, (pre_lens[0] - (W - 1) + j) % R] = prefix0[j]
    for j in range(lens[0]):
        expected[2, (pre_lens[0] + j) % R] = x[cu[0] + j]
    for j in range(R):
        pos = int(seq_lens[1]) - R + j
        expected[5, pos % R] = x[cu[2] - R + j]
    assert torch.equal(conv_cache, expected)

    # Publication: the chunk-end window of req0 and req1 landed in their
    # live slots; the input checkpoint and the PAD row's slot are untouched.
    torch.testing.assert_close(
        ckpt[4].to(DTYPE), _window(x[cu[0] : cu[1]], prefix0, lens[0]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        ckpt[5].to(DTYPE), x[cu[2] - (W - 1) : cu[2]], atol=0, rtol=0
    )
    assert torch.equal(ckpt[3].to(DTYPE), prefix0)
    assert torch.equal(ckpt[7], page7)


def test_prefill_ring_write_gate(device: str) -> None:
    """The pos >= through - R gate: chunk rows and the restored window get
    exactly one writer each; positions older than R never land."""
    D = 2048
    lens = [850, 4, 7, 5]
    pre_lens = [129, 130, 131, 0]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, num_slots=10, seed=1
    )
    cache_indices = torch.tensor(
        [2, 5, 7, PAD_SLOT_ID], dtype=torch.int32, device=device
    )
    has_initial_state = torch.tensor([True, True, True, False], device=device)
    seq_lens = torch.tensor(
        [p + n for p, n in zip(pre_lens, lens)], dtype=torch.int32, device=device
    )
    table, ckpt = make_ckpt(len(lens), D, device)
    for i in range(3):
        table[i, 0] = i + 1
    old = conv_cache.clone()

    inkling_ring_sconv(
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        seq_lens,
        table,
        ckpt,
        None,
        num_extends=4,
        publish_live=False,
    )

    expected = old.clone()
    cu = cu_seqlens.tolist()
    for i in range(3):
        slot = int(cache_indices[i])
        through = int(seq_lens[i])
        # Restored window rows within the last R positions.
        for j in range(W - 1):
            pos = pre_lens[i] - (W - 1) + j
            if pos >= through - R:
                expected[slot, pos % R] = ckpt[i + 1, j].to(DTYPE)
        # Chunk rows within the last R positions.
        for j in range(lens[i]):
            pos = pre_lens[i] + j
            if pos >= through - R:
                expected[slot, pos % R] = x[cu[i] + j]
    assert torch.equal(conv_cache, expected)


def test_prefill_small_ring_gate(device: str) -> None:
    """Non-spec ring (R=4): a 2-token warm chunk keeps 2 chunk rows plus the
    2 newest restored rows -- exactly min(chunk_len + W-1, R) positions."""
    D = 2048
    small_r = W
    lens = [2]
    pre_len = 133
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, ring=small_r, seed=2
    )
    cache_indices = torch.tensor([3], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True], device=device)
    seq_lens = torch.tensor([pre_len + lens[0]], dtype=torch.int32, device=device)
    table, ckpt = make_ckpt(1, D, device)
    table[0, 0] = 2
    old = conv_cache.clone()

    inkling_ring_sconv(
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        seq_lens,
        table,
        ckpt,
        None,
        num_extends=1,
        publish_live=False,
    )

    expected = old.clone()
    through = pre_len + lens[0]
    for j in range(W - 1):
        pos = pre_len - (W - 1) + j
        if pos >= through - small_r:
            expected[3, pos % small_r] = ckpt[2, j].to(DTYPE)
    expected[3, (pre_len + 0) % small_r] = x[0]
    expected[3, (pre_len + 1) % small_r] = x[1]
    assert torch.equal(conv_cache, expected)


def test_sconv_prefill_then_lookback_window(device: str) -> None:
    """Round-1 draft lookback after a long prefill: the window starts at
    ``through - lookback`` and its first taps read down to ``through - 5`` --
    the prefill kernel's gated write must have persisted those rows."""
    D = 2048
    lb, k = 2, 4
    prefill_len = 850
    torch.manual_seed(7)
    x_full = torch.randn(prefill_len + k, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    conv_cache = torch.randn(4, R, D, device=device, dtype=DTYPE)
    cache_indices = torch.tensor([1], dtype=torch.int32, device=device)
    table, ckpt = inert_publish(1, D, device)

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
        table,
        ckpt,
        None,
        num_extends=1,
        publish_live=True,
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
        table,
        ckpt,
        None,
        num_extends=0,
        publish_live=False,
    )

    ref = ref_sconv(x_win, weight, x_full[start - (W - 1) : start])
    torch.testing.assert_close(y_win, ref, atol=ATOL, rtol=RTOL)


def test_prefill_checkpoint_taps_and_fp8(device: str) -> None:
    """Tap sources per request: a prefix with an input checkpoint reads the
    window; a hole slot or a cold request taps zeros; a PAD row is inert. A
    single fp8 field casts into the compute dtype."""
    D = 8
    lens = [2, 2, 2, 2]
    pre_lens = [131, 257, 0, 129]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, seed=3
    )
    cache_indices = torch.tensor(
        [1, 2, 4, PAD_SLOT_ID], dtype=torch.int32, device=device
    )
    has_initial_state = torch.tensor([True, True, False, True], device=device)
    seq_lens = torch.tensor(
        [p + n for p, n in zip(pre_lens, lens)], dtype=torch.int32, device=device
    )
    table, ckpt = make_ckpt(len(lens), D, device, dtype=torch.float8_e5m2)
    table[0, 0] = 3  # req0: a real input checkpoint
    # req1: a prefix but a hole in the input slot -> zeros
    y = inkling_ring_sconv(
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        seq_lens,
        table,
        ckpt,
        None,
        num_extends=4,
        publish_live=False,
    )

    zeros = torch.zeros(W - 1, D, device=device, dtype=DTYPE)
    prefixes = [ckpt[3].to(DTYPE), zeros, zeros, zeros]
    cu = cu_seqlens.tolist()
    for i in range(len(lens)):
        s, e = cu[i], cu[i + 1]
        ref = ref_sconv(x[s:e], weight, prefixes[i])
        torch.testing.assert_close(y[s:e], ref, atol=ATOL, rtol=RTOL)


def test_prefill_publishes_a_short_chunk_from_the_checkpoint(device: str) -> None:
    """A chunk shorter than W-1 tokens publishes a window that reaches back
    into its input checkpoint; a cold short chunk reaches into zeros."""
    D = 8
    lens = [1, 2]
    pre_lens = [131, 0]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, seed=11
    )
    cache_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True, False], device=device)
    seq_lens = torch.tensor(
        [p + n for p, n in zip(pre_lens, lens)], dtype=torch.int32, device=device
    )
    table, ckpt = make_ckpt(len(lens), D, device)
    table[0] = torch.tensor([3, 4], device=device)
    table[1] = torch.tensor([0, 5], device=device)
    checkpoint = ckpt[3].clone()

    inkling_ring_sconv(
        x,
        weight,
        conv_cache,
        cu_seqlens,
        seq_idx,
        cache_indices,
        has_initial_state,
        seq_lens,
        table,
        ckpt,
        None,
        num_extends=2,
        publish_live=True,
    )

    cu = cu_seqlens.tolist()
    zeros = torch.zeros(W - 1, D, device=device, dtype=DTYPE)
    torch.testing.assert_close(
        ckpt[4], _window(x[cu[0] : cu[1]], checkpoint, lens[0]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        ckpt[5], _window(x[cu[1] : cu[2]], zeros, lens[1]), atol=0, rtol=0
    )
    assert torch.equal(ckpt[3], checkpoint)


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

    y = decode(x, weight, conv_cache, cache_indices, seq_lens)

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
    reads only accepted history. Verify publishes nothing in-kernel."""
    torch.manual_seed(7)
    D, K, B = 512, 4, 2
    committed_len = 20
    steps = 3
    x_all = torch.randn(B, committed_len + K * (steps + 1), D, device=device).to(DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    conv_cache = torch.zeros(6, R, D, device=device, dtype=DTYPE)
    cache_indices = torch.tensor([1, 4], dtype=torch.int32, device=device)
    table, ckpt = make_ckpt(B, D, device)
    table[:, 1] = torch.tensor([2, 3], device=device)
    ckpt_before = ckpt.clone()
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
            xs,
            weight,
            conv_cache,
            qsl,
            seq_idx,
            cache_indices,
            ones,
            seq_lens,
            table,
            ckpt,
            None,
            num_extends=0,
            publish_live=False,
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
    assert torch.equal(ckpt, ckpt_before)


def test_sconv_chained_prefill_decode(device: str) -> None:
    """Full prefill == partial prefill + 3 decode steps (no epilogue: the
    prefill kernel persists its own tail), and the live slot ends up holding
    the same final window either way."""
    D = 2048
    num_decode = 3
    lens = [12, 16]
    x, weight, conv_cache, cu_seqlens, seq_idx = _make_prefill_inputs(
        lens, D, device, seed=5
    )
    cache_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([False, False], device=device)
    full_seq_lens = cu_seqlens.diff().to(torch.int32)
    table, ckpt = make_ckpt(len(lens), D, device)
    table[:, 1] = torch.tensor([2, 3], device=device)

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
        table,
        ckpt,
        None,
        num_extends=2,
        publish_live=True,
    )
    full_windows = ckpt[2:4].clone()

    # Chained: prefill over lens - 3 tokens (persists in-kernel), then 3
    # decode steps over the ring, each publishing its own window.
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
        table,
        ckpt,
        None,
        num_extends=2,
        publish_live=True,
    )
    for i in range(len(lens)):
        torch.testing.assert_close(
            ckpt[2 + i],
            x[cu[i] + part_lens[i] - (W - 1) : cu[i] + part_lens[i]],
            atol=0,
            rtol=0,
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
            decode(
                x_step,
                weight,
                conv_cache,
                cache_indices,
                step_seq_lens,
                table=table,
                ckpt_a=ckpt,
                publish_live=True,
            )
        )

    cu_p = cu_part.tolist()
    for i in range(len(lens)):
        s = cu[i]
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
    assert torch.equal(ckpt[2:4], full_windows)


def test_sconv_channel_sliced_cache_view(device: str) -> None:
    """Both kernels must work on a channel-sliced view of a wider ring."""
    torch.manual_seed(6)
    D, off = 2048, 64
    D_total = D + 3 * off
    num_slots = 8
    lens = [5, 2]
    pre_lens = [131, 257]
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
    table, ckpt = make_ckpt(B, D, device)
    table[0, 0] = 5
    table[1, 0] = 6
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
        table,
        ckpt,
        None,
        num_extends=2,
        publish_live=False,
    )
    cu = cu_seqlens.tolist()
    for i in range(B):
        s, e = cu[i], cu[i + 1]
        ref = ref_sconv(x[s:e], weight, ckpt[5 + i].to(DTYPE))
        torch.testing.assert_close(y[s:e], ref, atol=ATOL, rtol=RTOL)

    # Chunk rows and restored windows persist in-slice only.
    expected_view = snapshot[:, :, off : off + D].clone()
    for i in range(B):
        ci = int(cache_indices[i])
        through = int(seq_lens[i])
        for j in range(W - 1):
            pos = pre_lens[i] - (W - 1) + j
            if pos >= through - R:
                expected_view[ci, pos % R] = ckpt[5 + i, j].to(DTYPE)
        for j in range(lens[i]):
            expected_view[ci, (pre_lens[i] + j) % R] = x[cu[i] + j]
    assert torch.equal(cache_view, expected_view)
    assert torch.equal(cache_full[:, :, outside], snapshot[:, :, outside])


def test_decode_publish_window_and_split(device: str) -> None:
    """The decode kernel publishes each request's chunk-end window to its
    live slot: window rows may borrow ring history, channels split across
    two fields, and a hole live slot publishes nothing."""
    torch.manual_seed(8)
    D, wa = 8, 5
    B, K = 2, 4
    conv_cache = torch.randn(6, R, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    cache_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([130, 260 + 3], dtype=torch.int32, device=device)
    qsl = torch.arange(0, B * K + 1, K, dtype=torch.int32, device=device)
    x = torch.randn(B * K, D, device=device, dtype=DTYPE)
    ckpt_a = torch.zeros(8, W - 1, wa, dtype=DTYPE, device=device)
    ckpt_b = torch.zeros(8, W - 1, D - wa, dtype=DTYPE, device=device)
    table = torch.zeros(B, 2, dtype=torch.int32, device=device)
    table[0, 1] = 7  # req0's live slot; req1's is a hole

    inkling_ring_sconv(
        x,
        weight,
        conv_cache,
        qsl,
        seq_idx_from_cu_seqlens(qsl, B * K),
        cache_indices,
        torch.ones(B, dtype=torch.bool, device=device),
        seq_lens,
        table,
        ckpt_a,
        ckpt_b,
        num_extends=0,
        publish_live=True,
    )

    # req0: chunk positions 126..129; window ending at 130 = chunk rows 1..3.
    window = x[1:4]
    torch.testing.assert_close(ckpt_a[7], window[:, :wa], atol=0, rtol=0)
    torch.testing.assert_close(ckpt_b[7], window[:, wa:], atol=0, rtol=0)
    assert torch.equal(ckpt_a[:7], torch.zeros_like(ckpt_a[:7]))
    assert torch.equal(ckpt_b[:7], torch.zeros_like(ckpt_b[:7]))

    # A one-token decode borrows W-2 ring rows for its window: req1's verify
    # rows at positions 261 and 262 (x[6], x[7]) plus the new token.
    table[1, 1] = 6
    x1 = torch.randn(B, D, device=device, dtype=DTYPE)
    step_lens = torch.tensor([131, 264], dtype=torch.int32, device=device)
    decode(
        x1,
        weight,
        conv_cache,
        cache_indices,
        step_lens,
        table=table,
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        publish_live=True,
    )
    assert torch.equal(ring_rows_at(conv_cache, 2, [261, 262]), x[6:8])
    window1 = torch.cat([x[6:8], x1[1:2]])
    torch.testing.assert_close(ckpt_a[6], window1[:, :wa], atol=0, rtol=0)
    torch.testing.assert_close(ckpt_b[6], window1[:, wa:], atol=0, rtol=0)


def ref_ring_commit(ring, streams, live_pages, cache_indices, seq_lens, accepted, k):
    """Per-stream torch publication of each request's accepted window."""
    bs = accepted.shape[0]
    frontier = seq_lens.long() - k + accepted.long().clamp(1, k)
    slots = cache_indices.long()
    for layer, offset, group, fields in streams:
        pages = live_pages[group, :bs].long()
        dim = sum(field.shape[-1] for field in fields)
        state = ring[layer][:, :, offset : offset + dim]
        rows = fields[0].shape[1]
        pos = frontier.view(bs, 1) - rows + torch.arange(rows, device=ring.device)
        window = state[slots.clamp_min(0).view(bs, 1), pos.remainder(state.shape[1])]
        window = torch.where((pos >= 0).unsqueeze(-1), window, torch.zeros_like(window))
        parts = window.split([field.shape[-1] for field in fields], dim=-1)
        for field, values in zip(fields, parts):
            for b in range(bs):
                if slots[b] >= 0 and pages[b] > 0:
                    field[pages[b]] = values[b].to(field.dtype)


def test_ring_commit_publishes_every_streams_accepted_window(device: str) -> None:
    """One commit writes the accepted window of every layer's streams: fields
    split a stream's channels, a second field dtype takes its own table, ring
    rows wrap, positions before 0 are zeros, and padded rows and hole live
    blocks write nothing."""
    from tokenspeed_kernel.ops.conv.triton import _inkling_ring_commit_kernel

    torch.manual_seed(9)
    layers, k, pages = 3, 4, 10
    ring = torch.randn(layers, 6, R, 24, device=device, dtype=DTYPE)
    # Page-strided per-layer views, as the cache arena lays fields out.
    k_planes = torch.full((pages, layers, W - 1, 5), 7.0, device=device, dtype=DTYPE)
    v_planes = torch.full((pages, layers, W - 1, 3), 7.0, device=device, dtype=DTYPE)
    hidden = torch.full((pages, layers, W - 1, 16), 7.0, device=device).to(
        torch.float8_e5m2
    )
    expected = [t.clone() for t in (k_planes, v_planes, hidden)]

    def streams(k_planes, v_planes, hidden):
        return [
            entry
            for layer in range(layers)
            for entry in (
                (layer, 0, 0, (k_planes[:, layer], v_planes[:, layer])),
                (layer, 8, 1, (hidden[:, layer],)),
            )
        ]

    cache_indices = torch.tensor(
        [1, PAD_SLOT_ID, 3, 4, 5], device=device, dtype=torch.int32
    )
    seq_lens = torch.tensor([130, 50, 6, 2, 263], device=device, dtype=torch.int32)
    accepted = torch.tensor([2, 4, 0, 4, 9], device=device, dtype=torch.int32)
    live_pages = torch.tensor(
        [[7, 3, 0, 2, 9], [6, 8, 5, 0, 4]], device=device, dtype=torch.int32
    )

    tables = inkling_commit_tables(ring, streams(k_planes, v_planes, hidden))
    assert [table.dtype for table in tables] == [DTYPE, torch.float8_e5m2]
    inkling_ring_commit(
        ring, tables, live_pages, cache_indices, seq_lens, accepted, verify_width=k
    )
    ref_ring_commit(
        ring, streams(*expected), live_pages, cache_indices, seq_lens, accepted, k
    )
    assert torch.equal(k_planes, expected[0])
    assert torch.equal(v_planes, expected[1])
    assert torch.equal(hidden.view(torch.uint8), expected[2].view(torch.uint8))
    # Every written page changed; the others kept the sentinel.
    assert not torch.equal(k_planes[7], torch.full_like(k_planes[7], 7.0))
    assert torch.equal(k_planes[3], torch.full_like(k_planes[3], 7.0))
    # The kernel indexes request vectors by row, so a strided one is refused.
    with pytest.raises(ValueError, match="contiguous"):
        inkling_ring_commit(
            ring,
            tables,
            live_pages,
            cache_indices,
            seq_lens,
            accepted.repeat_interleave(2)[::2],
            verify_width=k,
        )

    with assert_no_triton_compile(_inkling_ring_commit_kernel):
        for bs in (1, 2, 16):
            inkling_ring_commit(
                ring,
                tables,
                torch.zeros(2, bs, device=device, dtype=torch.int32),
                torch.zeros(bs, device=device, dtype=torch.int32),
                torch.full((bs,), 40, device=device, dtype=torch.int32),
                torch.ones(bs, device=device, dtype=torch.int32),
                verify_width=k,
            )


def test_prefill_matches_decode_on_short_warm_chunks(device: str) -> None:
    """Cross-kernel consistency: a short warm extend served by the prefill
    kernel (checkpoint taps) must produce the same output as the decode
    kernel over a ring seeded with the same window."""
    torch.manual_seed(9)
    D, B, k = 512, 2, 4
    pre_lens = [131, 389]
    x = torch.randn(B * k, D, device=device, dtype=DTYPE)
    weight = torch.randn(D, W, device=device, dtype=DTYPE) * 0.5
    qsl = torch.arange(0, B * k + 1, k, dtype=torch.int32, device=device)
    seq_idx = seq_idx_from_cu_seqlens(qsl, B * k)
    cache_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    ones = torch.ones(B, dtype=torch.bool, device=device)
    seq_lens = torch.tensor([p + k for p in pre_lens], dtype=torch.int32, device=device)

    table, ckpt = make_ckpt(B, D, device)
    table[0, 0] = 2
    table[1, 0] = 3

    ring_a = torch.randn(4, R, D, device=device, dtype=DTYPE)
    y_prefill = inkling_ring_sconv(
        x,
        weight,
        ring_a,
        qsl,
        seq_idx,
        cache_indices,
        ones,
        seq_lens,
        table,
        ckpt,
        None,
        num_extends=2,
        publish_live=False,
    )

    ring_b = torch.randn(4, R, D, device=device, dtype=DTYPE)
    for b in range(B):
        seed_ring_prefix(
            ring_b, int(cache_indices[b]), pre_lens[b], ckpt[2 + b].to(DTYPE)
        )
    inert_table, inert_ckpt = inert_publish(B, D, device)
    y_decode = inkling_ring_sconv(
        x,
        weight,
        ring_b,
        qsl,
        seq_idx,
        cache_indices,
        ones,
        seq_lens,
        inert_table,
        inert_ckpt,
        None,
        num_extends=0,
        publish_live=False,
    )
    torch.testing.assert_close(y_prefill, y_decode, atol=0, rtol=0)


@pytest.mark.parametrize("block_size,taps", [(8, 2), (6, 3), (8, 1)])
def test_dflash2_grouped_conv_matches_reference(
    block_size: int, taps: int, device: str
) -> None:
    torch.manual_seed(0)
    rows, num_groups, group_size = 3 * block_size, 5, 16
    channels = num_groups * group_size
    x = torch.randn(rows, channels, device=device, dtype=DTYPE)
    # Two sides of the projection share one buffer, so delta is a strided slice.
    projection = torch.randn(rows, 2, taps, num_groups, device=device, dtype=DTYPE)
    base = torch.randn(2, taps, channels, device=device, dtype=DTYPE)

    for side in (0, 1):
        y = dflash2_grouped_conv(
            x, projection[:, side], base[side], block_size, group_size
        )
        expected = torch_block_grouped_conv(
            x, projection[:, side], base[side], block_size, group_size
        )
        torch.testing.assert_close(y, expected, atol=ATOL, rtol=RTOL)
