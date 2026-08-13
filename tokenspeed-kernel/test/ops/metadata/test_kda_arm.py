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
"""``kda_arm_compose`` against the eager tensor chain it replaces.

The compose decides, per batch row, whether the previous verify's window may
fuse into this one. Ownership lives in a device slot table indexed by
request pool index; the compose must reproduce the gather/where chain bit
for bit (a wrong row silently replays another request's window), must
condemn gate-failed rows IN the table (the flush gates on it), and must not
block the host, because it runs at metadata prep while the previous step's
forward is still executing.
"""

import time

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.metadata import kda_arm_compose  # noqa: E402

DEV = "cuda"


def _reference(flat, table, rpis, steps, expect, anchor, commit, committed, G, base, t):
    """The eager chain: identity + causal gates, then per-group selects."""
    cap = flat.shape[1]
    bs = rpis.shape[0]
    state_in = flat[2 : 2 + G, :bs].clone()
    slot = table.to(torch.int64).gather(0, rpis.to(torch.int64))
    has = (slot >= 0) & (slot < steps.shape[0])
    safe = slot.clamp(min=0)
    ok = commit[0].gather(0, safe) == state_in[0].to(torch.int64)
    if committed is not None:
        ok = ok & (expect.gather(0, safe) == committed)
    condemned = has & ~ok
    fuse = has & ok
    table_out = table.clone()
    table_out[rpis.to(torch.int64)[condemned]] = -1
    neg = torch.full_like(slot, -1)
    out = flat.clone()
    out[0, :bs] = torch.where(fuse, base + safe * t, neg).to(torch.int32)
    out[1, :bs] = torch.where(
        fuse, steps.to(torch.int64).gather(0, safe), torch.zeros_like(slot)
    ).to(torch.int32)
    for g in range(G):
        out[2 + g, :bs] = torch.where(
            fuse, anchor[g].to(torch.int64).gather(0, safe), state_in[g].to(torch.int64)
        ).to(torch.int32)
        out[2 + G + g, :bs] = torch.where(fuse, commit[g].gather(0, safe), neg).to(
            torch.int32
        )
    return out, table_out


def _case(bs, pend_bs, G, seed, with_causal=True, cap=None, table_size=64):
    torch.manual_seed(seed)
    cap = cap or ((bs + 3) & ~3)
    flat = torch.full((2 + 2 * G, cap), -1, dtype=torch.int32, device=DEV)
    pages = torch.randint(1, 40, (G, pend_bs), device=DEV, dtype=torch.int32)
    # Each batch row is a distinct pool index; most own a pending slot, some
    # own none (-1), some slots have moved pages (recycled rpi -> impostor).
    rpis = torch.randperm(table_size, device=DEV)[:bs].to(torch.int64)
    table = torch.full((table_size,), -1, dtype=torch.int32, device=DEV)
    slot = torch.randint(-1, pend_bs, (bs,), device=DEV, dtype=torch.int64)
    owned = slot >= 0
    table[rpis[owned]] = slot[owned].to(torch.int32)
    safe = slot.clamp_min(0)
    for g in range(G):
        flat[2 + g, :bs] = pages[g].gather(0, safe)
    moved = torch.rand(bs, device=DEV) < 0.25
    flat[2, :bs] = torch.where(moved, flat[2, :bs] + 100, flat[2, :bs])
    steps = torch.randint(0, 4, (pend_bs,), device=DEV, dtype=torch.int32)
    expect = torch.randint(5, 50, (pend_bs,), device=DEV, dtype=torch.int64)
    committed = None
    if with_causal:
        committed = expect.gather(0, safe).clone()
        stale = torch.rand(bs, device=DEV) < 0.2
        committed[stale] += 1  # causal gate must reject these rows
    return (
        flat,
        table,
        rpis,
        steps,
        expect,
        pages.clone(),
        pages.to(torch.int64),
        committed,
    )


@pytest.mark.parametrize(
    "bs,pend_bs,G,causal",
    [
        (1, 1, 1, True),
        (2, 3, 3, True),
        (2, 3, 3, False),
        (32, 32, 3, True),
        (257, 300, 4, True),  # crosses the kernel's BLOCK
        (7, 4, 2, True),  # pending smaller than the batch
    ],
)
def test_matches_the_eager_chain_bitwise(bs, pend_bs, G, causal):
    flat, table, rpis, steps, expect, anchor, commit, committed = _case(
        bs, pend_bs, G, seed=bs + G, with_causal=causal, table_size=max(64, bs + 8)
    )
    base, t = 12, 3
    want, want_table = _reference(
        flat, table, rpis, steps, expect, anchor, commit, committed, G, base, t
    )
    kda_arm_compose(
        flat,
        table,
        rpis,
        steps,
        expect,
        anchor,
        commit,
        committed,
        flat[2 : 2 + G],
        base_offset=base,
        draft_token_num=t,
        num_groups=G,
    )
    torch.cuda.synchronize()
    assert torch.equal(flat, want)
    assert torch.equal(table, want_table)


def test_out_of_range_pool_index_degrades_to_no_fuse():
    """A pool index past the table must not gather or scatter past it."""
    G, table_size = 2, 8
    flat, table, rpis, steps, expect, anchor, commit, committed = _case(
        4, 4, G, seed=9, table_size=table_size
    )
    rpis = rpis.clone()
    rpis[1] = table_size + 100  # out of contract
    # Oracle: the same batch with row 1 pointing at a valid-but-unowned
    # index -- an out-of-range index must behave exactly like "owns nothing".
    unowned = (table == -1).nonzero()[0, 0]
    ref_rpis = rpis.clone()
    ref_rpis[1] = unowned
    want, want_table = _reference(
        flat, table, ref_rpis, steps, expect, anchor, commit, committed, G, 0, 2
    )
    # The kernel skips a dead lane outright; the arm's pre-fill already
    # neutralized it, so "untouched" is the contract, not "rewritten".
    want[:, 1] = flat[:, 1]
    kda_arm_compose(
        flat,
        table,
        rpis,
        steps,
        expect,
        anchor,
        commit,
        committed,
        flat[2 : 2 + G],
        base_offset=0,
        draft_token_num=2,
        num_groups=G,
    )
    torch.cuda.synchronize()
    assert int(flat[0, 1]) == -1, "out-of-range row must not fuse"
    assert torch.equal(flat, want)
    assert torch.equal(table, want_table)


def test_padded_tail_rows_are_untouched():
    """Rows beyond the live batch belong to the previous (larger) round and
    must keep the neutral values the fill already wrote."""
    bs, G = 3, 3
    flat, table, rpis, steps, expect, anchor, commit, committed = _case(
        bs, 4, G, seed=5
    )
    flat[:, bs:] = -7
    kda_arm_compose(
        flat,
        table,
        rpis,
        steps,
        expect,
        anchor,
        commit,
        committed,
        flat[2 : 2 + G],
        base_offset=0,
        draft_token_num=2,
        num_groups=G,
    )
    torch.cuda.synchronize()
    assert bool((flat[:, bs:] == -7).all())


def test_empty_batch_is_a_noop():
    G = 2
    flat = torch.full((2 + 2 * G, 4), -3, dtype=torch.int32, device=DEV)
    table = torch.full((8,), -1, dtype=torch.int32, device=DEV)
    empty = torch.zeros(0, dtype=torch.int64, device=DEV)
    kda_arm_compose(
        flat,
        table,
        empty,
        torch.zeros(1, dtype=torch.int32, device=DEV),
        torch.zeros(1, dtype=torch.int64, device=DEV),
        torch.zeros((G, 1), dtype=torch.int32, device=DEV),
        torch.zeros((G, 1), dtype=torch.int64, device=DEV),
        None,
        flat[2 : 2 + G],
        base_offset=0,
        draft_token_num=2,
        num_groups=G,
    )
    torch.cuda.synchronize()
    assert bool((flat == -3).all())


def _variant_count():
    """Compiled specializations of the compose kernel on this device."""
    from tokenspeed_kernel.ops.metadata.kda_arm import _kda_arm_compose_kernel

    return sum(len(c[0]) for c in _kda_arm_compose_kernel.device_caches.values())


def test_one_compiled_variant_across_rounds():
    """Every round must reuse one compiled kernel.

    Triton keys its variants on scalar buckets and pointer alignment, and the
    module load behind a fresh variant implicitly synchronizes the device --
    at metadata prep that stalls the event loop for the whole queued forward.
    The caller pads its buffers so every row slice stays 16B-aligned; this
    pins that contract together with the scalars being non-specializing."""
    G, cap = 3, 32
    flat, table, rpis_full, steps, expect, anchor, commit, committed = _case(
        8, 32, G, seed=3, cap=cap, table_size=64
    )
    rpis_buf = torch.zeros(cap, dtype=torch.int64, device=DEV)
    rpis_buf[: rpis_full.shape[0]] = rpis_full

    def compose(bs, base):
        kda_arm_compose(
            flat,
            table,
            rpis_buf[:bs],
            steps,
            expect,
            anchor,
            commit,
            committed[:bs] if committed is not None else None,
            flat[2 : 2 + G],
            base_offset=base,
            draft_token_num=2,
            num_groups=G,
        )

    compose(8, 0)
    torch.cuda.synchronize()
    warm = _variant_count()
    for bs in (1, 4, 8, 17, 32):
        for base in (0, 64, 96):
            compose(bs, base)
    torch.cuda.synchronize()
    assert _variant_count() == warm, (
        f"compose compiled {_variant_count() - warm} extra variants; each one "
        "loads a module and synchronizes the device mid-decode"
    )


def test_does_not_block_the_host_on_a_busy_stream():
    """With the variant warm, the launch must return while the stream is
    still saturated -- metadata prep runs behind the previous forward."""
    G, cap = 3, 32
    flat, table, rpis, steps, expect, anchor, commit, committed = _case(
        8, 32, G, seed=4, cap=cap, table_size=64
    )
    args = (flat, table, rpis, steps, expect, anchor, commit, committed)
    kda_arm_compose(
        *args, flat[2 : 2 + G], base_offset=0, draft_token_num=2, num_groups=G
    )
    torch.cuda.synchronize()
    torch.cuda._sleep(400_000_000)
    t0 = time.perf_counter()
    kda_arm_compose(
        *args, flat[2 : 2 + G], base_offset=64, draft_token_num=2, num_groups=G
    )
    blocked = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize()
    assert blocked < 20, f"compose blocked the host for {blocked:.1f}ms"
