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
"""Compose the KDA lazy-commit control buffers for one verify round.

The fused verify reads four per-slot control rows -- payload base, replayed
step count, anchor page, commit page. Built eagerly that is a
gather/where/clamp chain per state group (~45 launches of a few hundred
bytes each); this is the same arithmetic in one launch against the
persistent buffers.

Ownership comes from a device-resident slot table indexed by request pool
index: ``table[rpi]`` is the pending payload slot that request owns (``-1``
= none). The table is the single source of truth for every device-side
write decision -- the compose reads it to arm the replay, and CLEARS the
row itself when a request fails the identity or causal gate, so the
standalone flush (which gates on the same table) can never write a window
the compose already condemned. The host never reads it back.

The compose is metadata prep, so it runs outside CUDA-graph capture and
takes its scalars as kernel arguments rather than device tensors.
"""

from tokenspeed_kernel._triton import tl, triton


# Every scalar here varies per round (batch size, parity half, window size).
# Left to specialize, Triton would JIT a fresh variant per divisible-by-16 or
# ==1 bucket and stall the event loop mid-decode to compile it.
@triton.jit(
    do_not_specialize=["base_offset", "t_prev", "cap", "bs", "pend_bs", "table_size"],
)
def _kda_arm_compose_kernel(
    flat_ptr,
    table_ptr,
    rpi_ptr,
    steps_ptr,
    expect_ptr,
    anchor_ptr,
    commit_ptr,
    committed_ptr,
    state_in_ptr,
    base_offset,
    t_prev,
    cap,
    bs,
    pend_bs,
    table_size,
    G: tl.constexpr,
    HAS_COMMITTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offs < bs
    rpi = tl.load(rpi_ptr + offs, mask=live, other=0).to(tl.int64)
    # An out-of-range pool index degrades to no-fuse instead of touching
    # memory past the table (a Triton gather has no bounds of its own).
    live = live & (rpi >= 0) & (rpi < table_size)
    slot = tl.load(table_ptr + rpi, mask=live, other=-1).to(tl.int64)
    # A slot past the record's width is equally out of contract; same fate.
    has = (slot >= 0) & (slot < pend_bs)
    safe = tl.maximum(slot, 0)

    # Identity gate: the pending's commit page must still be the page this
    # round reads. Group 0 is representative -- one recycled rpi moves every
    # group's page together.
    ok = tl.load(commit_ptr + safe, mask=live & has, other=0) == tl.load(
        state_in_ptr + offs, mask=live & has, other=-1
    ).to(tl.int64)
    if HAS_COMMITTED:
        # Causal gate: the committed position must be what the record expected.
        ok = ok & (
            tl.load(expect_ptr + safe, mask=live & has, other=0)
            == tl.load(committed_ptr + offs, mask=live & has, other=-1)
        )
    # A condemned row is cleared at the source: the flush gates its writes on
    # this same table, so the window can never land anywhere afterwards.
    tl.store(table_ptr + rpi, -1, mask=live & has & ~ok)
    fuse = has & ok

    tl.store(
        flat_ptr + offs,
        tl.where(fuse, base_offset + safe * t_prev, -1).to(tl.int32),
        mask=live,
    )
    tl.store(
        flat_ptr + cap + offs,
        tl.where(fuse, tl.load(steps_ptr + safe, mask=live & fuse, other=0), 0),
        mask=live,
    )
    for g in tl.static_range(G):
        anchor = tl.load(anchor_ptr + g * pend_bs + safe, mask=live & fuse, other=0)
        commit = tl.load(commit_ptr + g * pend_bs + safe, mask=live & fuse, other=0)
        # Rows without a fusable pending keep this round's committed page as
        # the anchor and skip the commit (-1).
        stale = tl.load(state_in_ptr + g * cap + offs, mask=live, other=-1)
        tl.store(
            flat_ptr + (2 + g) * cap + offs,
            tl.where(fuse, anchor.to(tl.int32), stale),
            mask=live,
        )
        tl.store(
            flat_ptr + (2 + G + g) * cap + offs,
            tl.where(fuse, commit.to(tl.int32), -1),
            mask=live,
        )


def kda_arm_compose(
    flat,
    slot_table,
    rpis,
    pending_steps,
    pending_expect,
    pending_anchor,
    pending_commit,
    committed,
    state_in,
    *,
    base_offset: int,
    draft_token_num: int,
    num_groups: int,
) -> None:
    """Write one verify round's lazy-commit control rows in a single launch.

    Args:
        flat: ``[2 + 2G, cap]`` int32 control storage. Row 0 takes the payload
            base row (``-1`` = no pending), row 1 the replayed step count,
            rows ``2..2+G`` the anchor page per state group, and the rest the
            commit page per group (``-1`` skips the commit).
        slot_table: ``[req_pool_size]`` int32 pending-slot ownership, indexed
            by request pool index (``-1`` = no pending). Read to arm each
            row; written back (``-1``) for rows whose identity or causal
            gate fails, condemning them for every later consumer.
        rpis: ``[bs]`` integer request pool index of each batch row (device).
        pending_steps: ``[pend_bs]`` int32 accepted length recorded per slot.
        pending_expect: ``[pend_bs]`` int64 committed position the record
            anticipated for the next round.
        pending_anchor: ``[G, pend_bs]`` int32 pre-window page per slot.
        pending_commit: ``[G, pend_bs]`` int64 destination page per slot.
        committed: ``[bs]`` int64 committed position of each batch row, or
            ``None`` to skip the causal gate.
        state_in: ``[G, cap]`` int32 committed page of each batch row this
            round -- the anchor fallback and the identity reference.
        base_offset: first payload ring row of the pending's parity half.
        draft_token_num: payload rows per slot (the window size ``T``).
        num_groups: number of state groups ``G``.

    Returns:
        None. ``flat`` and (condemned rows of) ``slot_table`` are written in
        place.
    """
    bs = rpis.shape[0]
    if bs == 0:
        return
    cap = flat.shape[1]
    block = 256
    _kda_arm_compose_kernel[(triton.cdiv(bs, block),)](
        flat,
        slot_table,
        rpis,
        pending_steps,
        pending_expect,
        pending_anchor,
        pending_commit,
        committed if committed is not None else rpis,
        state_in,
        base_offset=base_offset,
        t_prev=draft_token_num,
        cap=cap,
        bs=bs,
        pend_bs=pending_steps.shape[0],
        table_size=slot_table.shape[0],
        G=num_groups,
        HAS_COMMITTED=committed is not None,
        BLOCK=block,
        num_warps=4,
    )
