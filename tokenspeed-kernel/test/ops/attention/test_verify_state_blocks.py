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

"""The fused committed-state resolve must match the torch chain exactly.

The page it returns is where verify reads a request's recurrent state. A wrong
page silently feeds another request's state into this one, so the comparison is
integer equality.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.triton.verify_state_blocks import (  # noqa: E402
    verify_state_blocks,
)


def _reference(seq_lens, table, *, batch_size, draft_tokens, granularity):
    """The elementwise chain the kernel replaces, kept verbatim."""
    committed = (seq_lens[:batch_size].to(torch.int64) - draft_tokens).clamp_min(0)
    in_slots = torch.div(
        (committed - 1).clamp_min(0), granularity, rounding_mode="floor"
    )
    has_history = committed > 0
    slots_safe = in_slots.clamp(min=0, max=table.shape[1] - 1)
    pages = table[:batch_size].gather(1, slots_safe.unsqueeze(1)).squeeze(1)
    pages = torch.where(has_history, pages, torch.full_like(pages, -1)).to(torch.int32)
    return pages, committed


def _run(seq_lens, table, *, batch_size, draft_tokens, granularity):
    pages = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    committed = torch.empty(batch_size, dtype=torch.int64, device="cuda")
    verify_state_blocks(
        seq_lens,
        table,
        batch_size=batch_size,
        draft_tokens=draft_tokens,
        granularity=granularity,
        pages_out=pages,
        committed_out=committed,
    )
    return pages, committed


@pytest.mark.parametrize(
    ("batch_size", "draft_tokens"), [(1, 1), (1, 4), (3, 4), (8, 2), (32, 4), (512, 4)]
)
@pytest.mark.parametrize("granularity", [1, 64])
def test_matches_the_torch_chain(batch_size, draft_tokens, granularity):
    g = torch.Generator(device="cuda").manual_seed(batch_size + draft_tokens)
    seq_lens = torch.randint(
        0, 4096, (batch_size,), device="cuda", dtype=torch.int32, generator=g
    )
    table = torch.randint(
        0, 900, (batch_size, 24), device="cuda", dtype=torch.int32, generator=g
    )
    got_p, got_c = _run(
        seq_lens,
        table,
        batch_size=batch_size,
        draft_tokens=draft_tokens,
        granularity=granularity,
    )
    want_p, want_c = _reference(
        seq_lens,
        table,
        batch_size=batch_size,
        draft_tokens=draft_tokens,
        granularity=granularity,
    )
    assert torch.equal(got_p, want_p)
    assert torch.equal(got_c, want_c)


def test_a_request_without_committed_history_gets_no_page():
    """seq_len <= draft means nothing is committed yet; verify must not read a
    state page for it."""
    seq_lens = torch.tensor([0, 1, 4, 5], device="cuda", dtype=torch.int32)
    table = torch.randint(1, 900, (4, 24), device="cuda", dtype=torch.int32)
    got_p, got_c = _run(seq_lens, table, batch_size=4, draft_tokens=4, granularity=64)
    want_p, want_c = _reference(
        seq_lens, table, batch_size=4, draft_tokens=4, granularity=64
    )
    assert torch.equal(got_p, want_p)
    assert got_p[:3].tolist() == [-1, -1, -1]
    assert torch.equal(got_c, want_c)


def test_a_sequence_past_the_table_clamps_to_the_last_slot():
    """The torch chain clamped the slot; reading past the table would be worse."""
    seq_lens = torch.tensor([100000], device="cuda", dtype=torch.int32)
    table = torch.randint(1, 900, (1, 8), device="cuda", dtype=torch.int32)
    got_p, _ = _run(seq_lens, table, batch_size=1, draft_tokens=4, granularity=64)
    want_p, _ = _reference(
        seq_lens, table, batch_size=1, draft_tokens=4, granularity=64
    )
    assert torch.equal(got_p, want_p)
    assert got_p.item() == table[0, -1].item()


def test_rows_past_the_batch_are_not_read():
    seq_lens = torch.randint(1, 4096, (8,), device="cuda", dtype=torch.int32)
    table = torch.randint(1, 900, (8, 24), device="cuda", dtype=torch.int32)
    small, _ = _run(seq_lens, table, batch_size=3, draft_tokens=4, granularity=64)
    small = small.clone()
    seq_lens[3:] = 999999
    table[3:] = -7
    again, _ = _run(seq_lens, table, batch_size=3, draft_tokens=4, granularity=64)
    assert torch.equal(small, again)


def test_layouts_and_arguments_the_kernel_cannot_serve_are_rejected():
    seq_lens = torch.randint(1, 4096, (8,), device="cuda", dtype=torch.int32)
    table = torch.randint(1, 900, (8, 24), device="cuda", dtype=torch.int32)
    pages = torch.empty(8, dtype=torch.int32, device="cuda")
    committed = torch.empty(8, dtype=torch.int64, device="cuda")
    kw = dict(
        batch_size=8,
        draft_tokens=4,
        granularity=64,
        pages_out=pages,
        committed_out=committed,
    )

    with pytest.raises(ValueError, match="unit-stride"):
        verify_state_blocks(
            torch.zeros(16, device="cuda", dtype=torch.int32)[::2], table, **kw
        )
    with pytest.raises(ValueError, match="granularity"):
        verify_state_blocks(seq_lens, table, **{**kw, "granularity": 0})
    with pytest.raises(ValueError, match="exceeds"):
        verify_state_blocks(seq_lens, table, **{**kw, "batch_size": 9})
    with pytest.raises(ValueError, match="INT32"):
        verify_state_blocks(
            seq_lens,
            table,
            **{**kw, "pages_out": torch.empty(8, dtype=torch.int64, device="cuda")},
        )


from tokenspeed_kernel.ops.attention.triton.verify_state_blocks import (  # noqa: E402
    commit_state_pages,
)


def _commit_reference(accepted, committed, table, *, bs, draft_tokens, granularity):
    """The elementwise chain the commit kernel replaces, kept verbatim."""
    steps = accepted.to(torch.int64).clamp(min=1, max=draft_tokens)
    new_last = committed[:bs] + steps - 1
    slot = torch.div(new_last.clamp_min(0), granularity, rounding_mode="floor")
    safe = slot.clamp(max=table.shape[1] - 1)
    pages = table[:bs].gather(1, safe.unsqueeze(1)).squeeze(1).to(torch.int32)
    return pages, steps.to(torch.int32)


@pytest.mark.parametrize("bs", [1, 3, 8, 512])
@pytest.mark.parametrize("draft_tokens", [1, 4])
def test_commit_matches_the_torch_chain(bs, draft_tokens):
    g = torch.Generator(device="cuda").manual_seed(bs + draft_tokens)
    accepted = torch.randint(
        0, draft_tokens + 2, (bs,), device="cuda", dtype=torch.int32, generator=g
    )
    committed = torch.randint(
        0, 4096, (bs,), device="cuda", dtype=torch.int64, generator=g
    )
    table = torch.randint(
        0, 900, (bs, 24), device="cuda", dtype=torch.int32, generator=g
    )
    pages = torch.empty((3, bs), dtype=torch.int32, device="cuda")
    steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    commit_state_pages(
        accepted,
        committed,
        table,
        batch_size=bs,
        draft_tokens=draft_tokens,
        granularity=64,
        pages_out=pages,
        out_row=1,
        steps_out=steps,
    )
    want_p, want_s = _commit_reference(
        accepted, committed, table, bs=bs, draft_tokens=draft_tokens, granularity=64
    )
    assert torch.equal(pages[1], want_p)
    assert torch.equal(steps, want_s)


def test_commit_always_advances_at_least_one_step():
    """A round with no draft matches still advanced by the target token."""
    accepted = torch.zeros(4, device="cuda", dtype=torch.int32)
    committed = torch.tensor([0, 1, 63, 64], device="cuda", dtype=torch.int64)
    table = torch.randint(1, 900, (4, 24), device="cuda", dtype=torch.int32)
    pages = torch.empty((1, 4), dtype=torch.int32, device="cuda")
    steps = torch.empty(4, dtype=torch.int32, device="cuda")
    commit_state_pages(
        accepted,
        committed,
        table,
        batch_size=4,
        draft_tokens=4,
        granularity=64,
        pages_out=pages,
        out_row=0,
        steps_out=steps,
    )
    want_p, want_s = _commit_reference(
        accepted, committed, table, bs=4, draft_tokens=4, granularity=64
    )
    assert torch.equal(steps, want_s)
    assert steps.tolist() == [1, 1, 1, 1]
    assert torch.equal(pages[0], want_p)


def test_commit_writes_only_its_group_row():
    """Groups share one [groups, batch] buffer; a stray write lands in another
    group's write indices."""
    bs = 8
    accepted = torch.randint(0, 5, (bs,), device="cuda", dtype=torch.int32)
    committed = torch.randint(0, 4096, (bs,), device="cuda", dtype=torch.int64)
    table = torch.randint(1, 900, (bs, 24), device="cuda", dtype=torch.int32)
    pages = torch.full((3, bs), -9, dtype=torch.int32, device="cuda")
    steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    commit_state_pages(
        accepted,
        committed,
        table,
        batch_size=bs,
        draft_tokens=4,
        granularity=64,
        pages_out=pages,
        out_row=2,
        steps_out=steps,
    )
    assert (pages[0] == -9).all() and (pages[1] == -9).all()
    assert not (pages[2] == -9).any()


def test_commit_rejects_a_group_row_outside_the_buffer():
    bs = 4
    accepted = torch.zeros(bs, device="cuda", dtype=torch.int32)
    committed = torch.zeros(bs, device="cuda", dtype=torch.int64)
    table = torch.randint(1, 900, (bs, 8), device="cuda", dtype=torch.int32)
    pages = torch.empty((2, bs), dtype=torch.int32, device="cuda")
    steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="out_row"):
        commit_state_pages(
            accepted,
            committed,
            table,
            batch_size=bs,
            draft_tokens=4,
            granularity=64,
            pages_out=pages,
            out_row=2,
            steps_out=steps,
        )
