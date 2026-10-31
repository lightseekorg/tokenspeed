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

"""The verify resolve and the commit resolve over two-slot state tables.

The page verify reads is where a request's committed recurrent state lives; a
wrong page silently feeds another request's state into this one, so every
comparison is integer equality against the torch spelling.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.kda import (
    torch_commit_state_pages,
    torch_verify_state_blocks,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda._triton.state_pages import (  # noqa: E402
    _verify_state_blocks_groups_kernel,
    commit_state_pages,
    verify_commit_resolve,
    verify_state_blocks,
    verify_state_blocks_groups,
)
from tokenspeed_kernel.ops.kvcache.triton import (  # noqa: E402
    state_verify_commit_rows,
)
from utils import assert_no_triton_compile  # noqa: E402


def _two_slot_table(bs, *, generator, null_input_rows):
    table = torch.randint(
        1, 900, (bs, 2), device="cuda", dtype=torch.int32, generator=generator
    )
    table[null_input_rows, 0] = 0
    return table


def _run(table, *, batch_size):
    pages = torch.empty(batch_size, dtype=torch.int32, device="cuda")
    verify_state_blocks(table, batch_size=batch_size, pages_out=pages)
    return pages


@pytest.mark.parametrize("batch_size", [1, 3, 8, 32, 512])
def test_verify_reads_the_input_slot_when_present_else_the_live_slot(batch_size):
    g = torch.Generator(device="cuda").manual_seed(batch_size)
    table = _two_slot_table(
        batch_size, generator=g, null_input_rows=slice(0, batch_size, 2)
    )
    assert torch.equal(
        _run(table, batch_size=batch_size),
        torch_verify_state_blocks(table, batch_size=batch_size),
    )


def test_rows_past_the_batch_are_not_read():
    g = torch.Generator(device="cuda").manual_seed(7)
    table = _two_slot_table(8, generator=g, null_input_rows=slice(1, 8, 3))
    small = _run(table, batch_size=3).clone()
    table[3:] = -7
    assert torch.equal(small, _run(table, batch_size=3))


def test_layouts_and_arguments_the_kernel_cannot_serve_are_rejected():
    g = torch.Generator(device="cuda").manual_seed(1)
    table = _two_slot_table(8, generator=g, null_input_rows=slice(0, 0))
    pages = torch.empty(8, dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="unit-stride"):
        verify_state_blocks(
            torch.zeros(8, 4, device="cuda", dtype=torch.int32)[:, ::2],
            batch_size=8,
            pages_out=pages,
        )
    with pytest.raises(ValueError, match="slots"):
        verify_state_blocks(table[:, :1], batch_size=8, pages_out=pages)
    with pytest.raises(ValueError, match="exceeds"):
        verify_state_blocks(table, batch_size=9, pages_out=pages)
    with pytest.raises(ValueError, match="INT32"):
        verify_state_blocks(
            table,
            batch_size=8,
            pages_out=torch.empty(8, dtype=torch.int64, device="cuda"),
        )


@pytest.mark.parametrize("bs", [1, 3, 8, 512])
@pytest.mark.parametrize("draft_tokens", [1, 4])
def test_commit_targets_the_live_slot(bs, draft_tokens):
    g = torch.Generator(device="cuda").manual_seed(bs + draft_tokens)
    accepted = torch.randint(
        0, draft_tokens + 2, (bs,), device="cuda", dtype=torch.int32, generator=g
    )
    table = _two_slot_table(bs, generator=g, null_input_rows=slice(0, bs, 2))
    pages = torch.empty((3, bs), dtype=torch.int32, device="cuda")
    steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    commit_state_pages(
        accepted,
        table,
        batch_size=bs,
        draft_tokens=draft_tokens,
        pages_out=pages,
        out_row=1,
        steps_out=steps,
    )
    want_p, want_s = torch_commit_state_pages(
        accepted, table, bs=bs, draft_tokens=draft_tokens
    )
    assert torch.equal(pages[1], want_p)
    assert torch.equal(steps, want_s)


def test_commit_always_advances_at_least_one_step():
    """A round with no draft matches still advanced by the target token."""
    accepted = torch.zeros(4, device="cuda", dtype=torch.int32)
    g = torch.Generator(device="cuda").manual_seed(4)
    table = _two_slot_table(4, generator=g, null_input_rows=slice(0, 0))
    pages = torch.empty((1, 4), dtype=torch.int32, device="cuda")
    steps = torch.empty(4, dtype=torch.int32, device="cuda")
    commit_state_pages(
        accepted,
        table,
        batch_size=4,
        draft_tokens=4,
        pages_out=pages,
        out_row=0,
        steps_out=steps,
    )
    assert steps.tolist() == [1, 1, 1, 1]
    assert torch.equal(pages[0], table[:, 1])


def test_commit_writes_only_its_group_row():
    """Groups share one [groups, batch] buffer; a stray write lands in another
    group's write indices."""
    bs = 8
    g = torch.Generator(device="cuda").manual_seed(8)
    accepted = torch.randint(0, 5, (bs,), device="cuda", dtype=torch.int32)
    table = _two_slot_table(bs, generator=g, null_input_rows=slice(0, 0))
    pages = torch.full((3, bs), -9, dtype=torch.int32, device="cuda")
    steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    commit_state_pages(
        accepted,
        table,
        batch_size=bs,
        draft_tokens=4,
        pages_out=pages,
        out_row=2,
        steps_out=steps,
    )
    assert (pages[0] == -9).all() and (pages[1] == -9).all()
    assert torch.equal(pages[2], table[:, 1])


def test_commit_rejects_a_group_row_outside_the_buffer():
    """An out-of-range group row would write another group's indices or past
    the stacked buffer; both directions are refused before the launch."""
    bs = 4
    g = torch.Generator(device="cuda").manual_seed(2)
    accepted = torch.zeros(bs, device="cuda", dtype=torch.int32)
    table = _two_slot_table(bs, generator=g, null_input_rows=slice(0, 0))
    pages = torch.empty((2, bs), dtype=torch.int32, device="cuda")
    steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    for out_row in (2, -1):
        with pytest.raises(ValueError, match="out_row"):
            commit_state_pages(
                accepted,
                table,
                batch_size=bs,
                draft_tokens=4,
                pages_out=pages,
                out_row=out_row,
                steps_out=steps,
            )


def test_commit_honours_a_destination_row_stride_wider_than_the_batch():
    """A group buffer wider than the batch is inside the contract; a row offset
    that assumes packing shifts every later group's write indices."""
    bs = 8
    g = torch.Generator(device="cuda").manual_seed(5)
    accepted = torch.randint(0, 5, (bs,), device="cuda", dtype=torch.int32)
    table = _two_slot_table(bs, generator=g, null_input_rows=slice(0, bs, 3))
    wide = torch.full((3, bs + 1), -9, dtype=torch.int32, device="cuda")
    steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    commit_state_pages(
        accepted,
        table,
        batch_size=bs,
        draft_tokens=4,
        pages_out=wide,
        out_row=1,
        steps_out=steps,
    )
    want_p, want_s = torch_commit_state_pages(accepted, table, bs=bs, draft_tokens=4)
    assert torch.equal(wide[1, :bs], want_p)
    assert torch.equal(steps, want_s)
    assert (wide[0] == -9).all() and (wide[2] == -9).all()
    assert wide[1, bs].item() == -9


def test_cpu_callers_get_the_portable_path_with_the_kernel_result():
    """The backends' metadata tests build these tensors on CPU, where the
    wrapper must not demand a GPU; the portable spelling resolves the same
    pages and steps as the kernel for the same table."""
    bs = 6
    g = torch.Generator(device="cuda").manual_seed(6)
    table = _two_slot_table(bs, generator=g, null_input_rows=slice(1, bs, 2))
    accepted = torch.randint(0, 6, (bs,), device="cuda", dtype=torch.int32)

    cpu_pages = torch.empty(bs, dtype=torch.int32)
    verify_state_blocks(table.cpu(), batch_size=bs, pages_out=cpu_pages)
    assert torch.equal(cpu_pages, _run(table, batch_size=bs).cpu())

    cpu_stack = torch.full((2, bs), -9, dtype=torch.int32)
    cpu_steps = torch.empty(bs, dtype=torch.int32)
    commit_state_pages(
        accepted.cpu(),
        table.cpu(),
        batch_size=bs,
        draft_tokens=4,
        pages_out=cpu_stack,
        out_row=1,
        steps_out=cpu_steps,
    )
    gpu_stack = torch.full((2, bs), -9, dtype=torch.int32, device="cuda")
    gpu_steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    commit_state_pages(
        accepted,
        table,
        batch_size=bs,
        draft_tokens=4,
        pages_out=gpu_stack,
        out_row=1,
        steps_out=gpu_steps,
    )
    assert torch.equal(cpu_stack, gpu_stack.cpu())
    assert torch.equal(cpu_steps, gpu_steps.cpu())
    assert (cpu_stack[0] == -9).all()


@pytest.mark.parametrize("bs", [1, 4, 300])
def test_live_pages_out_records_the_live_slot(bs):
    g = torch.Generator(device="cuda").manual_seed(bs)
    table = _two_slot_table(bs, generator=g, null_input_rows=slice(0, bs, 2))
    pages = torch.empty(bs, dtype=torch.int32, device="cuda")
    live = torch.full((bs + 2,), -5, dtype=torch.int32, device="cuda")
    verify_state_blocks(table, batch_size=bs, pages_out=pages, live_pages_out=live)
    assert torch.equal(pages, torch_verify_state_blocks(table, batch_size=bs))
    assert torch.equal(live[:bs], table[:, 1])
    assert live[bs:].tolist() == [-5, -5]


@pytest.mark.parametrize("bs", [1, 3, 4])
def test_commit_resolve_matches_the_per_group_kernels(bs):
    """One launch gives the pages, steps and rows that commit_state_pages per
    group, state_verify_commit_rows and the gather of each layer's read page
    give."""
    g = torch.Generator(device="cuda").manual_seed(bs)
    groups, width, stride = 2, 4, 8
    tables = [
        _two_slot_table(bs, generator=g, null_input_rows=slice(0, bs, 2))
        for _ in range(groups)
    ]
    tables[1][0, 1] = 0  # a null live page: destination -1
    accepted = torch.randint(0, 6, (bs,), device="cuda", dtype=torch.int32, generator=g)
    group_sel = torch.tensor([0, 1, 1, 0, 1], device="cuda")
    layers = group_sel.numel()
    read = torch.zeros(groups, stride, dtype=torch.int32, device="cuda")
    live = torch.zeros(groups, stride, dtype=torch.int32, device="cuda")
    for i, t in enumerate(tables):
        verify_state_blocks(t, batch_size=bs, pages_out=read[i], live_pages_out=live[i])

    # The per-group reference.
    write = torch.empty(groups, bs, dtype=torch.int32, device="cuda")
    steps = torch.empty(bs, dtype=torch.int32, device="cuda")
    for i, t in enumerate(tables):
        commit_state_pages(
            accepted,
            t,
            batch_size=bs,
            draft_tokens=width,
            pages_out=write,
            out_row=i,
            steps_out=steps,
        )
    src, dst = torch.empty(2, layers * bs, dtype=torch.int32, device="cuda").unbind(0)
    state_verify_commit_rows(
        steps,
        write,
        src,
        dst,
        verify_width=width,
        num_layers=layers,
        group_indices=group_sel,
    )
    read_ref = read[:, :bs].index_select(0, group_sel).reshape(-1)

    out = torch.full((bs + 3 * layers * bs,), -9, dtype=torch.int32, device="cuda")
    got_steps = out[:bs]
    got_src, got_dst, got_read = out[bs:].view(3, layers * bs).unbind(0)
    verify_commit_resolve(
        accepted,
        live,
        read,
        group_sel,
        verify_width=width,
        steps_out=got_steps,
        src_rows=got_src,
        dst_rows=got_dst,
        read_rows=got_read,
    )
    assert torch.equal(got_steps, steps)
    assert torch.equal(got_src, src)
    assert torch.equal(got_dst, dst)
    assert torch.equal(got_read, read_ref)


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("groups", [1, 2, 3, 4])
@pytest.mark.parametrize("bs", [1, 5, 300])
def test_groups_resolve_as_the_per_group_kernel(groups, bs, packed):
    """All groups resolve in one launch, with the pages each group's own
    resolve gives, whether the tables share one upload or not."""
    gen = torch.Generator(device="cuda").manual_seed(groups * 1000 + bs)
    rows = bs + 2
    span = 2 * rows + 3
    # Group g's [rows, 2] table at a ragged offset in one int32 buffer.
    packed_buf = torch.randint(
        0, 900, (groups * span,), device="cuda", dtype=torch.int32, generator=gen
    )
    tables = [
        packed_buf[g * span + g : g * span + g + 2 * rows].view(rows, 2)
        for g in range(groups)
    ]
    if not packed:
        tables = [table.clone() for table in tables]
    pages = torch.full((groups, bs + 1), -7, dtype=torch.int32, device="cuda")
    live = torch.full_like(pages, -7)
    verify_state_blocks_groups(
        tables, batch_size=bs, pages_out=list(pages), live_pages_out=list(live)
    )

    for g, table in enumerate(tables):
        ref_pages = torch.empty(bs, dtype=torch.int32, device="cuda")
        ref_live = torch.empty_like(ref_pages)
        verify_state_blocks(
            table.clone(), batch_size=bs, pages_out=ref_pages, live_pages_out=ref_live
        )
        assert torch.equal(pages[g, :bs], ref_pages)
        assert torch.equal(live[g, :bs], ref_live)
    # Rows past the batch stay untouched.
    assert (pages[:, bs] == -7).all() and (live[:, bs] == -7).all()


def test_group_resolve_rejects_outputs_with_too_few_groups():
    tables = [torch.ones(4, 2, dtype=torch.int32, device="cuda") for _ in range(3)]
    pages = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="one source-page and one live-page"):
        verify_state_blocks_groups(
            tables,
            batch_size=4,
            pages_out=list(pages),
            live_pages_out=list(pages.clone()),
        )


def test_group_resolve_compiles_once_across_batches_and_offsets():
    """Batch sizes, table offsets and strides change every step; none may key
    a new binary."""
    buf = torch.randint(1, 900, (4096,), device="cuda", dtype=torch.int32)

    def run(bs, offset, stride):
        tables = [
            buf[offset + g * 1000 :].as_strided((bs, 2), (stride, 1)) for g in range(3)
        ]
        pages = torch.empty(3, bs, dtype=torch.int32, device="cuda")
        verify_state_blocks_groups(
            tables,
            batch_size=bs,
            pages_out=list(pages),
            live_pages_out=list(torch.empty_like(pages)),
        )

    run(2, 0, 2)
    with assert_no_triton_compile(_verify_state_blocks_groups_kernel):
        for bs, offset, stride in ((1, 1, 2), (16, 3, 4), (7, 16, 3), (300, 5, 2)):
            run(bs, offset, stride)
