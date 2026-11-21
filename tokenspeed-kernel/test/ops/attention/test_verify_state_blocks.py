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
    _verify_state_groups_kernel,
    commit_state_pages,
    verify_state_blocks,
    verify_state_groups,
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


def _packed_group_tables(groups, rows, width, *, seed):
    """Group tables as views of one packed upload, the way the bridge ships them."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(
        1,
        900,
        (groups * rows * width + 1,),
        device="cuda",
        dtype=torch.int32,
        generator=g,
    )
    # The odd leading word puts the tables at unaligned offsets.
    tables = [
        packed[1 + i * rows * width : 1 + (i + 1) * rows * width].view(rows, width)
        for i in range(groups)
    ]
    for i, table in enumerate(tables):
        table[i % 2 :: 2, 0] = 0
    return tables


def test_a_group_off_the_gpu_keeps_its_own_resolve():
    """Tables on different devices skip the shared launch, not the CPU path."""
    gpu, cpu = _packed_group_tables(2, 4, 3, seed=11)
    cpu = cpu.cpu()
    pages = [
        torch.full((6,), -9, dtype=torch.int32, device=table.device)
        for table in (gpu, cpu)
    ]
    verify_state_groups(
        [gpu, cpu],
        batch_size=4,
        padded_batch_size=6,
        pages_out=pages,
        live_pages_out=None,
        pad_page=-1,
    )
    for table, page in zip((gpu, cpu), pages):
        want = torch.full((6,), -1, dtype=torch.int32)
        want[:4] = torch_verify_state_blocks(table, batch_size=4).cpu()
        assert torch.equal(page.cpu(), want)


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "batch_size, padded_batch_size", [(1, 1), (3, 4), (5, 8), (0, 2)]
)
@pytest.mark.parametrize("live", [True, False])
def test_grouped_resolve_matches_each_group_and_pads_the_graph_rows(
    groups, batch_size, padded_batch_size, live
):
    """One launch per call does what a resolve per group plus padding fills did."""
    tables = _packed_group_tables(groups, max(batch_size, 1), 3, seed=groups)
    width = padded_batch_size + 3
    pages = [torch.full((width,), -9, dtype=torch.int32, device="cuda") for _ in tables]
    lives = [torch.full((width,), -9, dtype=torch.int32, device="cuda") for _ in tables]
    verify_state_groups(
        tables,
        batch_size=batch_size,
        padded_batch_size=padded_batch_size,
        pages_out=pages,
        live_pages_out=lives if live else None,
        pad_page=-1,
    )
    for table, page, live_page in zip(tables, pages, lives):
        want = torch.full((width,), -9, dtype=torch.int32, device="cuda")
        want[:padded_batch_size] = -1
        want[:batch_size] = torch_verify_state_blocks(table, batch_size=batch_size)
        assert torch.equal(page, want)
        want_live = torch.full((width,), -9, dtype=torch.int32, device="cuda")
        if live:
            want_live[:padded_batch_size] = -1
            want_live[:batch_size] = table[:batch_size, 1]
        assert torch.equal(live_page, want_live)


def test_grouped_resolve_compiles_once_across_batches_and_offsets():
    """The packed upload moves the tables with every batch; that must not recompile."""
    pages = [torch.empty(64, dtype=torch.int32, device="cuda") for _ in range(3)]
    lives = [torch.empty(64, dtype=torch.int32, device="cuda") for _ in range(3)]

    def resolve(batch_size, padded_batch_size):
        tables = _packed_group_tables(3, batch_size, 2, seed=batch_size)
        verify_state_groups(
            tables,
            batch_size=batch_size,
            padded_batch_size=padded_batch_size,
            pages_out=pages,
            live_pages_out=lives,
            pad_page=-1,
        )

    resolve(1, 1)
    resolve(16, 16)
    with assert_no_triton_compile(_verify_state_groups_kernel):
        for batch_size, padded_batch_size in (
            (2, 2),
            (3, 4),
            (7, 8),
            (32, 32),
            (33, 48),
        ):
            resolve(batch_size, padded_batch_size)


def test_grouped_resolve_rejects_mismatched_outputs():
    tables = _packed_group_tables(2, 4, 2, seed=3)
    pages = [torch.empty(4, dtype=torch.int32, device="cuda") for _ in tables]
    with pytest.raises(ValueError, match="own page outputs"):
        verify_state_groups(
            tables,
            batch_size=4,
            padded_batch_size=4,
            pages_out=pages[:1],
            live_pages_out=None,
            pad_page=-1,
        )
    with pytest.raises(ValueError, match="padded_batch_size rows"):
        verify_state_groups(
            tables,
            batch_size=4,
            padded_batch_size=5,
            pages_out=pages,
            live_pages_out=None,
            pad_page=-1,
        )
    with pytest.raises(ValueError, match="batch_size <= padded_batch_size"):
        verify_state_groups(
            tables,
            batch_size=4,
            padded_batch_size=3,
            pages_out=pages,
            live_pages_out=None,
            pad_page=-1,
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
