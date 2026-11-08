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
from tokenspeed_kernel.numerics.torch.attention.qsa import torch_qsa_bf16_kv_scratch
from tokenspeed_kernel.ops.attention.qsa import (
    qsa_bf16_kv_scratch,
    qsa_bf16_scratch_pages,
)
from tokenspeed_kernel.ops.attention.qsa.triton import (
    _qsa_bf16_kv_scratch_copy_kernel,
    _qsa_bf16_kv_scratch_remap_kernel,
    qwen4_exp_qsa_selected_slots,
)
from utils import assert_no_triton_compile, move_pages_past_int32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the QSA BF16 KV scratch kernels require CUDA or ROCm",
)

FP8 = torch.float8_e4m3fn


def _cache(slots: int, heads: int, *, interleaved: bool, generator, device):
    """FP8 K and V ``[slots, heads, 256]``; interleaved shares one buffer."""
    shape = (slots, 2, heads, 256) if interleaved else (2, slots, heads, 256)
    values = (torch.rand(shape, generator=generator, device=device) * 8 - 4).to(FP8)
    if interleaved:
        return values[:, 0], values[:, 1]
    return values[0], values[1]


def _table(requests, page_size, num_pages, *, shared, generator, device):
    """Distinct pages per request; request 1 may start with request 0's pages."""
    widths = [-(-(prefix + new) // page_size) for prefix, new in requests]
    free = (
        torch.randperm(num_pages - 1, generator=generator, device=device) + 1
    ).tolist()
    table = torch.zeros(
        (len(requests) + 1, max(widths) + 3), dtype=torch.int32, device=device
    )
    for row, width in enumerate(widths):
        share = min(shared, width, widths[0]) if row == 1 else 0
        table[row, :share] = table[0, :share]
        for column in range(share, width):
            table[row, column] = free.pop()
    return table


def _selection(requests, table, page_size, *, slots, generator, device):
    """Real QSA slots: random completed blocks, the open tail, -1 padding."""
    positions = torch.cat(
        [torch.arange(prefix, prefix + new, device=device) for prefix, new in requests]
    )
    owners = torch.repeat_interleave(
        torch.arange(len(requests), dtype=torch.int32, device=device),
        torch.tensor([new for _, new in requests], device=device),
    )
    complete = ((positions + 1) // 4).to(torch.int32)
    scores = torch.rand(
        (positions.numel(), int(complete.max()) + 1), generator=generator, device=device
    )
    blocks = torch.arange(scores.shape[1], device=device)
    scores = torch.where(blocks[None, :] < complete[:, None], scores, -1.0)
    top = min(512, scores.shape[1])
    chosen = scores.topk(top, dim=1)
    picked = torch.where(chosen.values >= 0, chosen.indices, -1).to(torch.int32)
    selected_blocks = torch.full(
        (positions.numel(), 512), -1, dtype=torch.int32, device=device
    )
    selected_blocks[:, :top] = picked
    selected = qwen4_exp_qsa_selected_slots(
        selected_blocks,
        complete,
        positions,
        owners,
        table,
        page_size,
        4,
        2048,
        enable_pdl=False,
    )
    # A partial last cache page holds no rows past the cache end; padding
    # comes in every non-positive flavour.
    selected[selected >= slots] = -1
    selected[0, 5] = 0
    selected[-1, 7] = -12345
    return selected


def qsa_bf16_kv_scratch_triton(*args, **kwargs):
    """The op through the registry, restricted to the Triton solution."""
    return qsa_bf16_kv_scratch(*args, **kwargs, override=None, solution="triton")


def _run(impl, k, v, selected, table, seq_lens, page_size, *, capacity, pool_pages):
    heads = k.shape[1]
    k_scratch = torch.full(
        (capacity * page_size, heads, 256),
        float("nan"),
        dtype=torch.bfloat16,
        device=k.device,
    )
    v_scratch = torch.full_like(k_scratch, float("nan"))
    k_scratch[:page_size].zero_()
    v_scratch[:page_size].zero_()
    # Stale entries must never be read: the copy writes what the remap reads.
    page_map = torch.full(
        (pool_pages + 7,), 1 << 30, dtype=torch.int32, device=k.device
    )
    out = torch.full_like(selected, 777)
    pages = qsa_bf16_scratch_pages(seq_lens.tolist(), page_size)
    result = impl(
        k,
        v,
        selected,
        table,
        seq_lens,
        num_requests=seq_lens.numel(),
        page_size=page_size,
        scratch_pages=pages,
        k_scratch=k_scratch,
        v_scratch=v_scratch,
        page_map=page_map,
        out_slots=out,
        enable_pdl=False,
    )
    assert result is out
    return k_scratch, v_scratch, out, pages


@pytest.mark.parametrize(
    "requests,page_size,heads,shared,tail,interleaved,length_dtype",
    [
        ([(3000, 160), (0, 96), (6000, 128)], 64, 2, 0, 0, False, torch.int32),
        ([(2000, 64), (1500, 80)], 64, 2, 20, 0, False, torch.int64),
        ([(1000, 70), (300, 40)], 64, 2, 0, 17, False, torch.int32),
        ([(700, 33), (0, 50)], 64, 2, 0, 0, True, torch.int32),
        ([(5000, 300), (40, 20)], 256, 1, 0, 0, False, torch.int32),
    ],
)
def test_bf16_kv_scratch_matches_reference(
    device, requests, page_size, heads, shared, tail, interleaved, length_dtype
):
    generator = torch.Generator(device=device).manual_seed(len(requests) + tail)
    pool_pages = sum(-(-(p + n) // page_size) for p, n in requests) + 40
    slots = pool_pages * page_size - tail
    k, v = _cache(
        slots, heads, interleaved=interleaved, generator=generator, device=device
    )
    table = _table(
        requests,
        page_size,
        pool_pages,
        shared=shared,
        generator=generator,
        device=device,
    )
    selected = _selection(
        requests, table, page_size, slots=slots, generator=generator, device=device
    )
    seq_lens = torch.tensor(
        [p + n for p, n in requests], dtype=length_dtype, device=device
    )
    kwargs = dict(capacity=pool_pages, pool_pages=pool_pages)
    k_scratch, v_scratch, out, pages = _run(
        qsa_bf16_kv_scratch_triton, k, v, selected, table, seq_lens, page_size, **kwargs
    )
    k_ref, v_ref, out_ref, _ = _run(
        torch_qsa_bf16_kv_scratch, k, v, selected, table, seq_lens, page_size, **kwargs
    )

    valid = selected > 0
    assert torch.equal(out[~valid], selected[~valid])
    assert bool(((out[valid] >= page_size) & (out[valid] < pages * page_size)).all())
    assert torch.equal(out[valid] % page_size, selected[valid] % page_size)
    # Every remapped slot holds the exact E4M3 -> BF16 value of its source.
    source = selected[valid].long()
    for scratch, cache in ((k_scratch, k), (v_scratch, v)):
        expected = cache[source].float().to(torch.bfloat16)
        assert torch.equal(
            scratch[out[valid].long()].view(torch.int16), expected.view(torch.int16)
        )
        assert bool((scratch[:page_size] == 0).all())
    if not shared:
        # A shared page may map to either copy; otherwise the layout is exact.
        assert torch.equal(out, out_ref)
        written = pages * page_size
        assert torch.equal(
            k_scratch[:written].view(torch.int16), k_ref[:written].view(torch.int16)
        )
        assert torch.equal(
            v_scratch[:written].view(torch.int16), v_ref[:written].view(torch.int16)
        )


# The runtime's page map aliases the previous layer's output, so a stale entry
# may hold any value, including a page of the current call.
@pytest.mark.parametrize("stale", [-7, 2])
def test_bf16_kv_scratch_keeps_slots_outside_the_plan_inside_the_copy(device, stale):
    generator = torch.Generator(device=device).manual_seed(5)
    requests, page_size = [(600, 20), (300, 30)], 64
    pool_pages = 40
    slots = pool_pages * page_size
    k, v = _cache(slots, 2, interleaved=False, generator=generator, device=device)
    table = _table(
        requests, page_size, pool_pages, shared=0, generator=generator, device=device
    )
    selected = _selection(
        requests, table, page_size, slots=slots, generator=generator, device=device
    )
    seq_lens = torch.tensor([620, 330], dtype=torch.int32, device=device)
    pages = qsa_bf16_scratch_pages(seq_lens.tolist(), page_size)
    k_scratch = torch.zeros(
        (pages * page_size, 2, 256), dtype=torch.bfloat16, device=device
    )
    v_scratch = torch.zeros_like(k_scratch)
    page_map = torch.full((pool_pages,), stale, dtype=torch.int32, device=device)
    out = torch.empty_like(selected)
    # A launch three pages short leaves the second request's last pages out:
    # their slots read stale entries and must become padding or a copied
    # page, never an address past the copy.
    qsa_bf16_kv_scratch_triton(
        k,
        v,
        selected,
        table,
        seq_lens,
        num_requests=2,
        page_size=page_size,
        scratch_pages=pages - 3,
        k_scratch=k_scratch,
        v_scratch=v_scratch,
        page_map=page_map,
        out_slots=out,
        enable_pdl=False,
    )
    valid = selected > 0
    remapped = out[valid]
    copied = (remapped >= page_size) & (remapped < (pages - 3) * page_size)
    assert bool(((remapped == -1) | copied).all())
    # Entries naming no page of this call turn their slots into padding.
    assert bool((remapped == -1).any()) == (stale < 0)


def test_bf16_kv_scratch_offsets_past_int32(device):
    generator = torch.Generator(device=device).manual_seed(7)
    requests, page_size, heads = [(900, 100), (0, 300)], 64, 2
    num_pages = sum(-(-(p + n) // page_size) for p, n in requests) + 2
    k, v = _cache(
        num_pages * page_size,
        heads,
        interleaved=False,
        generator=generator,
        device=device,
    )
    table = _table(
        requests, page_size, num_pages, shared=0, generator=generator, device=device
    )
    (k_far, v_far), far_table = move_pages_past_int32(
        (
            k.view(num_pages, page_size, heads, 256),
            v.view(num_pages, page_size, heads, 256),
        ),
        table,
    )
    k_far = k_far.view(-1, heads, 256)
    v_far = v_far.view(-1, heads, 256)
    assert k_far.numel() > 2**31
    selected = _selection(
        requests,
        far_table,
        page_size,
        slots=k_far.shape[0],
        generator=generator,
        device=device,
    )
    seq_lens = torch.tensor(
        [p + n for p, n in requests], dtype=torch.int32, device=device
    )
    far_pages = k_far.shape[0] // page_size
    k_scratch, v_scratch, out, pages = _run(
        qsa_bf16_kv_scratch_triton,
        k_far,
        v_far,
        selected,
        far_table,
        seq_lens,
        page_size,
        capacity=num_pages + 1,
        pool_pages=far_pages,
    )
    valid = selected > 0
    source = selected[valid].long()
    assert int(source.min()) * heads * 256 > 2**31
    expected = k_far[source].float().to(torch.bfloat16)
    assert torch.equal(
        k_scratch[out[valid].long()].view(torch.int16), expected.view(torch.int16)
    )
    expected = v_far[source].float().to(torch.bfloat16)
    assert torch.equal(
        v_scratch[out[valid].long()].view(torch.int16), expected.view(torch.int16)
    )


def test_bf16_kv_scratch_reuses_binaries_across_shapes(device):
    """Request, row and page counts, pool sizes and table widths are runtime."""
    page_size, heads = 64, 2

    def call(requests, pool_pages, extra_columns, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        slots = pool_pages * page_size
        k, v = _cache(
            slots, heads, interleaved=False, generator=generator, device=device
        )
        table = _table(
            requests,
            page_size,
            pool_pages,
            shared=0,
            generator=generator,
            device=device,
        )
        table = torch.nn.functional.pad(table, (0, extra_columns))
        selected = _selection(
            requests, table, page_size, slots=slots, generator=generator, device=device
        )
        seq_lens = torch.tensor(
            [p + n for p, n in requests], dtype=torch.int32, device=device
        )
        _run(
            qsa_bf16_kv_scratch_triton,
            k,
            v,
            selected,
            table,
            seq_lens,
            page_size,
            capacity=pool_pages,
            pool_pages=pool_pages,
        )

    call([(300, 40), (0, 24)], 64, 0, 1)
    torch.cuda.synchronize()
    with assert_no_triton_compile(
        _qsa_bf16_kv_scratch_copy_kernel, _qsa_bf16_kv_scratch_remap_kernel
    ):
        call([(1000, 17)], 80, 1, 2)
        call([(64, 64), (128, 1), (0, 200)], 97, 16, 3)
        call([(5, 3), (2000, 48), (33, 33), (0, 1)], 160, 5, 4)
        torch.cuda.synchronize()
