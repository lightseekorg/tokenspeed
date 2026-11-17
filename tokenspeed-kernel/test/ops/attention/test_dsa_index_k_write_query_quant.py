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

"""The fused index-K write and query quantization against the two launches it replaces.

``index_k_layernorm_fp8_scatter_and_quantize_query`` must leave every byte of
the index-K plane (padding, masked and untouched rows included) as
``index_k_layernorm_fp8_scatter`` leaves it, and return exactly what
``quantize_index_query_weights`` returns for the same rows. Inputs are the
row-strided views of Kimi-K3.1's stacked projections; magnitudes span six
decades so every key and query head takes its own scale.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton
from tokenspeed_kernel.ops.attention.dsa._triton import topk as dsa_triton_topk
from tokenspeed_kernel.ops.kvcache import triton as kvcache_triton
from utils import assert_no_triton_compile

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

PAGE = 64
DIM = 128
HEADS = 32
ROW_BYTES = DIM + 4
PAGE_BYTES = PAGE * ROW_BYTES
# The index query's columns in K3.1's stacked [q_b; wq_b] output (TP8: 12 heads x 192).
Q_OFFSET = 12 * 192
Q_PITCH = Q_OFFSET + HEADS * DIM
SOFTMAX_SCALE = DIM**-0.5 * HEADS**-0.5
EPS = 1e-6


def _inputs(rows: int, *, seed: int, norm_dtype: torch.dtype = torch.float32):
    """Keys, head weights, the query and the key LayerNorm.

    Keys and head weights are views of one [wk; weights_proj] output, and the
    query is a view of the stacked query projection.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    order = torch.randperm(rows, device="cuda", generator=generator)
    magnitude = torch.logspace(-3, 3, rows, device="cuda")[order].unsqueeze(1)
    keys_weights = (randn(rows, DIM + HEADS) * magnitude + 1).to(torch.bfloat16)
    projected = (randn(rows, Q_PITCH) * magnitude).to(torch.bfloat16)
    q = projected[:, Q_OFFSET:].view(rows, HEADS, DIM)
    q[0, 0] = 0  # an all-zero head takes its scale from the amax floor
    q[-1, -1, 5] = 2.0e4  # a head whose amax dwarfs the rest
    if rows > 2:
        q[1, 3, 9] = float("nan")
        keys_weights[2, 11] = float("nan")
    norm_weight = (torch.rand(DIM, device="cuda", generator=generator) + 0.5).to(
        norm_dtype
    )
    norm_bias = (randn(DIM) * 0.1).to(norm_dtype)
    return dict(
        index_k=keys_weights[:, :DIM],
        head_weights=keys_weights[:, DIM:],
        q=q,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
    )


def _plane(pages: int, page_pad: int, *, seed: int):
    """A random-byte plane and the backing storage to compare.

    The plane is the packed slot matrix, or a page view whose pages sit
    ``page_pad`` bytes apart.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    backing = torch.randint(
        0,
        256,
        (pages, PAGE_BYTES + page_pad),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    if page_pad == 0:
        return backing.view(pages * PAGE, ROW_BYTES), backing
    return backing[:, :PAGE_BYTES], backing


def _view_like(plane: torch.Tensor, twin: torch.Tensor):
    """``plane`` re-made over ``twin``, a copy of its ``backing``."""
    if plane.shape[1] == ROW_BYTES:
        return twin.view(plane.shape)
    return twin[:, :PAGE_BYTES]


def _write_both(plane, backing, slots, write_mask, inputs):
    """Run the fused launch on ``plane`` and the two launches on a copy.

    Returns the fused outputs, the separate ones, and the copy's backing.
    """
    twin = backing.clone()
    kvcache_triton.index_k_layernorm_fp8_scatter(
        _view_like(plane, twin),
        inputs["index_k"],
        inputs["norm_weight"],
        inputs["norm_bias"],
        slots,
        eps=EPS,
        page_size=PAGE,
        group_size=DIM,
        write_mask=write_mask,
    )
    expected = dsa_triton.quantize_index_query_weights(
        inputs["q"], inputs["head_weights"], SOFTMAX_SCALE
    )
    got = dsa_triton.index_k_layernorm_fp8_scatter_and_quantize_query(
        plane,
        inputs["index_k"],
        inputs["norm_weight"],
        inputs["norm_bias"],
        slots,
        inputs["q"],
        inputs["head_weights"],
        eps=EPS,
        page_size=PAGE,
        group_size=DIM,
        write_mask=write_mask,
        softmax_scale=SOFTMAX_SCALE,
    )
    return got, expected, twin


def _assert_bitwise(got, expected) -> None:
    (q_fp8, folded), (ref_q, ref_w) = got, expected
    assert q_fp8.dtype == torch.float8_e4m3fn and q_fp8.is_contiguous()
    assert folded.dtype == torch.float32 and folded.is_contiguous()
    assert torch.equal(q_fp8.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(folded.view(torch.int32), ref_w.view(torch.int32))


def _slots(rows: int, pages: int, *, seed: int, padded: bool):
    """Distinct slots over every page.

    With ``padded``, a quarter of the rows are unowned padding at the dummy
    slot 0 or at -1, as serving masks them.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    slots = torch.randperm(pages * PAGE, device="cuda", generator=generator)[:rows]
    if not padded:
        return slots, None
    owned = torch.ones(rows, dtype=torch.bool, device="cuda")
    padding = torch.randperm(rows, device="cuda", generator=generator)[: rows // 4]
    owned[padding] = False
    slots[padding] = torch.where(padding % 2 == 0, 0, -1)
    return slots, owned


@pytest.mark.parametrize("page_pad", [0, 256])
@pytest.mark.parametrize("padded", [False, True], ids=["all_owned", "padded"])
@pytest.mark.parametrize("rows_per_request", range(1, 9))
def test_decode_and_verify_rows_match_the_two_launches(
    rows_per_request: int, padded: bool, page_pad: int
) -> None:
    rows = 3 * rows_per_request
    seed = 10 * rows_per_request + padded + page_pad
    inputs = _inputs(rows, seed=seed)
    plane, backing = _plane(6, page_pad, seed=seed)
    slots, owned = _slots(rows, 6, seed=seed, padded=padded)

    got, expected, twin = _write_both(plane, backing, slots, owned, inputs)

    assert torch.equal(backing, twin)
    _assert_bitwise(got, expected)


@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
def test_many_rows_match_the_two_launches(norm_dtype: torch.dtype) -> None:
    """4096 rows (512 requests of 8) match the two launches.

    A key LayerNorm summed in another order would flip some of their BF16
    roundings.
    """
    rows = 4096
    inputs = _inputs(rows, seed=1, norm_dtype=norm_dtype)
    plane, backing = _plane(80, 0, seed=2)
    slots, _ = _slots(rows, 80, seed=3, padded=False)

    got, expected, twin = _write_both(plane, backing, slots, None, inputs)

    assert torch.equal(backing, twin)
    _assert_bitwise(got, expected)


def test_far_pages_past_int32_offsets() -> None:
    """Pages past 2**31 bytes of the plane take the same bytes.

    Nothing lands before them.
    """
    rows, pages = 24, 3
    first = 2**31 // PAGE_BYTES + 1
    backing = torch.empty(first + pages, PAGE_BYTES, dtype=torch.uint8, device="cuda")
    far = backing[first - 1 :]
    far.copy_(_plane(pages + 1, 0, seed=4)[1])
    before = far.clone()
    inputs = _inputs(rows, seed=5)
    slots, owned = _slots(rows, pages, seed=6, padded=True)
    slots = torch.where(owned, slots + first * PAGE, slots)
    assert int(slots.max()) * ROW_BYTES > 2**31

    kvcache_triton.index_k_layernorm_fp8_scatter(
        backing,
        inputs["index_k"],
        inputs["norm_weight"],
        inputs["norm_bias"],
        slots,
        eps=EPS,
        page_size=PAGE,
        group_size=DIM,
        write_mask=owned,
    )
    expected_far = far.clone()
    far.copy_(before)
    got = dsa_triton.index_k_layernorm_fp8_scatter_and_quantize_query(
        backing,
        inputs["index_k"],
        inputs["norm_weight"],
        inputs["norm_bias"],
        slots,
        inputs["q"],
        inputs["head_weights"],
        eps=EPS,
        page_size=PAGE,
        group_size=DIM,
        write_mask=owned,
        softmax_scale=SOFTMAX_SCALE,
    )

    assert torch.equal(far, expected_far)
    assert torch.equal(far[0], before[0])  # the page before the written ones
    assert not torch.equal(far[1:], before[1:])
    _assert_bitwise(
        got,
        dsa_triton.quantize_index_query_weights(
            inputs["q"], inputs["head_weights"], SOFTMAX_SCALE
        ),
    )


def test_no_recompile_across_batch_shapes() -> None:
    plane, _ = _plane(64, 0, seed=7)

    def run(rows: int, padded: bool) -> None:
        inputs = _inputs(rows, seed=rows)
        slots, owned = _slots(rows, 64, seed=rows, padded=padded)
        dsa_triton.index_k_layernorm_fp8_scatter_and_quantize_query(
            plane,
            inputs["index_k"],
            inputs["norm_weight"],
            inputs["norm_bias"],
            slots,
            inputs["q"],
            inputs["head_weights"],
            eps=EPS,
            page_size=PAGE,
            group_size=DIM,
            write_mask=owned,
            softmax_scale=SOFTMAX_SCALE,
        )

    for padded in (False, True):
        run(3, padded)
    with assert_no_triton_compile(dsa_triton_topk._index_k_write_quantize_query_kernel):
        for rows in (1, 7, 8, 64, 300, 2048, 4096):
            for padded in (False, True):
                run(rows, padded)


def test_graph_replay_follows_new_rows() -> None:
    rows = 3 * 8
    inputs = _inputs(rows, seed=8)
    plane, backing = _plane(6, 256, seed=9)
    slots, owned = _slots(rows, 6, seed=10, padded=True)

    def run():
        return dsa_triton.index_k_layernorm_fp8_scatter_and_quantize_query(
            plane,
            inputs["index_k"],
            inputs["norm_weight"],
            inputs["norm_bias"],
            slots,
            inputs["q"],
            inputs["head_weights"],
            eps=EPS,
            page_size=PAGE,
            group_size=DIM,
            write_mask=owned,
            softmax_scale=SOFTMAX_SCALE,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    # New rows in place: other keys, queries, weights, slots and owners.
    fresh = _inputs(rows, seed=11)
    for name in ("index_k", "head_weights", "q"):
        inputs[name].copy_(fresh[name])
    new_slots, new_owned = _slots(rows, 6, seed=12, padded=True)
    assert not torch.equal(new_owned, owned)
    slots.copy_(new_slots)
    owned.copy_(new_owned)
    twin = backing.clone()
    graph.replay()
    torch.cuda.synchronize()

    kvcache_triton.index_k_layernorm_fp8_scatter(
        _view_like(plane, twin),
        inputs["index_k"],
        inputs["norm_weight"],
        inputs["norm_bias"],
        slots,
        eps=EPS,
        page_size=PAGE,
        group_size=DIM,
        write_mask=owned,
    )
    assert torch.equal(backing, twin)
    _assert_bitwise(
        captured,
        dsa_triton.quantize_index_query_weights(
            inputs["q"], inputs["head_weights"], SOFTMAX_SCALE
        ),
    )


def test_rows_of_another_count_are_refused() -> None:
    inputs = _inputs(4, seed=13)
    plane, _ = _plane(2, 0, seed=14)
    slots, _ = _slots(4, 2, seed=15, padded=False)
    with pytest.raises(ValueError, match="fused write quantizes the queries"):
        dsa_triton.index_k_layernorm_fp8_scatter_and_quantize_query(
            plane,
            inputs["index_k"],
            inputs["norm_weight"],
            inputs["norm_bias"],
            slots,
            inputs["q"][:3],
            inputs["head_weights"][:3],
            eps=EPS,
            page_size=PAGE,
            group_size=DIM,
            write_mask=None,
            softmax_scale=SOFTMAX_SCALE,
        )
