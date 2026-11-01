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

"""Sorted-route MXFP4 prefill split at the intermediate (gfx950)."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the Gluon MXFP4 sorted prefill tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.moe import (  # noqa: E402
    moe_sorted_down,
    moe_sorted_gate_up,
    prepare_moe_sorted_down_weight,
    warmup_moe_sorted_prefill,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import (  # noqa: E402
    moe_sorting,
    prefill_stage1,
    prefill_stage2,
    scale,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    sorted_route_capacity,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (  # noqa: E402
    empty_swizzled_cdna4_mxfp4_scale,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)

EXPERTS = 32
HIDDEN = 1024
INTER = 2048
TOP_K = 8
SLICES = 4


def _codes(g, *shape):
    return torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8, generator=g)


def _scales(g, *shape):
    return torch.randint(118, 124, shape, device="cuda", dtype=torch.uint8, generator=g)


def _experts(w13: torch.Tensor, w13_scale: torch.Tensor) -> torch.nn.Module:
    """A processed all-local expert bank holding these gate/up rows."""
    w = torch.nn.Module()
    inter = w13.shape[1] // 2
    w.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(w13_scale.clone(), requires_grad=False)
    # The down projection is not used by the gate/up half.
    w.w2_weight = torch.nn.Parameter(
        torch.zeros(EXPERTS, HIDDEN, inter // 2, device="cuda", dtype=torch.uint8),
        requires_grad=False,
    )
    w.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (EXPERTS, HIDDEN, inter // 32), 127, device="cuda", dtype=torch.uint8
        ),
        requires_grad=False,
    )
    w.w13_input_layout = "concatenated"
    w.num_experts = EXPERTS
    w.num_local_experts = EXPERTS
    w.top_k = TOP_K
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, w)
    return w


def _inputs(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = _codes(g, rows, HIDDEN // 2)
    scales = empty_swizzled_cdna4_mxfp4_scale(rows, HIDDEN // 32, device="cuda")
    scales.copy_(_scales(g, *scales.shape))
    ids = torch.rand(rows, EXPERTS, device="cuda", generator=g).topk(TOP_K, dim=-1)
    weights = torch.rand(rows, TOP_K, device="cuda", generator=g)
    return (packed, scales), weights, ids.indices.to(torch.int32).contiguous()


def _logical_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Read ``[rows, cols]`` E8M0 values from contiguous CDNA4 scale panels."""
    m = torch.arange(rows, device=scales.device)[:, None]
    k = torch.arange(cols, device=scales.device)[None, :]
    swizzled = ((((k // 8) * 4 + k % 4) * 16 + m % 16) * 2 + (k % 8) // 4) * 2 + (
        m % 32
    ) // 16
    return scales.flatten()[swizzled + (m // 32) * (cols * 32)]


def _gate_up(w, rows: int, seed: int):
    inter = w.w13_weight_triton_tensor.gluon_package_prefill_weight.shape[1] // 2
    capacity = sorted_route_capacity(rows, EXPERTS, TOP_K)
    data = torch.zeros((capacity, inter // 2), device="cuda", dtype=torch.uint8)
    scales = torch.zeros(
        ((capacity + 31) // 32 * 32, inter // 32), device="cuda", dtype=torch.uint8
    )
    x, weights, ids = _inputs(rows, seed)
    routes = moe_sorted_gate_up(x, weights, ids, w, data, scales)
    return routes, data, scales


@pytest.mark.parametrize(
    "rows, experts, start, block, minimum, runtime_rows",
    [
        (8192, 256, 0, 64, False, False),  # expert TP prefill: 256-route chunks
        (8192, 32, 64, 64, False, False),  # expert parallel: 1024-route chunks
        (2049, 256, 0, 128, False, False),
        (150, 32, 32, 32, False, True),
        (100, 32, 0, 32, True, False),  # one fused chunk
        (193, 256, 0, 16, False, False),
    ],
)
def test_deterministic_sort_is_the_stable_expert_order(
    rows, experts, start, block, minimum, runtime_rows
) -> None:
    g = torch.Generator(device="cuda").manual_seed(rows)
    ids = torch.rand(rows, 256, device="cuda", generator=g).topk(TOP_K, dim=-1)
    ids = ids.indices.to(torch.int32).contiguous()
    weights = torch.rand(rows, TOP_K, device="cuda", generator=g)

    def sort():
        return moe_sorting.gluon_moe_sorting(
            ids,
            weights,
            experts,
            HIDDEN,
            torch.bfloat16,
            block,
            use_minimum_route_chunks=minimum,
            runtime_rows=runtime_rows,
            deterministic_order=True,
            expert_start=start,
        )

    sorted_ids, sorted_weights, expert_ids, valid, _ = sort()
    # Reference: each local expert's routes in flat top-k order, padded to blocks.
    flat = ids.flatten().cpu() - start
    expected = []
    expected_weights = []
    expected_blocks = []
    for expert in range(experts):
        routes = torch.nonzero(flat == expert).flatten()
        if routes.numel() == 0:
            continue
        padded = (routes.numel() + block - 1) // block * block
        packed = (routes % TOP_K) << 24 | (routes // TOP_K)
        expected += packed.tolist() + [TOP_K << 24 | rows] * (padded - routes.numel())
        expected_weights += weights.flatten().cpu()[routes].tolist()
        expected_weights += [None] * (padded - routes.numel())
        expected_blocks += [expert] * (padded // block)
    assert int(valid[0]) == len(expected)
    assert sorted_ids[: len(expected)].cpu().tolist() == expected
    assert expert_ids[: len(expected_blocks)].cpu().tolist() == expected_blocks
    actual_weights = sorted_weights[: len(expected)].cpu().tolist()
    assert [a for a, e in zip(actual_weights, expected_weights) if e is not None] == [
        e for e in expected_weights if e is not None
    ]
    again = sort()[0]
    assert torch.equal(again[: len(expected)], sorted_ids[: len(expected)])


@pytest.mark.parametrize("rows", [9, 300, 1100])
def test_gate_up_slices_concatenate_into_the_full_intermediate(rows: int) -> None:
    """Each rank's slice is a K range of the full intermediate, row for row."""
    g = torch.Generator(device="cuda").manual_seed(3)
    gate = _codes(g, EXPERTS, INTER, HIDDEN // 2)
    up = _codes(g, EXPERTS, INTER, HIDDEN // 2)
    gate_scale = _scales(g, EXPERTS, INTER, HIDDEN // 32)
    up_scale = _scales(g, EXPERTS, INTER, HIDDEN // 32)
    full_routes, full, full_scales = _gate_up(
        _experts(torch.cat((gate, up), 1), torch.cat((gate_scale, up_scale), 1)),
        rows,
        seed=rows,
    )
    width = INTER // SLICES
    sorted_rows = int(full_routes.num_valid_ids[0])
    token = full_routes.sorted_ids[:sorted_rows] & 0xFFFFFF
    real = token < rows
    full_logical = _logical_scales(full_scales, sorted_rows, INTER // 32)
    for index in range(SLICES):
        part = slice(index * width, (index + 1) * width)
        w = _experts(
            torch.cat((gate[:, part], up[:, part]), 1),
            torch.cat((gate_scale[:, part], up_scale[:, part]), 1),
        )
        routes, data, scales = _gate_up(w, rows, seed=rows)
        assert torch.equal(
            routes.sorted_ids[:sorted_rows], full_routes.sorted_ids[:sorted_rows]
        )
        columns = slice(index * width // 2, (index + 1) * width // 2)
        assert torch.equal(data[:sorted_rows][real], full[:sorted_rows, columns][real])
        logical = _logical_scales(scales, sorted_rows, width // 32)
        groups = slice(index * width // 32, (index + 1) * width // 32)
        assert torch.equal(logical[real], full_logical[:, groups][real])


def test_column_split_down_matches_the_full_projection_columns() -> None:
    g = torch.Generator(device="cuda").manual_seed(5)
    w2 = _codes(g, EXPERTS, HIDDEN, INTER // 2)
    w2_scale = _scales(g, EXPERTS, HIDDEN, INTER // 32)
    w13 = _codes(g, EXPERTS, 2 * INTER, HIDDEN // 2)
    w13_scale = _scales(g, EXPERTS, 2 * INTER, HIDDEN // 32)
    rows = 517
    routes, data, scales = _gate_up(_experts(w13, w13_scale), rows, seed=7)
    full = moe_sorted_down(
        data,
        scales,
        routes,
        prepare_moe_sorted_down_weight(w2, w2_scale),
        torch.empty((rows * TOP_K, HIDDEN), device="cuda", dtype=torch.bfloat16),
    )
    width = HIDDEN // SLICES
    for index in range(SLICES):
        part = slice(index * width, (index + 1) * width)
        columns = moe_sorted_down(
            data,
            scales,
            routes,
            prepare_moe_sorted_down_weight(
                w2[:, part].contiguous(), w2_scale[:, part].contiguous()
            ),
            torch.empty((rows * TOP_K, width), device="cuda", dtype=torch.bfloat16),
        )
        assert torch.equal(columns.view(torch.int16), full[:, part].view(torch.int16))

    # Every route against an FP32 product of the dequantized operands.
    sorted_rows = int(routes.num_valid_ids[0])
    token = (routes.sorted_ids[:sorted_rows] & 0xFFFFFF).long()
    slot = (routes.sorted_ids[:sorted_rows] >> 24).long()
    real = token < rows
    activation = dequantize_mxfp4(
        data[:sorted_rows], _logical_scales(scales, sorted_rows, INTER // 32)
    )
    experts = torch.repeat_interleave(
        routes.sorted_expert_ids[: sorted_rows // routes.block_m].long(),
        routes.block_m,
    )
    weight = dequantize_mxfp4(w2, w2_scale)
    expected = torch.empty((sorted_rows, HIDDEN), device="cuda")
    for expert in experts.unique().tolist():
        rows_of = experts == expert
        expected[rows_of] = activation[rows_of].float() @ weight[expert].float().T
    actual = full.view(rows, TOP_K, HIDDEN)[token[real], slot[real]].float()
    expected = expected[real]
    # The kernel rounds each FP32-accumulated route once to BF16.
    torch.testing.assert_close(
        actual, expected, atol=2**-8 * float(expected.abs().max()), rtol=2**-7
    )


def test_warmup_covers_every_row_count_in_its_range() -> None:
    g = torch.Generator(device="cuda").manual_seed(9)
    w = _experts(
        _codes(g, EXPERTS, 2 * INTER, HIDDEN // 2),
        _scales(g, EXPERTS, 2 * INTER, HIDDEN // 32),
    )
    weight = prepare_moe_sorted_down_weight(
        _codes(g, EXPERTS, HIDDEN // SLICES, INTER // 2),
        _scales(g, EXPERTS, HIDDEN // SLICES, INTER // 32),
    )
    warmup_moe_sorted_prefill(w, weight, 192, 1100)
    kernels = (
        moe_sorting._moe_sorting_stage1_kernel,
        moe_sorting._moe_sorting_stage2_kernel,
        moe_sorting._moe_sorting_stage3_kernel,
        moe_sorting._moe_sorting_stage4_kernel,
        scale._gather_package_cdna4_scale_kernel,
        prefill_stage1.gluon_mxfp4_moe_stage1_kernel,
        prefill_stage2.gluon_mxfp4_moe_stage2_1x2_kernel,
    )
    with assert_no_triton_compile(*kernels):
        for rows in (192, 199, 256, 257, 300, 511, 512, 640, 1023, 1024, 1025, 1100):
            routes, data, scales = _gate_up(w, rows, seed=rows)
            moe_sorted_down(
                data,
                scales,
                routes,
                weight,
                torch.empty(
                    (rows * TOP_K, HIDDEN // SLICES),
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
            )
    torch.cuda.synchronize()
