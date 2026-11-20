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
from torch.profiler import ProfilerActivity, profile
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


def _experts(
    w13: torch.Tensor, w13_scale: torch.Tensor, hidden: int = HIDDEN
) -> torch.nn.Module:
    """A processed all-local expert bank holding these gate/up rows."""
    w = torch.nn.Module()
    inter = w13.shape[1] // 2
    w.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(w13_scale.clone(), requires_grad=False)
    # The down projection is not used by the gate/up half.
    w.w2_weight = torch.nn.Parameter(
        torch.zeros(EXPERTS, hidden, inter // 2, device="cuda", dtype=torch.uint8),
        requires_grad=False,
    )
    w.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (EXPERTS, hidden, inter // 32), 127, device="cuda", dtype=torch.uint8
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


def test_stage1_rejects_rows_past_32_bit_offsets() -> None:
    """Activation rows are gathered with 32-bit offsets; the launcher bounds them."""
    rows, k_packed = 2**20, 2048
    g = torch.Generator(device="cuda").manual_seed(5)
    with pytest.raises(ValueError, match="2\\^31 bytes"):
        prefill_stage1.invoke_gluon_mxfp4_moe_stage1(
            torch.empty((rows, k_packed), device="cuda", dtype=torch.uint8),
            _codes(g, 1, 128, k_packed),
            None,
            torch.zeros(64, device="cuda", dtype=torch.int32),
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.empty((64, 64), device="cuda", dtype=torch.bfloat16),
            TOP_K,
            w1_scale=_scales(g, 1, 128, k_packed // 16),
            a1_scale=torch.empty(
                (64, k_packed // 16), device="cuda", dtype=torch.uint8
            ),
            block_m=64,
            b_preshuffled=True,
            b_layout=prefill_stage1.B_LAYOUT_GDOT128.value,
            num_warps=2,
            wide_tile=False,
        )


def _package_bank(g, hidden: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Package-prefill (GDOT128) gate/up weights and scales of random experts."""
    w = _experts(
        _codes(g, EXPERTS, 2 * INTER, hidden // 2),
        _scales(g, EXPERTS, 2 * INTER, hidden // 32),
        hidden=hidden,
    )
    tensor = w.w13_weight_triton_tensor
    return (
        tensor.gluon_package_prefill_weight.view(torch.uint8),
        tensor.gluon_package_prefill_scale.view(torch.uint8),
    )


def _stage1(
    w13,
    w13_scale,
    packed,
    sorted_scales,
    sorted_ids,
    expert_ids,
    valid,
    block_m: int,
    wide_tile: bool,
    activation: dict,
    tokens: int,
):
    """Stage 1's sorted rows and scale panels, in buffers pre-filled with
    sentinels so a row one kernel writes and the other skips also differs."""
    em = sorted_ids.shape[0]
    data = torch.full((em, INTER // 2), 0xA5, device="cuda", dtype=torch.uint8)
    out_scales = torch.full(
        ((em + 31) // 32 * 32, INTER // 32), 0x5A, device="cuda", dtype=torch.uint8
    )
    prefill_stage1.invoke_gluon_mxfp4_moe_stage1(
        packed,
        w13,
        None,
        sorted_ids,
        expert_ids,
        valid,
        data,
        TOP_K,
        w1_scale=w13_scale,
        a1_scale=sorted_scales,
        block_m=block_m,
        num_warps=block_m // 32,
        wide_tile=wide_tile,
        b_preshuffled=True,
        b_layout=prefill_stage1.B_LAYOUT_GDOT128.value,
        output_sorted=True,
        output_quantized=True,
        out_scale=out_scales,
        output_k=INTER,
        source_tokens=tokens,
        **activation,
    )
    return data, out_scales


def _routing(rows: int, routing: str, g):
    if routing == "uniform":
        ids = torch.rand(rows, EXPERTS, device="cuda", generator=g).topk(TOP_K, -1)
        ids = ids.indices
    else:
        p = 1.0 / torch.arange(1, EXPERTS + 1, device="cuda").pow(1.1)
        ids = torch.multinomial(p.expand(rows, EXPERTS), TOP_K, generator=g)
    weights = torch.rand(rows, TOP_K, device="cuda", generator=g)
    return ids.to(torch.int32).contiguous(), weights


def _sort(ids, weights, hidden: int, block_m: int):
    """The sort's routes, its route weights, block experts and valid count."""
    sorted_ids, sorted_weights, expert_ids, valid, _ = moe_sorting.gluon_moe_sorting(
        ids,
        weights,
        EXPERTS,
        hidden,
        torch.bfloat16,
        block_m,
        use_minimum_route_chunks=False,
        runtime_rows=False,
        deterministic_order=True,
    )
    return sorted_ids, sorted_weights, expert_ids, valid


def _sorted_stage1_inputs(rows: int, hidden: int, block_m: int, routing: str):
    g = torch.Generator(device="cuda").manual_seed(rows + block_m)
    packed = _codes(g, rows, hidden // 2)
    scales = empty_swizzled_cdna4_mxfp4_scale(rows, hidden // 32, device="cuda")
    scales.copy_(_scales(g, *scales.shape))
    ids, weights = _routing(rows, routing, g)
    sorted_ids, _, expert_ids, valid = _sort(ids, weights, hidden, block_m)
    sorted_scales = scale.gather_package_cdna4_scale(
        scales,
        sorted_ids,
        source_rows=rows,
        cols=hidden,
        top_k=TOP_K,
        flatten_topk=False,
    )
    return packed, sorted_scales, sorted_ids, expert_ids, valid


_SILU = {}
# A clamped SwiGLU with a linear term, and SiTU, exercise the other epilogue
# branches.
_CLAMPED = {"swiglu_alpha": 1.702, "swiglu_limit": 7.0, "swiglu_beta": 1.0}
_SITU = {"swiglu_alpha": 1.0, "situ_linear_beta": 4.0}


@pytest.mark.parametrize("routing", ["uniform", "zipf"])
@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize(
    "rows",
    [
        200,  # small round: most blocks end in padding
        1031,  # ragged tail
        4099,
        8193,  # past 8,192 rows, where serving takes 128-row sort blocks
        16391,  # 16k round with a ragged tail
    ],
)
def test_wide_stage1_tile_writes_the_general_tile_bits(
    rows: int, block_m: int, routing: str
) -> None:
    """The wide stage-1 tile writes every sorted row and scale bit for bit as
    the general kernel does, over 64- and 128-row sort blocks."""
    g = torch.Generator(device="cuda").manual_seed(31)
    w13, w13_scale = _package_bank(g, HIDDEN)
    inputs = _sorted_stage1_inputs(rows, HIDDEN, block_m, routing)
    valid = int(inputs[-1][0])
    assert valid > 0
    general = _stage1(w13, w13_scale, *inputs, block_m, False, _SILU, rows)
    wide = _stage1(w13, w13_scale, *inputs, block_m, True, _SILU, rows)
    # The general kernel wrote every valid row.
    assert not torch.all(general[0][:valid] == 0xA5, dim=1).any()
    assert torch.equal(wide[0], general[0])
    assert torch.equal(wide[1], general[1])


@pytest.mark.parametrize(
    "hidden",
    [
        512,  # two K tiles: no steady-state loop
        1280,  # five K tiles: an odd steady count takes the single-tile step
        2048,
    ],
)
@pytest.mark.parametrize("activation", [_SILU, _CLAMPED, _SITU])
def test_wide_stage1_tile_bits_hold_across_k_tiles_and_activations(
    hidden: int, activation: dict
) -> None:
    g = torch.Generator(device="cuda").manual_seed(hidden)
    w13, w13_scale = _package_bank(g, hidden)
    rows = 777
    inputs = _sorted_stage1_inputs(rows, hidden, 64, "zipf")
    general = _stage1(w13, w13_scale, *inputs, 64, False, activation, rows)
    wide = _stage1(w13, w13_scale, *inputs, 64, True, activation, rows)
    assert torch.equal(wide[0], general[0])
    assert torch.equal(wide[1], general[1])


def _expert_storage(t: torch.Tensor) -> torch.Tensor:
    """``[experts, bytes]`` storage under a package alias (a strided view of
    swizzled storage, so its logical elements must not be copied one by one)."""
    return t.as_strided((t.shape[0], t.stride(0)), (t.stride(0), 1))


def _bank_of(t: torch.Tensor, experts: int, slot: int, expert: int):
    """A package alias of ``experts`` experts whose ``slot`` holds ``t``'s
    ``expert``; the other slots are left uninitialized."""
    raw = torch.empty((experts, t.stride(0)), device="cuda", dtype=torch.uint8)
    raw[slot].copy_(_expert_storage(t)[expert])
    return raw.as_strided((experts, *t.shape[1:]), t.stride())


def test_wide_stage1_tile_reads_an_expert_past_2_31_bytes() -> None:
    """Expert offsets are 64-bit: the last expert of the bank starts at least
    2^31 bytes into the weights (about 2.1 GB, only that expert written and
    routed). Its rows must equal those of the same expert in a one-expert
    bank, from both stage-1 kernels."""
    g = torch.Generator(device="cuda").manual_seed(41)
    w13, w13_scale = _package_bank(g, HIDDEN)
    far = -(-(2**31) // w13.stride(0))
    big = _bank_of(w13, far + 1, far, 3)
    big_scale = _bank_of(w13_scale, far + 1, far, 3)
    small = _bank_of(w13, 1, 0, 3)
    small_scale = _bank_of(w13_scale, 1, 0, 3)
    assert far * big.stride(0) >= 2**31

    rows, block_m = 200, 128
    blocks = (rows + block_m - 1) // block_m
    tokens = torch.arange(blocks * block_m, device="cuda", dtype=torch.int32)
    # One route per token in slot 0; the tail block pads with token ``rows``.
    sorted_ids = torch.where(tokens < rows, tokens, rows).to(torch.int32)
    valid = torch.tensor([blocks * block_m], device="cuda", dtype=torch.int32)
    packed = _codes(g, rows, HIDDEN // 2)
    sorted_scales = torch.zeros(
        (blocks * block_m, HIDDEN // 32), device="cuda", dtype=torch.uint8
    )
    sorted_scales[:rows].copy_(_scales(g, rows, HIDDEN // 32))

    def run(weights, scales, expert, wide_tile):
        expert_ids = torch.full((blocks,), expert, device="cuda", dtype=torch.int32)
        return _stage1(
            weights,
            scales,
            packed,
            sorted_scales,
            sorted_ids,
            expert_ids,
            valid,
            block_m,
            wide_tile,
            _SILU,
            rows,
        )

    reference = run(small, small_scale, 0, False)
    for wide_tile in (False, True):
        far_rows = run(big, big_scale, far, wide_tile)
        assert torch.equal(far_rows[0], reference[0])
        assert torch.equal(far_rows[1], reference[1])
    del big, big_scale
    torch.cuda.empty_cache()


def test_wide_stage1_tile_compiles_once_per_sort_block() -> None:
    """Row, route and token counts are runtime values of the wide tile."""
    g = torch.Generator(device="cuda").manual_seed(43)
    w13, w13_scale = _package_bank(g, HIDDEN)

    def run(rows, block_m):
        inputs = _sorted_stage1_inputs(rows, HIDDEN, block_m, "uniform")
        _stage1(w13, w13_scale, *inputs, block_m, True, _SILU, rows)

    for block_m in (64, 128):
        run(300, block_m)
    with assert_no_triton_compile(
        prefill_stage1.gluon_mxfp4_moe_stage1_e2m1_wide_gfx950
    ):
        for block_m in (64, 128):
            for rows in (192, 199, 256, 1025, 4097, 8192, 8193, 12001):
                run(rows, block_m)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "block_m, num_warps, gdot128",
    [(32, 1, True), (64, 4, True), (64, 2, False)],
)
def test_wide_stage1_tile_rejects_what_it_does_not_serve(
    block_m: int, num_warps: int, gdot128: bool
) -> None:
    g = torch.Generator(device="cuda").manual_seed(5)
    rows = 256
    with pytest.raises(ValueError, match="wide stage-1 tile"):
        prefill_stage1.invoke_gluon_mxfp4_moe_stage1(
            _codes(g, rows, HIDDEN // 2),
            _codes(g, 1, 2 * INTER, HIDDEN // 2),
            None,
            torch.zeros(rows, device="cuda", dtype=torch.int32),
            torch.zeros(rows // block_m, device="cuda", dtype=torch.int32),
            torch.tensor([rows], device="cuda", dtype=torch.int32),
            torch.empty((rows, INTER // 2), device="cuda", dtype=torch.uint8),
            TOP_K,
            w1_scale=_scales(g, 1, 2 * INTER, HIDDEN // 32),
            a1_scale=torch.empty(
                (rows, HIDDEN // 32), device="cuda", dtype=torch.uint8
            ),
            block_m=block_m,
            b_preshuffled=True,
            b_layout=(
                prefill_stage1.B_LAYOUT_GDOT128
                if gdot128
                else prefill_stage1.B_LAYOUT_MFMA16
            ).value,
            output_sorted=True,
            output_quantized=True,
            out_scale=torch.empty(
                (rows, INTER // 32), device="cuda", dtype=torch.uint8
            ),
            num_warps=num_warps,
            wide_tile=True,
        )


def _down_bank(g, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Package-prefill (GDOT128) down weights and scales of random experts."""
    weight = prepare_moe_sorted_down_weight(
        _codes(g, EXPERTS, HIDDEN, k // 2), _scales(g, EXPERTS, HIDDEN, k // 32)
    )
    return (
        weight.gluon_package_prefill_weight.view(torch.uint8),
        weight.gluon_package_prefill_scale.view(torch.uint8),
    )


def _stage2(
    w2,
    w2_scale,
    inter,
    inter_scale,
    sorted_ids,
    sorted_weights,
    expert_ids,
    valid,
    tokens: int,
    block_m: int,
    sort_block_m: int,
    wide_tile: bool,
    do_finalize: bool,
):
    """Stage 2's per-route BF16 rows (``do_finalize=False``, filled with a
    sentinel first so an unwritten route differs), or the reduced token rows
    with routed weights."""
    deferred = None
    if do_finalize:
        out = torch.empty((tokens, HIDDEN), device="cuda", dtype=torch.bfloat16)
    else:
        deferred = torch.full(
            (tokens * TOP_K, HIDDEN), -7.0, device="cuda", dtype=torch.bfloat16
        )
        out = deferred.view(tokens, TOP_K * HIDDEN)[:, :HIDDEN]
    result = prefill_stage2.invoke_gluon_mxfp4_moe_stage2_1x2(
        inter,
        None,
        w2,
        sorted_ids,
        expert_ids,
        valid,
        out,
        TOP_K,
        do_finalize,
        initialize_unowned_routes=False,
        deferred_out=deferred,
        w2_scale=w2_scale,
        a2_scale=inter_scale,
        sorted_weights=sorted_weights,
        b_preshuffled=True,
        b_layout=prefill_stage1.B_LAYOUT_GDOT128.value,
        block_m=block_m,
        sort_block_m=sort_block_m,
        force_reduce=True,
        input_sorted=True,
        a_format="e2m1",
        wide_tile=wide_tile,
    )
    return result.view(torch.int16)


# The BF16 bits ``_stage2`` fills unwritten routes with.
_SENTINEL_BITS = int(torch.tensor(-7.0, dtype=torch.bfloat16).view(torch.int16))


def _sorted_stage2_inputs(rows: int, k: int, sort_block_m: int, routing: str):
    """Routes sorted into ``sort_block_m``-row blocks and random sorted E2M1
    rows and scale panels standing for stage 1's output."""
    g = torch.Generator(device="cuda").manual_seed(rows + k)
    ids, weights = _routing(rows, routing, g)
    sorted_ids, sorted_weights, expert_ids, valid = _sort(
        ids, weights, HIDDEN, sort_block_m
    )
    em = sorted_ids.shape[0]
    inter = _codes(g, em, k // 2)
    inter_scale = _scales(g, (em + 31) // 32 * 32, k // 32)
    return inter, inter_scale, sorted_ids, sorted_weights, expert_ids, valid


@pytest.mark.parametrize("routing", ["uniform", "zipf"])
@pytest.mark.parametrize("sort_block_m", [64, 128])
@pytest.mark.parametrize("rows", [200, 1031, 3072, 8193, 16391, 24576])
def test_wide_stage2_tile_writes_the_general_tile_bits(
    rows: int, sort_block_m: int, routing: str
) -> None:
    """On the same sort, the wide stage-2 tile writes every route's BF16 row
    bit for bit as the general tile on the sort's block height does."""
    g = torch.Generator(device="cuda").manual_seed(51)
    w2, w2_scale = _down_bank(g, INTER)
    inputs = _sorted_stage2_inputs(rows, INTER, sort_block_m, routing)
    general = _stage2(
        w2, w2_scale, *inputs, rows, sort_block_m, sort_block_m, False, False
    )
    wide = _stage2(w2, w2_scale, *inputs, rows, 64, sort_block_m, True, False)
    # Every route is local, so the general tile wrote every row.
    assert not (general == _SENTINEL_BITS).all(dim=1).any()
    assert torch.equal(wide, general)


@pytest.mark.parametrize(
    "k",
    [
        512,  # two K tiles: no steady-state loop
        1280,  # five K tiles: an odd steady count takes the single-tile step
        2048,
    ],
)
@pytest.mark.parametrize("do_finalize", [False, True])
def test_wide_stage2_tile_bits_hold_across_k_tiles_and_finalize(
    k: int, do_finalize: bool
) -> None:
    """Also with routed weights applied and the topk reduction (finalize)."""
    g = torch.Generator(device="cuda").manual_seed(k)
    w2, w2_scale = _down_bank(g, k)
    rows = 777
    inputs = _sorted_stage2_inputs(rows, k, 64, "zipf")
    general = _stage2(w2, w2_scale, *inputs, rows, 64, 64, False, do_finalize)
    wide = _stage2(w2, w2_scale, *inputs, rows, 64, 64, True, do_finalize)
    assert torch.equal(wide, general)


def _by_route(routes, data, scales, count: int) -> torch.Tensor:
    """Stage 1's packed row and scales of each route, keyed token * TOP_K + slot."""
    valid = int(routes.num_valid_ids[0])
    packed_ids = routes.sorted_ids[:valid]
    token, slot = packed_ids & 0xFFFFFF, packed_ids >> 24
    real = token < count
    order = (token * TOP_K + slot)[real].long()
    rows = torch.zeros(
        (count * TOP_K, INTER // 2 + INTER // 32), device="cuda", dtype=torch.uint8
    )
    rows[order, : INTER // 2] = data[:valid][real]
    rows[order, INTER // 2 :] = _logical_scales(scales, valid, INTER // 32)[real]
    return rows


@pytest.mark.parametrize("routing", ["uniform", "zipf"])
@pytest.mark.parametrize("rows", [200, 1031, 3072, 8193, 16391, 24576])
def test_wide_tiles_write_every_route_as_the_general_kernels(
    monkeypatch, rows: int, routing: str
) -> None:
    """A grouped configuration with ``wide_tiles`` moves both stages to 64-row
    sort blocks and their wide tiles; each route's intermediate and
    down-projection row are the same bits as the general kernels write on the
    density rule's blocks (128 rows past 1,280 tokens here)."""
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import tuning

    g = torch.Generator(device="cuda").manual_seed(61)
    w = _experts(
        _codes(g, EXPERTS, 2 * INTER, HIDDEN // 2),
        _scales(g, EXPERTS, 2 * INTER, HIDDEN // 32),
    )
    down = prepare_moe_sorted_down_weight(
        _codes(g, EXPERTS, HIDDEN, INTER // 2), _scales(g, EXPERTS, HIDDEN, INTER // 32)
    )
    x = (
        _codes(g, rows, HIDDEN // 2),
        empty_swizzled_cdna4_mxfp4_scale(rows, HIDDEN // 32, device="cuda"),
    )
    x[1].copy_(_scales(g, *x[1].shape))
    ids, weights = _routing(rows, routing, g)

    def layer():
        capacity = sorted_route_capacity(rows, EXPERTS, TOP_K)
        data = torch.zeros((capacity, INTER // 2), device="cuda", dtype=torch.uint8)
        scales = torch.zeros(
            ((capacity + 31) // 32 * 32, INTER // 32), device="cuda", dtype=torch.uint8
        )
        routes = moe_sorted_gate_up(x, weights, ids, w, data, scales)
        deferred = torch.full(
            (rows * TOP_K, HIDDEN), -7.0, device="cuda", dtype=torch.bfloat16
        )
        moe_sorted_down(data, scales, routes, down, deferred)
        return routes.block_m, _by_route(routes, data, scales, rows), deferred

    today_block, today_inter, today_out = layer()
    # This test shape has no table entry; give it the wide rows a tuned shape
    # (Muse Spark's) has.
    wide = tuning._GroupedMoEConfig(
        64,
        2,
        use_minimum_route_chunks=False,
        runtime_rows=False,
        wide_tiles=True,
    )
    key = (HIDDEN, INTER, INTER, EXPERTS, EXPERTS, TOP_K)
    monkeypatch.setitem(tuning._GROUPED_MOE_CONFIGS, key, ((1, 1 << 20, wide),))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        block, inter, out = layer()
        torch.cuda.synchronize()
    launched = {event.name for event in prof.events()}
    # Both stages ran their wide kernels, not a silent general-kernel fallback.
    for name in (
        "gluon_mxfp4_moe_stage1_e2m1_wide_gfx950",
        "gluon_mxfp4_moe_stage2_e2m1_wide_gfx950",
    ):
        assert any(event.startswith(name) for event in launched), name
    assert block == 64
    assert today_block == (128 if rows > 1280 else 64)
    assert torch.equal(inter, today_inter)
    assert torch.equal(out.view(torch.int16), today_out.view(torch.int16))


def test_wide_stage2_tile_reads_an_expert_past_2_31_bytes() -> None:
    """The last expert's down weights start at least 2^31 bytes in (about
    2.1 GB, only that expert written and routed); its rows equal those of
    the same expert in a one-expert bank."""
    g = torch.Generator(device="cuda").manual_seed(71)
    w2, w2_scale = _down_bank(g, INTER)
    far = -(-(2**31) // w2.stride(0))
    big = _bank_of(w2, far + 1, far, 5)
    big_scale = _bank_of(w2_scale, far + 1, far, 5)
    small = _bank_of(w2, 1, 0, 5)
    small_scale = _bank_of(w2_scale, 1, 0, 5)
    assert far * big.stride(0) >= 2**31

    rows, block_m = 200, 64
    blocks = (rows + block_m - 1) // block_m
    slots = torch.arange(blocks * block_m, device="cuda", dtype=torch.int32)
    # One route per token in slot 0; the tail block pads with token ``rows``.
    sorted_ids = torch.where(slots < rows, slots, rows).to(torch.int32)
    sorted_weights = torch.rand(blocks * block_m, device="cuda", generator=g)
    valid = torch.tensor([blocks * block_m], device="cuda", dtype=torch.int32)
    inter = _codes(g, blocks * block_m, INTER // 2)
    inter_scale = _scales(g, blocks * block_m, INTER // 32)

    def run(weights, scales, expert, wide_tile):
        expert_ids = torch.full((blocks,), expert, device="cuda", dtype=torch.int32)
        return _stage2(
            weights,
            scales,
            inter,
            inter_scale,
            sorted_ids,
            sorted_weights,
            expert_ids,
            valid,
            rows,
            block_m,
            block_m,
            wide_tile,
            False,
        )

    reference = run(small, small_scale, 0, False)
    for wide_tile in (False, True):
        assert torch.equal(run(big, big_scale, far, wide_tile), reference)
    del big, big_scale
    torch.cuda.empty_cache()


def test_wide_stage2_tile_compiles_once_per_sort_block() -> None:
    """Row, route and token counts are runtime values of the wide stage-2 tile."""
    g = torch.Generator(device="cuda").manual_seed(73)
    w2, w2_scale = _down_bank(g, INTER)

    def run(rows, sort_block_m):
        inputs = _sorted_stage2_inputs(rows, INTER, sort_block_m, "uniform")
        _stage2(w2, w2_scale, *inputs, rows, 64, sort_block_m, True, False)

    for sort_block_m in (64, 128):
        run(300, sort_block_m)
    with assert_no_triton_compile(
        prefill_stage2.gluon_mxfp4_moe_stage2_e2m1_wide_gfx950
    ):
        for sort_block_m in (64, 128):
            for rows in (192, 199, 256, 1025, 4097, 8192, 8193, 12001):
                run(rows, sort_block_m)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "block_m, do_finalize, force_reduce",
    [
        (128, False, True),  # the wide tile is 64 rows
        (64, True, False),  # atomic combine, not reduce mode
    ],
)
def test_wide_stage2_tile_rejects_what_it_does_not_serve(
    block_m: int, do_finalize: bool, force_reduce: bool
) -> None:
    g = torch.Generator(device="cuda").manual_seed(75)
    w2, w2_scale = _down_bank(g, INTER)
    rows = 256
    inter, inter_scale, sorted_ids, sorted_weights, expert_ids, valid = (
        _sorted_stage2_inputs(rows, INTER, 128, "uniform")
    )
    with pytest.raises(ValueError, match="wide stage-2 tile"):
        prefill_stage2.invoke_gluon_mxfp4_moe_stage2_1x2(
            inter,
            None,
            w2,
            sorted_ids,
            expert_ids,
            valid,
            torch.empty((rows, HIDDEN), device="cuda", dtype=torch.bfloat16),
            TOP_K,
            do_finalize,
            initialize_unowned_routes=False,
            w2_scale=w2_scale,
            a2_scale=inter_scale,
            sorted_weights=sorted_weights,
            b_preshuffled=True,
            b_layout=prefill_stage1.B_LAYOUT_GDOT128.value,
            block_m=block_m,
            sort_block_m=128,
            force_reduce=force_reduce,
            input_sorted=True,
            a_format="e2m1",
            wide_tile=True,
        )


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


def test_wide_stage1_tile_writes_sorted_rows_past_2_31_bytes() -> None:
    """Stage 1's sorted output is addressed in 64 bits: with enough sorted
    rows that the last 64-row block starts at least 2^31 bytes into the
    intermediate (about 2.1 GB), that block's rows and scale panel equal a
    one-block run on the same tokens and expert."""
    g = torch.Generator(device="cuda").manual_seed(91)
    w13, w13_scale = _package_bank(g, HIDDEN)
    block_m, tokens = 64, 64
    blocks = -(-(2**31) // (block_m * (INTER // 2))) + 1
    em = blocks * block_m
    assert (em - block_m) * (INTER // 2) >= 2**31
    packed = _codes(g, tokens, HIDDEN // 2)
    token_scales = _scales(g, tokens, HIDDEN // 32)

    def run(rows):
        # Every block repeats the same tokens in slot 0 on expert 3.
        sorted_ids = torch.arange(rows, device="cuda", dtype=torch.int32) % tokens
        sorted_scales = token_scales.repeat(rows // tokens, 1)
        experts = torch.full((rows // block_m,), 3, device="cuda", dtype=torch.int32)
        valid = torch.tensor([rows], device="cuda", dtype=torch.int32)
        return _stage1(
            w13,
            w13_scale,
            packed,
            sorted_scales,
            sorted_ids,
            experts,
            valid,
            block_m,
            True,
            _SILU,
            tokens,
        )

    ref_data, ref_scales = run(block_m)
    data, scales = run(em)
    assert torch.equal(data[em - block_m :], ref_data)
    assert torch.equal(scales[em - block_m : em], ref_scales[:block_m])
    del data, scales
    torch.cuda.empty_cache()


def test_wide_stage2_tile_reads_sorted_rows_past_2_31_bytes() -> None:
    """Stage 2 reads its sorted input and scale panels from a 64-bit
    per-tile base: only the last 64-row block, starting at least 2^31 bytes
    into the intermediate (about 2.1 GB), has an expert, and its routes equal
    a one-block run on the same rows."""
    g = torch.Generator(device="cuda").manual_seed(93)
    w2, w2_scale = _down_bank(g, INTER)
    block_m, tokens = 64, 64
    blocks = -(-(2**31) // (block_m * (INTER // 2))) + 1
    em = blocks * block_m
    assert (em - block_m) * (INTER // 2) >= 2**31
    last = _codes(g, block_m, INTER // 2)
    last_scale = _scales(g, block_m, INTER // 32)
    last_weights = torch.rand(block_m, device="cuda", generator=g)

    def run(rows):
        sorted_ids = torch.arange(rows, device="cuda", dtype=torch.int32) % tokens
        sorted_weights = torch.zeros(rows, device="cuda")
        sorted_weights[rows - block_m :] = last_weights
        # Blocks without a local expert (-1) are skipped; only the last runs.
        experts = torch.full((rows // block_m,), -1, device="cuda", dtype=torch.int32)
        experts[-1] = 5
        inter = torch.empty((rows, INTER // 2), device="cuda", dtype=torch.uint8)
        inter[rows - block_m :] = last
        inter_scale = torch.empty((rows, INTER // 32), device="cuda", dtype=torch.uint8)
        inter_scale[rows - block_m :] = last_scale
        valid = torch.tensor([rows], device="cuda", dtype=torch.int32)
        return _stage2(
            w2,
            w2_scale,
            inter,
            inter_scale,
            sorted_ids,
            sorted_weights,
            experts,
            valid,
            tokens,
            block_m,
            block_m,
            True,
            False,
        )

    reference = run(block_m)
    assert torch.equal(run(em), reference)
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "b_layout",
    [prefill_stage1.B_LAYOUT_GDOT128.value, prefill_stage1.B_LAYOUT_N16.value],
)
@pytest.mark.parametrize("input_sorted", [False, True])
def test_wide_tiles_serve_only_gdot128_banks_from_token_order(
    b_layout: int, input_sorted: bool
) -> None:
    """A configuration asking for the wide tiles gets them only for calls the
    wide kernels serve: GDOT128 banks, and token-order rows in stage 1. An
    N16 bank or sorted stage-1 rows keep the general kernels instead of
    reaching a launcher that rejects them."""
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import tuning

    gdot128 = b_layout == prefill_stage1.B_LAYOUT_GDOT128.value
    assert tuning._stage1_wide_tile(
        True, 64, 4096, 4096, "e2m1", b_layout, input_sorted
    ) == (gdot128 and not input_sorted)
    assert (
        tuning._stage2_wide_tile(True, 64, 4096, 4096, "e2m1", b_layout, reduces=True)
        == gdot128
    )
