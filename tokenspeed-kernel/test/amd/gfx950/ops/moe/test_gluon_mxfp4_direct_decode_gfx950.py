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

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon MXFP4 direct decode tests",
        allow_module_level=True,
    )


from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_warmup  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage1 import (  # noqa: E402
    _stage1_mxfp4_direct_mfma_gluon,
    gluon_mxfp4_moe_stage1_route_tiles_gfx950,
    invoke_stage1_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage2 import (  # noqa: E402
    _stage2_mxfp4_direct_mfma_gluon,
    gluon_mxfp4_moe_stage2_route_tiles_gfx950,
    invoke_stage2_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _quantize_mxfp4_activation,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (  # noqa: E402
    _max_padded_route_capacity,
    _moe_sorting_stage1_kernel,
    _moe_sorting_stage2_kernel,
    _moe_sorting_stage3_kernel,
    _moe_sorting_stage4_kernel,
    gluon_moe_sorting,
    gluon_mxfp4_moe_route_descriptors_gfx950,
    gluon_mxfp4_moe_route_pack_gfx950,
    gluon_mxfp4_moe_route_tiles_gfx950,
    gluon_mxfp4_moe_sort_histogram_prefix_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_stage1 import (  # noqa: E402
    gluon_mxfp4_moe_stage1_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_stage2 import (  # noqa: E402
    gluon_mxfp4_moe_stage2_1x2_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (  # noqa: E402
    _local_route_map_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale import (  # noqa: E402
    _gather_package_cdna4_scale_kernel,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (  # noqa: E402
    empty_swizzled_cdna4_mxfp4_scale,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)

NUM_EXPERTS = 4
HIDDEN = 512
INTERMEDIATE = 256
TOPK = 2
# Inputs dequantize exactly; leave only a small margin above BF16's worst-case
# relative rounding error (2**-8) when comparing against the FP32 reference.
_BF16_ATOL = 4e-3
_BF16_RTOL = 4e-3
# ``_E2M1_VALUES`` without the +-6 endpoint, which every group carries anyway.
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _exact_mxfp4_activation(
    rows: int, cols: int, generator: torch.Generator
) -> torch.Tensor:
    """Build a bf16 tensor that MXFP4-quantizes losslessly at unit scale.

    Every 32-element group is drawn from the E2M1 grid and pinned to an
    amax of 6.0, so the quantizer must select ``2**0`` and the payload
    round-trips exactly.  That keeps activation quantization out of the
    tolerance budget for the reference comparison below.
    """
    grid = torch.tensor(_E2M1_GRID, device="cuda")
    codes = torch.randint(
        0, len(_E2M1_GRID), (rows, cols), device="cuda", generator=generator
    )
    signs = (
        torch.randint(0, 2, (rows, cols), device="cuda", generator=generator) * 2 - 1
    )
    values = grid[codes] * signs
    values = values.view(rows, -1, 32)
    values[:, :, 0] = 6.0
    return values.view(rows, cols).to(torch.bfloat16)


def _assert_lossless(reference: torch.Tensor, packed: torch.Tensor) -> None:
    """Fail loudly if the quantizer did not pick the assumed unit scale."""
    unit_scale = torch.full(
        (packed.shape[0], packed.shape[1] * 2 // 32),
        127,
        dtype=torch.uint8,
        device=packed.device,
    )
    torch.testing.assert_close(
        dequantize_mxfp4(packed, unit_scale).to(torch.bfloat16),
        reference,
        atol=0,
        rtol=0,
    )


def _make_runtime_weights(
    generator: torch.Generator,
    *,
    num_experts: int = NUM_EXPERTS,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    scale: int = 127,
) -> tuple[torch.Tensor, ...]:
    """Random MXFP4 experts, both raw (for the reference) and gdot128-shuffled."""

    def payload(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=generator
        )

    def unit_scales(*shape: int) -> torch.Tensor:
        return torch.full(shape, scale, dtype=torch.uint8, device="cuda")

    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    raw_w13 = payload(num_experts, 2 * intermediate_size, hidden_size // 2)
    raw_w2 = payload(num_experts, hidden_size, intermediate_size // 2)
    raw_w13_scale = unit_scales(num_experts, 2 * intermediate_size, hidden_size // 32)
    raw_w2_scale = unit_scales(num_experts, hidden_size, intermediate_size // 32)
    module.w13_weight = torch.nn.Parameter(raw_w13.clone(), requires_grad=False)
    module.w13_weight_scale = torch.nn.Parameter(
        raw_w13_scale.clone(), requires_grad=False
    )
    module.w2_weight = torch.nn.Parameter(raw_w2.clone(), requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(
        raw_w2_scale.clone(), requires_grad=False
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)

    return (
        raw_w13,
        raw_w13_scale,
        raw_w2,
        raw_w2_scale,
        _extract_gluon_raw_w(module.w13_weight_triton_tensor),
        _extract_gluon_raw_s(module.w13_precision_config.b_mx_scale),
        _extract_gluon_raw_w(module.w2_weight_triton_tensor),
        _extract_gluon_raw_s(module.w2_precision_config.b_mx_scale),
    )


def _round_robin_topk(num_tokens: int) -> torch.Tensor:
    ids = (
        torch.arange(num_tokens, device="cuda")[:, None]
        + torch.arange(TOPK, device="cuda")
    ) % NUM_EXPERTS
    return ids.to(torch.int32)


def _logical_cdna4_scales(
    scales: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Read only the logical rows from a padded CDNA4 scale allocation."""
    logical_m = torch.arange(rows, device=scales.device)[:, None]
    logical_k = torch.arange(cols, device=scales.device)[None, :]
    m_in_block = logical_m % 32
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = logical_k // 8
    k_hi = (logical_k % 8) // 4
    k_lo = logical_k % 4
    swizzled_k = (((k_block * 4 + k_lo) * 16 + m_lo) * 2 + k_hi) * 2 + m_hi
    offsets = swizzled_k * scales.stride(0) + (logical_m // 32) * scales.stride(1)
    # Logical flattening interleaves the separate 32-row storage blocks.
    physical = scales.as_strided((scales.numel(),), (1,))
    return physical[offsets]


@pytest.mark.parametrize("num_tokens", [1, 2])
@pytest.mark.parametrize("block_n", [16, 32])
def test_direct_mfma_decode_stage1_matches_reference(
    num_tokens: int, block_n: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260809 + num_tokens)
    raw_w13, raw_w13_scale, _, _, w13, w13_scale, _, _ = _make_runtime_weights(
        generator
    )

    hidden = _exact_mxfp4_activation(num_tokens, HIDDEN, generator)
    topk_ids = _round_robin_topk(num_tokens)
    hidden_mxfp4, hidden_scale = _quantize_mxfp4_activation(hidden)
    _assert_lossless(hidden, hidden_mxfp4)

    actual = torch.empty(
        (num_tokens * TOPK, INTERMEDIATE), dtype=torch.bfloat16, device="cuda"
    )
    invoke_stage1_mxfp4_mfma_decode_gluon(
        hidden_mxfp4,
        hidden_scale,
        w13,
        w13_scale,
        topk_ids,
        actual,
        TOPK,
        0,
        None,
        BLOCK_N=block_n,
    )
    torch.cuda.synchronize()

    alpha, limit, beta = 1.702, 7.0, 1.0
    expected = torch.empty_like(actual, dtype=torch.float32)
    for token in range(num_tokens):
        for slot in range(TOPK):
            expert = int(topk_ids[token, slot])
            w = dequantize_mxfp4(raw_w13[expert], raw_w13_scale[expert])
            acc = hidden[token].float() @ w.t()
            gate = acc[0::2].clamp(max=limit)
            linear = acc[1::2].clamp(-limit, limit)
            silu = gate / (1.0 + torch.exp(-alpha * gate))
            expected[token * TOPK + slot] = silu * (linear + beta)

    torch.testing.assert_close(
        actual.float(), expected, atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_direct_stage1_fused_quantization_matches_staged(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260916 + num_tokens)
    _, _, _, _, w13, w13_scale, _, _ = _make_runtime_weights(generator)
    hidden = _exact_mxfp4_activation(num_tokens, HIDDEN, generator)
    topk_ids = _round_robin_topk(num_tokens)
    hidden_mxfp4, hidden_scale = _quantize_mxfp4_activation(hidden)

    staged = torch.empty(
        (num_tokens * TOPK, INTERMEDIATE),
        dtype=torch.bfloat16,
        device="cuda",
    )
    invoke_stage1_mxfp4_mfma_decode_gluon(
        hidden_mxfp4,
        hidden_scale,
        w13,
        w13_scale,
        topk_ids,
        staged,
        TOPK,
        0,
        None,
        BLOCK_N=64,
    )
    expected, expected_scale = _quantize_mxfp4_activation(staged)

    actual = torch.empty_like(expected)
    actual_scale = empty_swizzled_cdna4_mxfp4_scale(
        num_tokens * TOPK,
        INTERMEDIATE // 32,
        device=torch.device("cuda"),
    )
    invoke_stage1_mxfp4_mfma_decode_gluon(
        hidden_mxfp4,
        hidden_scale,
        w13,
        w13_scale,
        topk_ids,
        actual,
        TOPK,
        0,
        actual_scale,
        BLOCK_N=64,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    logical_rows = num_tokens * TOPK
    logical_cols = INTERMEDIATE // 32
    torch.testing.assert_close(
        _logical_cdna4_scales(actual_scale, logical_rows, logical_cols),
        _logical_cdna4_scales(expected_scale, logical_rows, logical_cols),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("num_tokens", [1, 2])
@pytest.mark.parametrize("pipeline_k", [True, False])
def test_direct_mfma_decode_stage2_matches_reference(
    num_tokens: int, pipeline_k: bool
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260810 + num_tokens)
    _, _, raw_w2, raw_w2_scale, _, _, w2, w2_scale = _make_runtime_weights(generator)

    inter = _exact_mxfp4_activation(num_tokens * TOPK, INTERMEDIATE, generator)
    topk_ids = _round_robin_topk(num_tokens)
    topk_weights = torch.rand(
        (num_tokens, TOPK), dtype=torch.float32, device="cuda", generator=generator
    )
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    inter_mxfp4, inter_scale = _quantize_mxfp4_activation(inter)
    _assert_lossless(inter, inter_mxfp4)

    actual = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    invoke_stage2_mxfp4_mfma_decode_gluon(
        inter_mxfp4,
        inter_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        actual,
        TOPK,
        0,
        True,
        None,
        BLOCK_N=16,
        PIPELINE_K=pipeline_k,
    )
    torch.cuda.synchronize()

    # The kernel epilogue rounds each expert partial to bf16 and applies the
    # routed weight in bf16 before reducing top-k; mirror that ordering.
    expected = torch.zeros((num_tokens, HIDDEN), dtype=torch.float32, device="cuda")
    for token in range(num_tokens):
        for slot in range(TOPK):
            expert = int(topk_ids[token, slot])
            w = dequantize_mxfp4(raw_w2[expert], raw_w2_scale[expert])
            partial = (inter[token * TOPK + slot].float() @ w.t()).to(torch.bfloat16)
            gate = topk_weights[token, slot].to(torch.bfloat16)
            expected[token] += (partial * gate).float()

    torch.testing.assert_close(
        actual.float(), expected, atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@pytest.mark.parametrize(
    ("block_n", "pipeline_k"),
    [(16, True), (32, False), (32, True)],
)
def test_direct_mfma_decode_deferred_ep_routes(
    num_tokens: int,
    block_n: int,
    pipeline_k: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260915 + num_tokens)
    _, _, _, _, w13, w13_scale, w2, w2_scale = _make_runtime_weights(generator)
    hidden = _exact_mxfp4_activation(num_tokens, HIDDEN, generator)
    hidden_mxfp4, hidden_scale = _quantize_mxfp4_activation(hidden)
    expert_start = NUM_EXPERTS
    local_ids = expert_start + torch.arange(num_tokens, device="cuda") % NUM_EXPERTS
    remote_ids = torch.arange(num_tokens, device="cuda") % NUM_EXPERTS
    topk_ids = torch.stack((local_ids, remote_ids), dim=1).to(torch.int32)
    topk_weights = torch.rand(
        (num_tokens, TOPK),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    intermediate = torch.zeros(
        (num_tokens * TOPK, INTERMEDIATE),
        dtype=torch.bfloat16,
        device="cuda",
    )
    invoke_stage1_mxfp4_mfma_decode_gluon(
        hidden_mxfp4,
        hidden_scale,
        w13,
        w13_scale,
        topk_ids,
        intermediate,
        TOPK,
        expert_start,
        None,
        BLOCK_N=32,
    )
    inter_mxfp4, inter_scale = _quantize_mxfp4_activation(intermediate)
    route_output = torch.zeros(
        (num_tokens * TOPK, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
    )
    expanded_idx = torch.empty((num_tokens * TOPK,), dtype=torch.int32, device="cuda")
    invoke_stage2_mxfp4_mfma_decode_gluon(
        inter_mxfp4,
        inter_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        route_output,
        TOPK,
        expert_start,
        False,
        expanded_idx,
        BLOCK_N=block_n,
        PIPELINE_K=pipeline_k,
    )
    finalized = torch.empty(
        (num_tokens, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
    )
    invoke_stage2_mxfp4_mfma_decode_gluon(
        inter_mxfp4,
        inter_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        finalized,
        TOPK,
        expert_start,
        True,
        None,
        BLOCK_N=block_n,
        PIPELINE_K=pipeline_k,
    )
    torch.cuda.synchronize()

    route_output = route_output.view(num_tokens, TOPK, HIDDEN)
    torch.testing.assert_close(
        route_output[:, 1],
        torch.zeros_like(route_output[:, 1]),
        atol=0,
        rtol=0,
    )
    expected_map = torch.stack(
        (
            torch.arange(num_tokens, device="cuda", dtype=torch.int32) * TOPK,
            torch.full((num_tokens,), -1, device="cuda", dtype=torch.int32),
        ),
        dim=1,
    ).flatten()
    torch.testing.assert_close(expanded_idx, expected_map, atol=0, rtol=0)
    weighted = (
        (route_output.to(torch.bfloat16) * topk_weights.to(torch.bfloat16)[:, :, None])
        .float()
        .sum(dim=1)
    )
    torch.testing.assert_close(
        finalized.float(), weighted, atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize("rows", [5, 19])
def test_route_tiled_gemms_preserve_route_slots(rows: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(41)
    _, _, _, _, w13, s13, w2, s2 = _make_runtime_weights(generator, scale=120)
    packed, scales = _quantize_mxfp4_activation(
        _exact_mxfp4_activation(rows, HIDDEN, generator)
    )
    expert_start = NUM_EXPERTS
    ids = (
        torch.arange(
            expert_start, expert_start + TOPK, device="cuda", dtype=torch.int32
        )
        .expand(rows, TOPK)
        .clone()
    )
    ids[-1, -1] += NUM_EXPERTS
    weights = torch.full((rows, TOPK), 1 / TOPK, device="cuda")
    intermediate = torch.empty(
        (rows * TOPK, INTERMEDIATE // 2), dtype=torch.uint8, device="cuda"
    )
    reference_intermediate = torch.empty_like(intermediate)
    intermediate_scales = empty_swizzled_cdna4_mxfp4_scale(
        rows * TOPK, INTERMEDIATE // 32, device="cuda"
    )
    reference_scales = torch.empty_like(intermediate_scales)
    output = torch.empty((rows * TOPK, HIDDEN), dtype=torch.bfloat16, device="cuda")
    reference = torch.empty_like(output)
    mapping = torch.empty(rows * TOPK, dtype=torch.int32, device="cuda")
    reference_mapping = torch.empty_like(mapping)
    max_rows = 16 if rows <= 16 else 32
    sorted_routes = torch.empty(
        NUM_EXPERTS * max_rows, dtype=torch.int32, device="cuda"
    )
    descriptors = torch.empty(
        (NUM_EXPERTS * (max_rows // 16), 3), dtype=torch.int32, device="cuda"
    )
    counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device="cuda")
    group_count = torch.empty(1, dtype=torch.int32, device="cuda")

    invoke_stage1_mxfp4_mfma_decode_gluon(
        packed,
        scales,
        w13,
        s13,
        ids,
        reference_intermediate,
        TOPK,
        expert_start,
        reference_scales,
        BLOCK_N=64,
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
        swiglu_beta=0.0,
    )
    invoke_stage2_mxfp4_mfma_decode_gluon(
        reference_intermediate,
        reference_scales,
        w2,
        s2,
        ids,
        weights,
        reference,
        TOPK,
        expert_start,
        False,
        reference_mapping,
        BLOCK_N=16,
        BLOCK_K=128,
        PIPELINE_K=False,
    )
    if rows <= 16:
        gluon_mxfp4_moe_route_descriptors_gfx950[(1,)](
            ids,
            sorted_routes,
            descriptors,
            group_count,
            mapping,
            rows,
            EXPERT_START=expert_start,
            NUM_EXPERTS=NUM_EXPERTS,
            TOPK=TOPK,
            MAX_ROWS=max_rows,
            num_warps=4,
        )
    else:
        gluon_mxfp4_moe_route_pack_gfx950[(NUM_EXPERTS,)](
            ids,
            sorted_routes,
            counts,
            mapping,
            rows,
            EXPERT_START=expert_start,
            NUM_EXPERTS=NUM_EXPERTS,
            TOPK=TOPK,
            MAX_ROWS=max_rows,
            num_warps=4,
        )
        gluon_mxfp4_moe_route_tiles_gfx950[(1,)](
            counts,
            descriptors,
            group_count,
            NUM_EXPERTS=NUM_EXPERTS,
            BLOCK_M=16,
            MAX_ROWS=max_rows,
            num_warps=4,
        )
    gluon_mxfp4_moe_stage1_route_tiles_gfx950[(64,)](
        packed,
        scales,
        w13,
        s13,
        sorted_routes,
        descriptors,
        group_count,
        intermediate,
        intermediate_scales,
        HIDDEN=HIDDEN,
        INTERMEDIATE=INTERMEDIATE,
        TOPK=TOPK,
        X_ROW=packed.stride(0),
        XS0=scales.stride(0),
        XS1=scales.stride(1),
        WE=w13.stride(0),
        SE=s13.stride(0),
        SS0=s13.stride(1),
        SS1=s13.stride(2),
        OS0=intermediate_scales.stride(0),
        OS1=intermediate_scales.stride(1),
        BLOCK_N=64,
        BLOCK_K=128,
        BLOCK_M=16,
        num_warps=4,
    )
    gluon_mxfp4_moe_stage2_route_tiles_gfx950[(64,)](
        intermediate,
        intermediate_scales,
        w2,
        s2,
        sorted_routes,
        descriptors,
        group_count,
        output,
        INTERMEDIATE=INTERMEDIATE,
        OUTPUT=HIDDEN,
        XS0=intermediate_scales.stride(0),
        XS1=intermediate_scales.stride(1),
        WE=w2.stride(0),
        SE=s2.stride(0),
        SS0=s2.stride(1),
        SS1=s2.stride(2),
        BLOCK_N=16,
        BLOCK_K=128,
        BLOCK_M=16,
        num_warps=1,
    )
    owned = (ids.flatten() >= expert_start) & (
        ids.flatten() < expert_start + NUM_EXPERTS
    )
    expected_mapping = torch.where(
        owned, torch.arange(rows * TOPK, device="cuda", dtype=torch.int32), -1
    )
    torch.testing.assert_close(mapping, expected_mapping, atol=0, rtol=0)
    torch.testing.assert_close(
        intermediate[owned], reference_intermediate[owned], atol=0, rtol=0
    )
    torch.testing.assert_close(
        _logical_cdna4_scales(intermediate_scales, rows * TOPK, INTERMEDIATE // 32)[
            owned
        ],
        _logical_cdna4_scales(reference_scales, rows * TOPK, INTERMEDIATE // 32)[owned],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output[owned].view(torch.int16),
        reference[owned].view(torch.int16),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("ep_size", [1, 4, 8])
def test_package_muse_changed_graph_inputs(ep_size) -> None:

    local_experts = 256 // ep_size
    intermediate_size = 512 if ep_size == 1 else 4096
    expert_start = 0 if ep_size == 1 else local_experts
    generator = torch.Generator(device="cuda").manual_seed(20261003)
    runtime = _make_runtime_weights(
        generator,
        num_experts=local_experts,
        hidden_size=4096,
        intermediate_size=intermediate_size,
        scale=120,
    )
    w13, s13, w2, s2 = runtime[4:]
    del runtime
    module = SimpleNamespace(
        hidden_size=4096,
        intermediate_size=4096,
        num_local_experts=local_experts,
        num_experts=256,
        top_k=8,
        ep_rank=0 if ep_size == 1 else 1,
        w13_weight_triton_tensor=w13,
        w2_weight_triton_tensor=w2,
        w13_precision_config=SimpleNamespace(b_mx_scale=s13),
        w2_precision_config=SimpleNamespace(b_mx_scale=s2, out_dtype=torch.bfloat16),
    )
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        requires_deferred_finalize=True,
        routing_mode="precomputed_topk",
        ep_size=ep_size,
        ispp=intermediate_size,
        hidden=4096,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=False,
        internal_activation_dtype="mxfp4",
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution="gluon",
    )

    def warmup(max_tokens):
        previous_dtype = torch.get_default_dtype()
        try:
            # Weight loading invokes this lifecycle under the model's dtype.
            torch.set_default_dtype(torch.bfloat16)
            moe_warmup(plan, module, max_num_tokens=max_tokens)
        finally:
            torch.set_default_dtype(previous_dtype)

    warmup(48)
    kernels = (
        gluon_mxfp4_moe_route_descriptors_gfx950,
        gluon_mxfp4_moe_route_pack_gfx950,
        gluon_mxfp4_moe_route_tiles_gfx950,
        gluon_mxfp4_moe_stage1_route_tiles_gfx950,
        gluon_mxfp4_moe_stage2_route_tiles_gfx950,
        _stage1_mxfp4_direct_mfma_gluon,
        _stage2_mxfp4_direct_mfma_gluon,
        gluon_mxfp4_moe_sort_histogram_prefix_gfx950,
        _moe_sorting_stage1_kernel,
        _moe_sorting_stage2_kernel,
        _moe_sorting_stage3_kernel,
        _moe_sorting_stage4_kernel,
        gluon_mxfp4_moe_stage1_kernel,
        gluon_mxfp4_moe_stage2_1x2_kernel,
        _gather_package_cdna4_scale_kernel,
        _local_route_map_kernel,
    )
    # The actual startup lifecycle warms families; these held-out tails must
    # execute through public dispatch without compiling new variants.
    row_counts = (3, 5, 31, 53, 128, 191, 207, 8191, 8192)
    if ep_size == 1:
        # Exercise every density-selected row tile with an unwarmed token tail.
        row_counts = (*row_counts[:-2], 263, 1031, 2055, 4103, *row_counts[-2:])
    for rows in row_counts:
        if rows == 53:
            warmup(128)
        elif rows == 191:
            warmup(8192)
        hidden = _exact_mxfp4_activation(rows, 4096, generator)
        packed, scales = _quantize_mxfp4_activation(hidden)
        ids = (
            torch.arange(
                expert_start, expert_start + 8, device="cuda", dtype=torch.int32
            )
            .expand(rows, 8)
            .clone()
        )
        weights = torch.full((rows, 8), 0.125, device="cuda")
        intermediate = torch.empty(
            (rows * 8, intermediate_size // 2), dtype=torch.uint8, device="cuda"
        )
        intermediate_scales = empty_swizzled_cdna4_mxfp4_scale(
            rows * 8, intermediate_size // 32, device="cuda"
        )
        reference = torch.empty((rows * 8, 4096), dtype=torch.bfloat16, device="cuda")
        reference_map = torch.empty(rows * 8, dtype=torch.int32, device="cuda")
        guarded = torch.full(
            (rows * 8 + 2, 4096), 17, dtype=torch.bfloat16, device="cuda"
        )
        destination = guarded[1:-1]

        def run_reference():
            invoke_stage1_mxfp4_mfma_decode_gluon(
                packed,
                scales,
                w13,
                s13,
                ids,
                intermediate,
                8,
                expert_start,
                intermediate_scales,
                BLOCK_N=64,
                swiglu_alpha=1.0,
                swiglu_limit=0.0,
                swiglu_beta=0.0,
            )
            invoke_stage2_mxfp4_mfma_decode_gluon(
                intermediate,
                intermediate_scales,
                w2,
                s2,
                ids,
                weights,
                reference,
                8,
                expert_start,
                False,
                reference_map,
                BLOCK_N=16,
                BLOCK_K=128,
                PIPELINE_K=False,
            )

        def run_actual(output):
            return moe_apply(
                plan,
                (packed, scales),
                module,
                None,
                topk_weights=weights,
                topk_ids=ids,
                do_finalize=False,
                deferred_out=output,
            )

        run_reference()
        with assert_no_triton_compile(*kernels):
            allocated, _, allocated_map = run_actual(None)
            supplied, _, supplied_map = run_actual(destination)
            torch.testing.assert_close(supplied, allocated, atol=0, rtol=0)
            torch.testing.assert_close(supplied_map, allocated_map, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_reference()
                actual, _, actual_map = run_actual(destination)
        assert actual.data_ptr() == destination.data_ptr()
        for phase in range(3):
            new_hidden = _exact_mxfp4_activation(rows, 4096, generator)
            new_packed, new_scales = _quantize_mxfp4_activation(new_hidden)
            packed.copy_(new_packed)
            scales.copy_(new_scales)
            if phase == 0 and rows == 191 and ep_size > 1:
                # Maximizes B32 padding while keeping each token's top8 distinct.
                remaining = (
                    [161] * 9 + [89 - local_experts] + [1] * (local_experts - 10)
                )
                routes = []
                for _ in range(rows):
                    selected = sorted(
                        range(local_experts), key=lambda e: (-remaining[e], e)
                    )[:8]
                    assert all(remaining[e] > 0 for e in selected)
                    routes.append(selected)
                    for expert in selected:
                        remaining[expert] -= 1
                assert not any(remaining)
                ids.copy_(
                    torch.tensor(routes, device="cuda", dtype=torch.int32)
                    + expert_start
                )
                sorted_ids, _, _, valid, _ = gluon_moe_sorting(
                    ids,
                    weights,
                    local_experts,
                    4096,
                    torch.bfloat16,
                    32,
                    runtime_rows=True,
                    use_minimum_route_chunks=False,
                    expert_start=expert_start,
                )
                assert (
                    int(valid[0])
                    == sorted_ids.numel()
                    == _max_padded_route_capacity(rows * 8, local_experts, 32)
                    == (3488 if local_experts == 64 else 2496)
                )
            elif phase == 0:
                ids.copy_(
                    torch.arange(expert_start, expert_start + 8, device="cuda").expand(
                        rows, 8
                    )
                )
            elif phase == 1:
                ids.copy_(torch.arange(128, 136, device="cuda").expand(rows, 8))
            else:
                ids.copy_(
                    torch.stack(
                        [
                            torch.randperm(256, device="cuda", generator=generator)[:8]
                            for _ in range(rows)
                        ]
                    )
                )
            reference.fill_(float("nan"))
            actual.fill_(float("nan"))
            reference_map.fill_(-123)
            actual_map.fill_(-123)
            graph.replay()
            torch.cuda.synchronize()
            owned = (ids.flatten() >= expert_start) & (
                ids.flatten() < expert_start + local_experts
            )
            expected_map = torch.where(
                owned, torch.arange(rows * 8, device="cuda", dtype=torch.int32), -1
            )
            torch.testing.assert_close(actual_map, expected_map, rtol=0, atol=0)
            torch.testing.assert_close(reference_map, expected_map, rtol=0, atol=0)
            torch.testing.assert_close(
                actual[owned].view(torch.int16),
                reference[owned].view(torch.int16),
                rtol=0,
                atol=0,
            )
            assert bool(torch.isfinite(actual[owned]).all())
            torch.testing.assert_close(
                guarded[0], torch.full_like(guarded[0], 17), rtol=0, atol=0
            )
            torch.testing.assert_close(
                guarded[-1], torch.full_like(guarded[-1], 17), rtol=0, atol=0
            )
