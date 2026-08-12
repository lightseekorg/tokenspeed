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
from kimi3_reference import dequantize_mxfp4
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon MXFP4 direct decode tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage1 import (  # noqa: E402
    invoke_stage1_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage2 import (  # noqa: E402
    invoke_stage2_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _quantize_mxfp4_activation,
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
) -> tuple[torch.Tensor, ...]:
    """Random MXFP4 experts, both raw (for the reference) and gdot128-shuffled."""

    def payload(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=generator
        )

    def unit_scales(*shape: int) -> torch.Tensor:
        return torch.full(shape, 127, dtype=torch.uint8, device="cuda")

    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    raw_w13 = payload(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2)
    raw_w2 = payload(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2)
    raw_w13_scale = unit_scales(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32)
    raw_w2_scale = unit_scales(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 32)
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
