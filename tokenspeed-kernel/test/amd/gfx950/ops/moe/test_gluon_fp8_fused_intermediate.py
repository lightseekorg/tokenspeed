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

"""Fused intermediate quantization and producer numerical checks on gfx950."""

import pytest
import torch
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale
from utils import is_cdna4

pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small",
    reason="tokenspeed-kernel-amd is required for fused intermediate tests",
)

from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (  # noqa: E402
    gluon_fp8_packed_small_activation_gfx950,
)

pytestmark = pytest.mark.skipif(not is_cdna4(), reason="requires AMD gfx950 GPU")


def uses_reciprocal(groups):
    return (
        torch.cuda.get_device_properties(0).multi_processor_count == 256
        and groups >= 8192
    )


@gluon.jit
def separate_activation(partials, output, rows, LIMIT: gl.constexpr, I: gl.constexpr):
    offsets = gl.program_id(0) * 256 + gl.arange(
        0, 256, gl.BlockedLayout([1], [64], [4], [0])
    )
    row, col = offsets // I, offsets % I
    gate = gl.load(partials + row * (2 * I) + col, row < rows, other=0.0)
    up = gl.load(partials + row * (2 * I) + I + col, row < rows, other=0.0)
    gate = gl.minimum(gate, LIMIT)
    up = gl.clamp(up, -LIMIT, LIMIT)
    activated = gate / (1.0 + gl.exp(-gate)) * up
    gl.store(output + offsets, activated.to(gl.bfloat16), row < rows)


@pytest.mark.parametrize(
    "schedule,rows,ep,width",
    [
        ("small", 8, 1, 512),
        ("small", 8, 4, 2048),
        ("large", 64, 1, 512),
        ("large", 65, 4, 2048),
    ],
)
def test_producer_matches_quantized_reference(schedule, rows, ep, width):
    from test_gluon_fp8_packed_small import _fixture
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import packed_large, packed_small
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
        moe_align_block_size_device,
    )

    x, w13, _, scales, _, ids, weights = _fixture(rows, ep, 0, True, 4096, width)
    if schedule == "large" and ep == 4:
        ids[:, 0], ids[:, 1] = 0, 1
    block_m = 16 if schedule == "small" else 64
    sorted_ids, experts, _, count = moe_align_block_size_device(
        ids, weights, w13.shape[0], block_m, expert_start=0
    )
    qx, sx = shared(x)
    packed = pack_fp8_weight_n16k32(w13)
    actual_q = torch.zeros((rows * 8, width), device="cuda", dtype=torch.float8_e4m3fn)
    actual_s = torch.zeros((rows * 8, width // 128), device="cuda")
    options = (
        {"SPLIT_COUNT": 1}
        if schedule == "small"
        else {"USE_NT": False, "num_stages": 1}
    )
    if schedule == "small":
        new = packed_small.gluon_fp8_packed_small_stage1_gfx950
    else:
        new = packed_large.gluon_fp8_packed_large_stage1_gfx950
    grid = ((sorted_ids.numel() + block_m - 1) // block_m * (width // 128),)
    prefix = (qx.view(torch.uint8), sx, packed.view(torch.uint8), scales)
    suffix = (sorted_ids, experts, count, rows, sorted_ids.numel(), 7.0)
    new[grid](
        *prefix,
        actual_q,
        actual_s,
        *suffix,
        K=4096,
        I=width,
        TOPK=8,
        USE_RECIPROCAL=uses_reciprocal(rows * 8 * width // 128),
        num_warps=4,
        **options,
    )
    dequantized_x = qx.float() * sx.repeat_interleave(128, dim=1)
    actual = actual_q.float() * actual_s.repeat_interleave(128, dim=1)
    for expert in range(w13.shape[0]):
        tokens, slots = torch.where(ids == expert)
        if tokens.numel() == 0:
            continue
        weight = w13[expert].float() * scales[expert].repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
        gate, up = (dequantized_x[tokens] @ weight.T).chunk(2, dim=-1)
        activated = (
            torch.nn.functional.silu(gate.clamp(max=7.0)) * up.clamp(-7.0, 7.0)
        ).to(torch.bfloat16)
        expected_q, expected_s = shared(activated)
        expected = expected_q.float() * expected_s.repeat_interleave(128, dim=1)
        observed = actual[tokens * 8 + slots]
        assert (
            (observed - expected).norm() / expected.norm().clamp_min(1.0e-12)
        ) < 0.03
        assert (
            torch.nn.functional.cosine_similarity(
                observed.flatten(), expected.flatten(), dim=0
            )
            > 0.9995
        )


def shared(x):
    return quantize_fp8_with_scale(
        x, granularity="token_group", group_size=128, solution="triton"
    )


@pytest.mark.parametrize(
    "rows,width",
    [(1, 512), (3, 512), (32, 512), (511, 2048), (512, 2048), (513, 2048)],
)
def test_completion_matches_separate_and_graph(rows, width):
    generator = torch.Generator(device="cuda").manual_seed(3202)
    partials = torch.randn((rows, 2 * width), device="cuda", generator=generator) * 12
    if rows >= 32:
        partials[0].zero_()
        partials[1].fill_(1.0e-14)
    bf16 = torch.empty((rows, width), device="cuda", dtype=torch.bfloat16)
    separate_activation[((rows * width + 255) // 256,)](
        partials, bf16, rows, LIMIT=7.0, I=width, num_warps=4
    )
    expected_q, expected_s = shared(bf16)
    actual_q, actual_s = torch.empty_like(expected_q), torch.empty_like(expected_s)

    def fused():
        gluon_fp8_packed_small_activation_gfx950[((rows * width + 255) // 256,)](
            partials,
            actual_q,
            actual_s,
            rows,
            7.0,
            I=width,
            BLOCK=256,
            USE_RECIPROCAL=uses_reciprocal(rows * width // 128),
            num_warps=4,
        )

    fused()
    assert torch.equal(actual_s, expected_s)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused()
    partials.mul_(0.5)
    separate_activation[((rows * width + 255) // 256,)](
        partials, bf16, rows, LIMIT=7.0, I=width, num_warps=4
    )
    expected_q, expected_s = shared(bf16)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(actual_s, expected_s)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
