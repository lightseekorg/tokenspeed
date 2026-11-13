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

"""Block-scaled experts preserve portable rounding, remote rows and graph inputs."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from block_quantize_helpers import (  # noqa: E402
    triton_block_quantize,
    triton_block_quantize_nvfp4_values,
)
from tokenspeed_kernel.ops.moe._triton.block_scaled import _expert_mm  # noqa: E402
from tokenspeed_kernel.ops.moe._triton.common import _routing  # noqa: E402
from tokenspeed_kernel.ops.moe.block_scaled import (  # noqa: E402
    block_scaled_moe_routes,
    block_scaled_moe_routes_plan,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.block_scaled import (  # noqa: E402
    gluon_block_scaled_moe_routes_mm_gfx950,
)


def _weights(experts, n, k, fmt):
    values = torch.randn((experts * n, k), device="cuda", dtype=torch.bfloat16)
    packed, scales, _ = triton_block_quantize(values, fmt)
    if fmt == "nvfp4":
        outer = torch.linspace(0.25, 1.0, experts, device="cuda")
        denominator = torch.tensor([1536.0], device="cuda")
    else:
        outer, denominator = None, None
    return (
        packed.view(experts, n, -1),
        scales.view(experts, n, -1),
        outer,
        denominator,
    )


def _nvfp4_values(packed, scales):
    """BF16 values of NVFP4 codes: each E2M1 value times its E4M3 group scale."""
    codes = torch.stack([packed & 15, packed >> 4], -1).reshape(packed.shape[0], -1)
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    value = magnitude[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    return (value * scales.float().repeat_interleave(16, dim=1)).bfloat16()


@pytest.mark.parametrize(
    "fmt,bf16_x", [("nvfp4", False), ("nvfp4", True), ("mxfp8", False)]
)
@pytest.mark.parametrize("per_token", [True, False])
@pytest.mark.parametrize(
    "tiles", [(16, 128, 256), (16, 64, 256), (32, 128, 128), (64, 128, 128)]
)
# Ragged shapes take the masked loop. Whole tiles of NVFP4 weights and BF16
# activations take the LDS ring; 3 programs run several tiles each through it.
@pytest.mark.parametrize(
    "n,k,few_programs", [(96, 160, False), (128, 768, False), (128, 768, True)]
)
def test_expert_mm_masking_and_runtime_rows(
    fmt, bf16_x, per_token, tiles, n, k, few_programs
):
    torch.manual_seed(42)
    experts, top_k = 3, 2
    w, ws, wa, wd = _weights(experts, n, k, fmt)
    programs = (
        3
        if few_programs
        else torch.cuda.get_device_properties(0).multi_processor_count * 2
    )

    def execute(tokens):
        x = torch.randn(
            (tokens if per_token else tokens * top_k, k),
            device="cuda",
            dtype=torch.bfloat16,
        )
        if bf16_x:
            q, qs, qa, xb = triton_block_quantize_nvfp4_values(x)
            # The quantizer's BF16 values are exactly what its codes encode.
            assert torch.equal(xb, _nvfp4_values(q, qs))
        else:
            q, qs, qa = triton_block_quantize(x, fmt)
        # Expert 12 is always empty. Expert 13 belongs to the next EP shard.
        ids = torch.tensor([[10, 13], [11, 10]], device="cuda", dtype=torch.int32)
        ids = ids.repeat((tokens + 1) // 2, 1)[:tokens].contiguous()
        routes, counts = _routing(ids, experts, 10)
        reference = torch.zeros(
            (tokens * top_k, n), device="cuda", dtype=torch.bfloat16
        )
        actual = torch.zeros_like(reference)
        args = (
            q,
            qs,
            qa,
            w,
            ws,
            wa,
            wd,
            reference,
            routes,
            counts,
            tokens * top_k,
            programs,
            n,
            k,
            experts,
            top_k,
            per_token,
            fmt == "nvfp4",
        )
        _expert_mm[(programs,)](
            *args,
            16,
            64,
            64,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
        # BF16 activations: the exact values of the codes the reference decodes.
        x_operand = xb if bf16_x else q
        gluon_block_scaled_moe_routes_mm_gfx950[(programs,)](
            x_operand,
            *args[1:7],
            actual,
            *args[8:],
            *tiles,
            bf16_x,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        assert actual.view(tokens, top_k, n)[ids == 13].count_nonzero() == 0

    execute(8)
    with assert_no_triton_compile(gluon_block_scaled_moe_routes_mm_gfx950):
        for tokens in (1, 7, 9, 17, 32, 65, 129):
            execute(tokens)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [0, 1, 8, 17, 128, 1023, 1024, 1025])
def test_registered_routes_preserve_bf16_boundaries(fmt, tokens):
    torch.manual_seed(7)
    x = torch.randn((tokens, 160), device="cuda", dtype=torch.bfloat16)
    w13, w2 = _weights(3, 192, 160, fmt), _weights(3, 160, 96, fmt)
    weights = torch.rand((tokens, 2), device="cuda", dtype=torch.float32)
    ids = torch.tensor([[10, 13], [11, 10]], device="cuda", dtype=torch.int32)
    ids = ids.repeat((tokens + 1) // 2, 1)[:tokens].contiguous()
    args = (x, *w13, *w2, weights, ids)
    kwargs = dict(
        experts=14,
        expert_start=10,
        gate_cap=4.0,
        up_cap=25.0,
        zero_remote_routes=True,
        fold_slots=False,
        fold_into=None,
    )

    def routes(solution):
        # Both solutions read the experts in their loaded layout.
        plan = block_scaled_moe_routes_plan(
            formats=(fmt, fmt),
            hidden=160,
            intermediate=96,
            device=x.device,
            solution=solution,
        )
        return block_scaled_moe_routes(plan, *args, **kwargs)

    reference = routes("triton")
    actual = routes("gluon")
    automatic = routes(None)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(automatic, reference, atol=0, rtol=0)


def test_expert_graph_replay_reads_changed_routes_and_values():
    torch.manual_seed(11)
    tokens, n, k, top_k = 8, 96, 160, 2
    w, ws, wa, wd = _weights(3, n, k, "nvfp4")
    x = torch.randn((tokens, k), device="cuda", dtype=torch.bfloat16)
    q, qs, qa = triton_block_quantize(x, "nvfp4")
    ids = torch.tensor([[0, 1]] * tokens, device="cuda", dtype=torch.int32)
    routes, counts = _routing(ids, 3, 0)
    y = torch.zeros((tokens * top_k, n), device="cuda", dtype=torch.bfloat16)
    programs = torch.cuda.get_device_properties(0).multi_processor_count * 2
    args = (
        q,
        qs,
        qa,
        w,
        ws,
        wa,
        wd,
        y,
        routes,
        counts,
        tokens * top_k,
        programs,
        n,
        k,
        3,
        top_k,
        True,
        True,
    )

    def launch():
        gluon_block_scaled_moe_routes_mm_gfx950[(programs,)](
            *args,
            16,
            128,
            256,
            False,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    original = y.clone()
    q.copy_(q.bitwise_xor(0x88))
    ids.copy_(torch.tensor([[2, 0]] * tokens, device="cuda", dtype=torch.int32))
    new_routes, new_counts = _routing(ids, 3, 0)
    routes.copy_(new_routes)
    counts.copy_(new_counts)
    graph.replay()
    actual = y.clone()
    _expert_mm[(programs,)](
        *args,
        16,
        64,
        64,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    torch.testing.assert_close(actual, y, atol=0, rtol=0)
    assert not torch.equal(original, actual)
