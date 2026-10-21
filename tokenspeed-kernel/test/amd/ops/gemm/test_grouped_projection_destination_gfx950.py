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

"""The Torch fallback writes exact BF16 output without displacing Gluon dispatch."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import grouped_bf16_projection
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import kernel_override, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import is_cdna4

if not torch.cuda.is_available() or not is_cdna4():
    pytest.skip("requires AMD CDNA4", allow_module_level=True)


def _inputs(tokens, groups, layout):
    width, rank = 4096, 1024
    if layout == "tokens":
        x = torch.randn(
            tokens, groups * width + 512, device="cuda", dtype=torch.bfloat16
        )[:, : groups * width].unflatten(1, (groups, width))
    elif layout == "groups":
        x = torch.randn(tokens, groups + 1, width, device="cuda", dtype=torch.bfloat16)[
            :, :groups
        ]
    elif layout == "inner":
        x = torch.randn(tokens, groups, width * 2, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    else:
        x = torch.randn(tokens, groups, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        groups,
        rank * 2 if layout == "weights" else rank,
        width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if layout == "weights":
        weight = weight[:, ::2]
    return x, weight


@pytest.mark.parametrize("tokens", [2, 24, 192])
@pytest.mark.parametrize("layout", ["contiguous", "weights"])
@pytest.mark.parametrize("torch_first", [False, True])
def test_grouped_projection_dispatch_ignores_registration_order(
    tokens, layout, torch_first
):
    x, weight = _inputs(tokens, 2, layout)
    registry = KernelRegistry.get()
    specs = registry.get_for_operator("gemm", "grouped_bf16_projection")
    implementations = {spec.name: registry.get_impl(spec.name) for spec in specs}
    try:
        # Re-register the real implementations in both name orders to expose ties.
        for spec in sorted(specs, key=lambda spec: spec.name, reverse=torch_first):
            registry.register(spec, implementations[spec.name])
        for solution in (None, "torch"):
            expected_solution = (
                "gluon" if solution is None and weight.is_contiguous() else "torch"
            )
            selected = select_kernel(
                "gemm",
                "grouped_bf16_projection",
                format_signature(
                    x=dense_tensor_format(x.dtype),
                    weight=dense_tensor_format(weight.dtype),
                ),
                traits={
                    "batch": weight.shape[0],
                    "m": x.shape[0],
                    "n": weight.shape[1],
                    "k": weight.shape[2],
                    "a_inner_stride_one": x.stride(-1) == 1,
                    "b_contiguous": weight.is_contiguous(),
                    "b_inner_stride_one": weight.stride(-1) == 1,
                    "is_cuda": x.is_cuda,
                },
                solution=solution,
            )
            assert (
                selected.name == f"{expected_solution}_grouped_bf16_projection_gfx950"
            )
            actual = grouped_bf16_projection(x, weight, None, solution)
            expected = grouped_bf16_projection(x, weight, None, expected_solution)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        for spec in specs:
            registry.register(spec, implementations[spec.name])


@pytest.mark.parametrize("tokens", [6, 12, 24, 48, 96, 192])
@pytest.mark.parametrize("layout", ["contiguous", "tokens", "groups", "weights"])
def test_grouped_projection_direct_destination_exact(tokens, layout):
    x, weight = _inputs(tokens, 2, layout)
    expected = torch.einsum("tgd,grd->tgr", x, weight)
    actual = grouped_bf16_projection(x, weight, None, "torch")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    assert actual.flatten(1).data_ptr() == actual.data_ptr()
    out = torch.empty_like(actual)
    assert grouped_bf16_projection(x, weight, out, "torch") is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    with kernel_override(
        "gemm", "grouped_bf16_projection", "torch_grouped_bf16_projection"
    ):
        reference = grouped_bf16_projection(x, weight, None, "torch")
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "tokens,groups,layout",
    [
        (0, 2, "contiguous"),
        (1, 2, "contiguous"),
        (6, 1, "tokens"),
        (193, 2, "contiguous"),
        (6, 2, "inner"),
    ],
)
def test_grouped_projection_unsupported_geometry_keeps_reference(
    tokens, groups, layout
):
    x, weight = _inputs(tokens, groups, layout)
    expected = torch.einsum("tgd,grd->tgr", x, weight)
    actual = grouped_bf16_projection(x, weight, None, "torch")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()


def test_grouped_projection_capture_replay_eager_output_ownership():
    x, weight = _inputs(24, 2, "tokens")
    eager_before = grouped_bf16_projection(x, weight, None, "torch")
    before_copy = eager_before.clone()
    explicit = torch.empty_like(eager_before)
    grouped_bf16_projection(x, weight, explicit, "torch")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = grouped_bf16_projection(x, weight, None, "torch")
        grouped_bf16_projection(x, weight, explicit, "torch")
    x.normal_()
    weight.normal_()
    graph.replay()
    expected = torch.einsum("tgd,grd->tgr", x, weight)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    torch.testing.assert_close(explicit, expected, rtol=0, atol=0)
    eager_after = grouped_bf16_projection(x, weight, None, "torch")
    after_copy = eager_after.clone()
    x.add_(0.125)
    graph.replay()
    torch.testing.assert_close(eager_after, after_copy, rtol=0, atol=0)
    torch.testing.assert_close(eager_before, before_copy, rtol=0, atol=0)
