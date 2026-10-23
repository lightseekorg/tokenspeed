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

"""Kimi-K3 front fusion and its unsupported-input fallback."""

import math

import pytest
import torch
from tokenspeed_kernel.ops.moe import (
    latent_moe_input_projections,
    try_latent_moe_input_projections,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.selection import NoKernelFoundError, kernel_override
from tokenspeed_kernel.thirdparty.cuda.flashinfer_kimi_k3 import latent_front_available


@pytest.fixture(scope="module")
def weights():
    platform = current_platform()
    if (
        not platform.is_nvidia
        or platform.arch_version != ArchVersion(10, 3)
        or platform.sm_count != 152
        or not latent_front_available()
    ):
        pytest.skip("requires the GB300 Kimi-K3 source bundle")
    torch.manual_seed(621)
    return tuple(
        (torch.randn(rows, 7168, device="cuda") / math.sqrt(7168)).bfloat16()
        for rows in (896, 3584, 12288)
    )


def reference(x, weights):
    router, latent, shared = weights
    raw = torch.nn.functional.linear(x, shared)
    gate, up = raw.float().chunk(2, dim=-1)
    activated = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(up / 25.0)
    )
    return (
        torch.nn.functional.linear(x.float(), router.float()),
        torch.nn.functional.linear(x, latent),
        activated.bfloat16(),
    )


@pytest.mark.parametrize("rows", [2048, 2049, 4096, 8192])
def test_front_matches_reference_and_replays_changed_inputs(weights, rows):
    torch.manual_seed(147 + rows)
    x = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16)

    def run():
        return latent_moe_input_projections(
            x,
            *weights,
            gate_clamp=4.0,
            up_clamp=25.0,
            override="flashinfer_latent_input",
        )

    actual = run()
    assert actual[0].dtype == torch.float32
    for i, (value, expected) in enumerate(
        zip(actual, reference(x, weights), strict=True)
    ):
        tolerance = 1e-3 if i == 0 else 1e-2
        torch.testing.assert_close(value, expected, atol=tolerance, rtol=tolerance)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    x.normal_(std=0.25)
    graph.replay()
    for i, (value, expected) in enumerate(
        zip(captured, reference(x, weights), strict=True)
    ):
        tolerance = 1e-3 if i == 0 else 1e-2
        torch.testing.assert_close(value, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("rows", [1, 128, 1024, 2047, 16385])
def test_front_declines_unvalidated_rows(weights, rows):
    x = torch.zeros(rows, 7168, device="cuda", dtype=torch.bfloat16)
    assert (
        try_latent_moe_input_projections(x, *weights, gate_clamp=4.0, up_clamp=25.0)
        is None
    )
    assert torch.count_nonzero(x).item() == 0
    with pytest.raises(NoKernelFoundError):
        try_latent_moe_input_projections(
            x,
            *weights,
            gate_clamp=4.0,
            up_clamp=25.0,
            solution="flashinfer",
        )


@pytest.mark.parametrize("gate_clamp,up_clamp", [(1.0, 25.0), (4.0, None), (4.0, 1.0)])
def test_front_declines_other_activations(weights, gate_clamp, up_clamp):
    x = torch.zeros(2048, 7168, device="cuda", dtype=torch.bfloat16)
    assert (
        try_latent_moe_input_projections(
            x, *weights, gate_clamp=gate_clamp, up_clamp=up_clamp
        )
        is None
    )


def test_front_declines_sharded_shared_experts(weights):
    x = torch.zeros(2048, 7168, device="cuda", dtype=torch.bfloat16)
    router, latent, shared = weights
    shard = shared[:3072].clone()
    assert (
        try_latent_moe_input_projections(
            x, router, latent, shard, gate_clamp=4.0, up_clamp=25.0
        )
        is None
    )


def test_front_preserves_explicit_reference_override(weights):
    from tokenspeed_kernel.numerics.torch.moe import torch_latent_input

    x = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16) * 0.25
    with kernel_override("moe", "latent_input", "torch_latent_input"):
        actual = try_latent_moe_input_projections(
            x, *weights, gate_clamp=4.0, up_clamp=25.0
        )
    expected = torch_latent_input(
        hidden_states=x,
        router_weight=weights[0],
        routed_weight=weights[1],
        shared_gate_up_weight=weights[2],
        gate_clamp=4.0,
        up_clamp=25.0,
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
