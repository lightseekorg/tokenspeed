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
from utils import is_cdna4

pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx950.moe.fp16",
    reason="tokenspeed-kernel-amd is required for Gluon latent input tests",
)

from tokenspeed_kernel.registry import KernelRegistry  # noqa: E402
from tokenspeed_kernel.selection import (  # noqa: E402
    spec_matches_shape_traits,
    spec_matches_traits,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import (  # noqa: E402
    latent_input_largem,
    latent_input_mediumm,
    latent_input_small_batch,
)

requires_cdna4 = pytest.mark.skipif(
    not is_cdna4(), reason="AMD CDNA4 is required to launch Gluon latent input kernels"
)


@pytest.mark.parametrize("hidden_size", [64, 128, 192, 256, 7168])
def test_split_k_covers_every_tile(hidden_size: int) -> None:
    split_k = latent_input_small_batch._split_k(
        tokens=2, total_n=6016, hidden=hidden_size, block_m=16
    )
    assert (hidden_size // latent_input_small_batch._BLOCK_K) % split_k == 0


def test_split_k_does_not_drop_k_tiles() -> None:
    assert (
        latent_input_small_batch._split_k(
            tokens=2, total_n=6016, hidden=192, block_m=16
        )
        == 1
    )


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (128, "gluon_latent_input_small_batch_gfx950"),
        (129, "gluon_latent_input_small_batch_gfx950"),
        (320, "gluon_latent_input_small_batch_gfx950"),
        (321, "gluon_latent_input_mediumm_gfx950"),
        (640, "gluon_latent_input_mediumm_gfx950"),
        (641, None),
        (1280, None),
        (1281, "gluon_latent_input_largem_gfx950"),
        (2048, "gluon_latent_input_largem_gfx950"),
        (4095, "gluon_latent_input_largem_gfx950"),
        (4096, "gluon_latent_input_largem_gfx950"),
    ],
)
def test_k3_prefill_dispatch(tokens: int, expected: str | None) -> None:
    traits = {
        "tokens": tokens,
        "hidden_size": 7168,
        "num_experts": 896,
        "latent_size": 3584,
        "shared_size": 768,
        "inputs_contiguous": True,
        "weights_packed": True,
        "hidden_size_multiple_64": True,
    }
    names = (
        "gluon_latent_input_small_batch_gfx950",
        "gluon_latent_input_mediumm_gfx950",
        "gluon_latent_input_largem_gfx950",
    )
    matches = [
        name
        for name in names
        if (spec := KernelRegistry.get().get_by_name(name)) is not None
        and spec_matches_traits(spec, traits)
        and spec_matches_shape_traits(spec, traits)
    ]
    assert matches == ([] if expected is None else [expected])


@requires_cdna4
@pytest.mark.parametrize("tokens", [321, 512, 639])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_medium_routes_packed_projection_and_applies_situ(
    tokens: int,
    linear_beta: float | None,
) -> None:
    hidden_size = 7168
    widths = (896, 3584, 1536)
    beta = 4.0
    torch.manual_seed(7)
    packed = (
        torch.randn(sum(widths), hidden_size, dtype=torch.bfloat16, device="cuda")
        * 0.02
    )
    router_weight, routed_weight, shared_weight = packed.split(widths)
    hidden = (
        torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.05
    )
    actual = latent_input_mediumm.launch_gluon_latent_input_mediumm_gfx950(
        hidden,
        router_weight,
        routed_weight,
        shared_weight,
        packed,
        beta=beta,
        linear_beta=linear_beta,
    )
    expected_router = torch.nn.functional.linear(hidden.float(), router_weight.float())
    expected_routed = torch.nn.functional.linear(hidden, routed_weight)
    gate, up = torch.nn.functional.linear(hidden, shared_weight).chunk(2, dim=-1)
    expected_gate = beta * torch.tanh(gate.float() / beta) * torch.sigmoid(gate.float())
    expected_up = up.float()
    if linear_beta is not None:
        expected_up = linear_beta * torch.tanh(expected_up / linear_beta)
    expected_shared = (expected_gate * expected_up).bfloat16()
    torch.testing.assert_close(actual[0], expected_router, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[1], expected_routed, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(actual[2], expected_shared, atol=1.5e-4, rtol=1e-3)


@requires_cdna4
# 256 is one full row tile; 512 puts a second workgroup row in the grid so the
# pid_m addressing is exercised; 300 and 577 leave a partial final tile whose
# rows the load clamp duplicates and the epilogue must mask away.
@pytest.mark.parametrize("tokens", [256, 300, 512, 577])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_prefill_routes_packed_projection_and_applies_situ(
    tokens: int,
    linear_beta: float | None,
) -> None:
    hidden_size = 7168
    widths = (896, 3584, 1536)
    beta = 4.0
    torch.manual_seed(7)
    packed = (
        torch.randn(sum(widths), hidden_size, dtype=torch.bfloat16, device="cuda")
        * 0.02
    )
    router_weight, routed_weight, shared_weight = packed.split(widths)
    hidden = (
        torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.05
    )

    actual = latent_input_largem.launch_gluon_latent_input_largem_gfx950(
        hidden,
        router_weight,
        routed_weight,
        shared_weight,
        packed,
        beta=beta,
        linear_beta=linear_beta,
    )

    expected_router = torch.nn.functional.linear(hidden.float(), router_weight.float())
    expected_routed = torch.nn.functional.linear(hidden, routed_weight)
    gate, up = torch.nn.functional.linear(hidden, shared_weight).chunk(2, dim=-1)
    expected_gate = beta * torch.tanh(gate.float() / beta) * torch.sigmoid(gate.float())
    expected_up = up.float()
    if linear_beta is not None:
        expected_up = linear_beta * torch.tanh(expected_up / linear_beta)
    expected = (
        expected_router,
        expected_routed,
        (expected_gate * expected_up).bfloat16(),
    )
    # Expert selection reads the router in FP32, so the packed GEMM must not
    # round those logits to the activation dtype.
    assert actual[0].dtype == torch.float32
    assert actual[0].shape == (tokens, widths[0])
    assert actual[1].shape == (tokens, widths[1])
    assert actual[2].shape == (tokens, widths[2] // 2)
    # Both sides multiply exactly representable BF16 pairs, but the kernel sums
    # them in a different order than the reference over K=7168. The router and
    # routed tolerances match the portable packed-projection test at this input
    # scale. The shared output stays tighter: its BF16 rounding error is about
    # one ULP, and this bound still rejects zero output or an omitted clamp.
    torch.testing.assert_close(actual[0], expected[0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(actual[2], expected[2], atol=1.5e-4, rtol=1e-3)
