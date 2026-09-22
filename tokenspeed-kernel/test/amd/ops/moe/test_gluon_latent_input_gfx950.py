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

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon latent input tests",
        allow_module_level=True,
    )


from tokenspeed_kernel.registry import KernelRegistry  # noqa: E402
from tokenspeed_kernel.selection import spec_matches_shape_traits  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import (  # noqa: E402
    latent_input_prefill,
    latent_input_small_batch,
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
    [(2048, False), (4096, True), (4097, False), (4352, True)],
)
def test_prefill_dispatch_bounds(tokens: int, expected: bool) -> None:
    spec = KernelRegistry.get().get_by_name("gluon_latent_input_prefill_gfx950")
    assert spec is not None
    assert spec_matches_shape_traits(spec, {"tokens": tokens}) is expected


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_prefill_routes_packed_projection_and_applies_situ(
    linear_beta: float | None,
) -> None:
    tokens = 256
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
        torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.02
    )

    actual = latent_input_prefill.launch_gluon_latent_input_prefill_gfx950(
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
    assert actual[0].dtype == torch.float32
    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual[2], expected[2], atol=1e-5, rtol=1e-5)
