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

import inspect
import os

import pytest
import torch
from tokenspeed_kernel.platform import (
    current_platform,
    pdl_enabled,
    set_pdl_enabled,
)

import tokenspeed.runtime.layers.hyperconnection as hyperconnection_module
from tokenspeed.runtime.layers.hyperconnection import (
    GatedResidualSimple,
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def test_runtime_uses_tokenspeed_kernel_boundary() -> None:
    source = inspect.getsource(hyperconnection_module)
    assert "from tokenspeed_kernel import" in source
    assert "import triton" not in source
    assert "@triton.jit" not in source


def test_server_args_is_the_authoritative_pdl_switch(monkeypatch) -> None:
    for name in ("TORCHINDUCTOR_ENABLE_PDL", "TRTLLM_ENABLE_PDL"):
        monkeypatch.setenv(name, os.environ.get(name, "0"))
    previous = pdl_enabled()
    args = object.__new__(ServerArgs)
    args.device = "cuda"
    try:
        args.disable_pdl = True
        assert not args.configure_pdl()
        assert not pdl_enabled()
        assert os.environ["TORCHINDUCTOR_ENABLE_PDL"] == "0"
        assert os.environ["TRTLLM_ENABLE_PDL"] == "0"

        args.disable_pdl = False
        expected = current_platform().is_hopper_plus
        assert args.configure_pdl() is expected
        assert pdl_enabled() is expected
        expected_value = "1" if expected else "0"
        assert os.environ["TORCHINDUCTOR_ENABLE_PDL"] == expected_value
        assert os.environ["TRTLLM_ENABLE_PDL"] == expected_value
    finally:
        set_pdl_enabled(previous)


def test_non_power_of_two_hc_scales_projection_results() -> None:
    hc_count, hidden_size, lowrank = 3, 8, 6
    mixer = GatedResidualSimple(
        HyperConnectionConfig(
            hc_count=hc_count,
            hidden_size=hidden_size,
            hc_lowrank=lowrank,
            params_dtype=torch.float32,
        )
    )
    down_weight = torch.randn(lowrank, hc_count * hidden_size)
    inject_weight = torch.randn(hc_count, hc_count * hidden_size)
    param = mixer.mix_inject_proj.weight
    loader = param.weight_loader
    loader(param, down_weight, "mix")
    loader(param, inject_weight, "inject")

    torch.testing.assert_close(param[:lowrank], down_weight)
    torch.testing.assert_close(param[lowrank:], inject_weight)
    assert mixer._projection_scale == pytest.approx(1.0 / hc_count)

    hyper_input = torch.randn(5, hc_count * hidden_size)
    block_output = torch.randn(5, hidden_size)
    mixed, residuals = mixer.mix(hyper_input)
    combined = mixer.combine(block_output, residuals)
    normalized = residuals[1]

    down = torch.nn.functional.linear(normalized, down_weight) / hc_count
    gate = mixer.input_mix_weight_up(torch.nn.functional.silu(down))
    expected_mixed = (
        torch.sigmoid(gate).unflatten(-1, (hc_count, hidden_size))
        * normalized.unflatten(-1, (hc_count, hidden_size))
    ).mean(dim=-2)
    torch.testing.assert_close(mixed, expected_mixed)

    inject = 2 * torch.sigmoid(
        torch.nn.functional.linear(normalized, inject_weight) / hc_count
    )
    expected_combined = hyper_input.unflatten(
        -1, (hc_count, hidden_size)
    ) + block_output.unsqueeze(-2) * inject.unsqueeze(-1)
    torch.testing.assert_close(combined, expected_combined.flatten(-2))


def test_power_of_two_hc_folds_projection_scale_exactly() -> None:
    hc_count, hidden_size, lowrank = 4, 8, 6
    mixer = GatedResidualSimple(
        HyperConnectionConfig(
            hc_count=hc_count,
            hidden_size=hidden_size,
            hc_lowrank=lowrank,
            params_dtype=torch.bfloat16,
        )
    )
    down_weight = torch.randn(lowrank, hc_count * hidden_size).bfloat16()
    inject_weight = torch.randn(hc_count, hc_count * hidden_size).bfloat16()
    param = mixer.mix_inject_proj.weight
    param.weight_loader(param, down_weight, "mix")
    param.weight_loader(param, inject_weight, "inject")

    assert mixer._projection_scale == 1.0
    torch.testing.assert_close(
        param[:lowrank], (down_weight / hc_count).to(param.dtype)
    )
    torch.testing.assert_close(
        param[lowrank:], (inject_weight / hc_count).to(param.dtype)
    )


def test_grouped_gemma_rmsnorm_cpu_reference() -> None:
    norm = GroupedGemmaRMSNorm(24, 1e-6, group_size=8)
    loaded_weight = torch.randn(24)
    norm.weight.weight_loader(norm.weight, loaded_weight)
    x = torch.randn(5, 24)
    actual = norm(x)

    grouped = x.float().unflatten(-1, (3, 8))
    expected = (
        grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).flatten(-2) * (1.0 + loaded_weight)
    torch.testing.assert_close(actual, expected)
