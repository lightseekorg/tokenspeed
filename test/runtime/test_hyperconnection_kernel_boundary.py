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
from unittest import mock

import pytest
import torch
from tokenspeed_kernel import (
    gated_residual_combine,
    gated_residual_mix,
    grouped_gemma_rmsnorm,
)

import tokenspeed.runtime.layers.hyperconnection as hyperconnection_module
from tokenspeed.runtime.layers.hyperconnection import (
    GatedResidualSimple,
    HyperConnectionConfig,
)


def test_runtime_uses_tokenspeed_kernel_boundary() -> None:
    source = inspect.getsource(hyperconnection_module)
    assert "from tokenspeed_kernel import" in source
    assert "import triton" not in source
    assert "@triton.jit" not in source


def test_kernel_boundary_is_gpu_only() -> None:
    normalized = torch.empty(1, 8)
    projection = torch.empty(4, 8)
    up = torch.empty(8, 2)
    with pytest.raises(ValueError, match="requires GPU tensors"):
        gated_residual_mix(normalized, projection, up, 2, 4, 2)
    with pytest.raises(ValueError, match="requires GPU tensors"):
        gated_residual_combine(torch.empty(1, 4), normalized, torch.empty(1, 2), 2, 4)
    with pytest.raises(ValueError, match="requires GPU tensors"):
        grouped_gemma_rmsnorm(normalized, torch.empty(8), 4, 1e-6)


def test_up_weight_loader_prepares_kernel_cache(monkeypatch) -> None:
    prepare = mock.Mock(return_value=True)
    monkeypatch.setattr(
        hyperconnection_module, "prepare_gated_residual_weight_cache", prepare
    )
    lowrank = 3
    mixer = GatedResidualSimple(
        HyperConnectionConfig(hc_count=2, hidden_size=4, hc_lowrank=lowrank)
    )
    param = mixer.input_mix_weight_up.weight
    loaded = torch.randn_like(param)

    param.weight_loader(param, loaded)

    torch.testing.assert_close(param, loaded)
    prepare.assert_called_once_with(param, lowrank)


def test_up_weight_loader_rejects_shape_change(monkeypatch) -> None:
    prepare = mock.Mock(return_value=True)
    monkeypatch.setattr(
        hyperconnection_module, "prepare_gated_residual_weight_cache", prepare
    )
    mixer = GatedResidualSimple(
        HyperConnectionConfig(hc_count=2, hidden_size=4, hc_lowrank=3)
    )
    param = mixer.input_mix_weight_up.weight

    with pytest.raises(ValueError, match="shape mismatch"):
        param.weight_loader(param, torch.empty(8, 4))

    prepare.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_non_power_of_two_hc_scales_projection_results() -> None:
    hc_count, hidden_size, lowrank = 3, 8, 6
    mixer = GatedResidualSimple(
        HyperConnectionConfig(
            hc_count=hc_count,
            hidden_size=hidden_size,
            hc_lowrank=lowrank,
            params_dtype=torch.float32,
        )
    ).cuda()
    down_weight = torch.randn(lowrank, hc_count * hidden_size, device="cuda")
    inject_weight = torch.randn(hc_count, hc_count * hidden_size, device="cuda")
    param = mixer.mix_inject_proj.weight
    loader = param.weight_loader
    loader(param, down_weight, "mix")
    loader(param, inject_weight, "inject")

    torch.testing.assert_close(param[:lowrank], down_weight)
    torch.testing.assert_close(param[lowrank:], inject_weight)
    assert mixer._projection_scale == pytest.approx(1.0 / hc_count)

    hyper_input = torch.randn(5, hc_count * hidden_size, device="cuda")
    block_output = torch.randn(5, hidden_size, device="cuda")
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
