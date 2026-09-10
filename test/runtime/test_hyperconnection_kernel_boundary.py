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
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel import (
    gated_residual_combine,
    gated_residual_combine_norm,
    gated_residual_mix,
    grouped_gemma_rmsnorm,
)

import tokenspeed.runtime.layers.hyperconnection as hyperconnection_module
from tokenspeed.runtime.layers.hyperconnection import (
    GatedResidualSimple,
    HyperConnectionConfig,
)
from tokenspeed.runtime.models.qwen4_exp import _Qwen4ExpDecoderMixin


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
        gated_residual_mix(
            normalized, projection, up, 2, 4, 2, weights_independent=False
        )
    with pytest.raises(ValueError, match="requires GPU tensors"):
        gated_residual_combine(torch.empty(1, 4), normalized, torch.empty(1, 2), 2, 4)
    with pytest.raises(ValueError, match="requires GPU tensors"):
        grouped_gemma_rmsnorm(normalized, torch.empty(8), 4, 1e-6)
    with pytest.raises(ValueError, match="requires GPU tensors"):
        gated_residual_combine_norm(
            torch.empty(1, 4),
            normalized,
            torch.empty(1, 2),
            torch.empty(8),
            2,
            4,
            1e-6,
            preload_residual=False,
        )


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
@pytest.mark.parametrize("per_branch_norm", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mix_fuses_previous_combine_with_its_own_norm(
    per_branch_norm: bool, dtype: torch.dtype
) -> None:
    config = HyperConnectionConfig(
        hc_count=3,
        hidden_size=13,
        hc_lowrank=7,
        rms_norm_eps=1e-6,
        params_dtype=dtype,
        hc_per_branch_norm=per_branch_norm,
    )
    previous = GatedResidualSimple(config, use_mix=True, use_combine=True).to(
        device="cuda", dtype=dtype
    )
    current = GatedResidualSimple(config, use_mix=True, use_combine=True).to(
        device="cuda", dtype=dtype
    )
    with torch.no_grad():
        previous.hc_norm.weight.fill_(0.25)
        current.hc_norm.weight.fill_(-0.125)
    residual = torch.randn(5, 39, device="cuda", dtype=dtype)
    block = torch.randn(5, 13, device="cuda", dtype=dtype)
    _, previous_residuals = previous.mix(
        residual, block_output=None, inject_logits=None, preload_residual=False
    )
    combined = previous.combine(block, previous_residuals)
    expected_mix, expected_residuals = current.mix(
        combined, block_output=None, inject_logits=None, preload_residual=False
    )

    with mock.patch.object(
        current.hc_norm,
        "forward",
        side_effect=AssertionError("unexpected separate norm"),
    ):
        actual_mix, actual_residuals = current.mix(
            residual,
            block_output=block,
            inject_logits=previous_residuals[2],
            preload_residual=True,
        )

    torch.testing.assert_close(actual_mix, expected_mix, rtol=0, atol=0)
    for actual, expected in zip(actual_residuals, expected_residuals, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The next injection must retain the unnormalized residual from the fused op.
    next_block = torch.randn_like(block)
    torch.testing.assert_close(
        current.combine(next_block, actual_residuals),
        current.combine(next_block, expected_residuals),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("communication", ["idle", "unchanged", "row_slice"])
def test_attention_to_mlp_fusion_after_communication(communication: str) -> None:
    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=8,
        hc_lowrank=6,
        rms_norm_eps=1e-6,
        params_dtype=torch.float32,
        hc_per_branch_norm=True,
    )
    attention = GatedResidualSimple(config, use_mix=True, use_combine=True).cuda()
    mlp = GatedResidualSimple(config, use_mix=True, use_combine=True).cuda()
    rows = 0 if communication == "idle" else 6
    residual = torch.randn(rows, 32, device="cuda")
    output = torch.randn(rows, 8, device="cuda")
    _, residuals = attention.mix(
        residual, block_output=None, inject_logits=None, preload_residual=False
    )
    row_slice = slice(2, 5) if communication == "row_slice" else slice(None)
    communicated_residual = residual[row_slice]
    communicated_output = output[row_slice]
    aligned_residuals = attention.norm_for(communicated_residual, residuals)
    combined = attention.combine(communicated_output, aligned_residuals)
    expected_mix, expected_residuals = mlp.mix(
        combined, block_output=None, inject_logits=None, preload_residual=False
    )
    post_attn_comm = mock.Mock(
        return_value=(communicated_output, communicated_residual)
    )
    layer = SimpleNamespace(
        attn_hyper_connection=attention,
        mlp_hyper_connection=mlp,
        comm_manager=SimpleNamespace(post_attn_comm=post_attn_comm),
    )
    ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: communication == "idle")
    )

    with mock.patch.object(
        attention, "combine", side_effect=AssertionError("unexpected separate combine")
    ):
        actual_mix, actual_residuals = _Qwen4ExpDecoderMixin._finish_attention(
            layer, output, residuals, ctx
        )

    torch.testing.assert_close(actual_mix, expected_mix, rtol=0, atol=0)
    for actual, expected in zip(actual_residuals, expected_residuals, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert post_attn_comm.call_count == (0 if communication == "idle" else 1)


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
    mixed, residuals = mixer.mix(
        hyper_input, block_output=None, inject_logits=None, preload_residual=False
    )
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


def test_runtime_declares_loaded_hc_weights_independent(monkeypatch) -> None:
    mixer = GatedResidualSimple(
        HyperConnectionConfig(hc_count=4, hidden_size=8, hc_lowrank=3)
    )
    value = torch.randn(2, 32)
    mixed = torch.randn(2, 8)
    inject = torch.randn(2, 4)
    monkeypatch.setattr(mixer, "_normalize", lambda x: x)
    call = mock.Mock(return_value=(mixed, inject))
    monkeypatch.setattr(hyperconnection_module, "gated_residual_mix", call)
    mixer.mix(value, block_output=None, inject_logits=None, preload_residual=False)
    assert call.call_args.kwargs["weights_independent"] is True
