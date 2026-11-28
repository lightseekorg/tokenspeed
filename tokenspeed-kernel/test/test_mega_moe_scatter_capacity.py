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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Unit tests for token_scatter_size workspace sizing in MegaMoE runners."""

from __future__ import annotations

import sys
import types

import pytest
import torch
from tokenspeed_kernel.ops import tuning


def _install_fake_flashinfer_kimi(monkeypatch):
    fake_fi = types.ModuleType("flashinfer")
    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")
    fake_ksrc = types.ModuleType("flashinfer.moe_ep.kernel_src")
    fake_sm100 = types.ModuleType("flashinfer.moe_ep.kernel_src.sm100")
    fake_kimi = types.ModuleType("flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe")
    fake_src = types.ModuleType(
        "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src"
    )
    fake_shim = types.ModuleType(
        "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.shim"
    )
    fake_runtime = types.ModuleType(
        "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.shim.runtime"
    )
    fake_helpers = types.ModuleType(
        "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src.helpers"
    )
    fake_epoch = types.ModuleType(
        "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src.helpers.epoch"
    )
    fake_dw = types.ModuleType(
        "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src.helpers.device_workspace"
    )
    fake_dw.DeviceWorkspace = object
    fake_runtime.KimiWorkspace = object
    fake_helpers.epoch = fake_epoch
    fake_helpers.device_workspace = fake_dw
    fake_shim.runtime = fake_runtime
    fake_src.helpers = fake_helpers
    fake_kimi.shim = fake_shim
    fake_kimi.src = fake_src
    fake_sm100._kimi_k3_megamoe = fake_kimi
    fake_ksrc.sm100 = fake_sm100
    fake_moe_ep.kernel_src = fake_ksrc
    fake_fi.moe_ep = fake_moe_ep
    for name, mod in (
        ("flashinfer", fake_fi),
        ("flashinfer.moe_ep", fake_moe_ep),
        ("flashinfer.moe_ep.kernel_src", fake_ksrc),
        ("flashinfer.moe_ep.kernel_src.sm100", fake_sm100),
        ("flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe", fake_kimi),
        ("flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.shim", fake_shim),
        (
            "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.shim.runtime",
            fake_runtime,
        ),
        ("flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src", fake_src),
        (
            "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src.helpers",
            fake_helpers,
        ),
        (
            "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src.helpers.epoch",
            fake_epoch,
        ),
        (
            "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.src.helpers.device_workspace",
            fake_dw,
        ),
    ):
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)


def test_kimi_mega_moe_runner_divides_capacity_by_token_scatter_size(
    monkeypatch,
) -> None:
    _install_fake_flashinfer_kimi(monkeypatch)
    sys.modules.pop("tokenspeed_kernel.thirdparty.flashinfer.kimi_mega_moe", None)
    from tokenspeed_kernel.thirdparty.flashinfer import kimi_mega_moe

    created: list[int] = []

    class _FakeRunner:
        def __init__(
            self,
            group,
            num_experts,
            hidden_size,
            intermediate_size,
            top_k,
            situ_beta,
            situ_linear_beta,
            max_tokens,
            weights,
        ) -> None:
            self.max_tokens = max_tokens
            created.append(max_tokens)

    monkeypatch.setattr(kimi_mega_moe, "KimiMegaMoERunner", _FakeRunner)
    monkeypatch.setattr(kimi_mega_moe, "_runners", {})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(tuning, "_autotune_max_num_tokens", 8192)

    weights = types.SimpleNamespace(
        num_experts=896,
        hidden_size=3584,
        intermediate_size=3072,
        top_k=16,
        activation_situ_beta=1.0,
        activation_situ_linear_beta=7.0,
        _moe_backend_state=object(),
    )
    group = object()

    r1 = kimi_mega_moe.get_kimi_mega_moe_runner({"process_group": group}, weights)
    r8 = kimi_mega_moe.get_kimi_mega_moe_runner(
        {"process_group": group, "token_scatter_size": 8}, weights
    )
    r16 = kimi_mega_moe.get_kimi_mega_moe_runner(
        {"process_group": group, "token_scatter_size": 16}, weights
    )
    assert (r1.max_tokens, r8.max_tokens, r16.max_tokens) == (8192, 1024, 512)
    assert created == [8192, 1024, 512]

    with pytest.raises(ValueError, match="token_scatter_size must be >= 1"):
        kimi_mega_moe.get_kimi_mega_moe_runner(
            {"process_group": group, "token_scatter_size": 0}, weights
        )
