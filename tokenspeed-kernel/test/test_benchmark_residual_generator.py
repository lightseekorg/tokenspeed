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
import tokenspeed_kernel.benchmark.generators.residual as residual_generator
import torch
from tokenspeed_kernel.benchmark.harness import BenchmarkCaseError, BenchmarkRequest


def _request(mode: str, **parameters: object) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="residual",
        mode=mode,
        parameters={
            "model_profile": "dsv41_flash_tp4",
            "tokens": 192,
            "hc_mult": 4,
            "hidden_size": 5120,
            "dtype": "bfloat16",
            "sinkhorn_iters": 20,
            "rms_eps": 1e-20,
            "hc_eps": 1e-6,
            **parameters,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


def test_mhc_post_generator_selects_with_operation_traits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_selection(family, mode, _signature, **kwargs):
        captured.update(family=family, mode=mode, traits=kwargs["traits"])
        raise RuntimeError("selection captured")

    monkeypatch.setattr(residual_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(residual_generator, "select_kernel", capture_selection)

    with pytest.raises(RuntimeError, match="selection captured"):
        residual_generator.prepare_mhc_post(_request("mhc_post"), None)

    assert captured == {
        "family": "residual",
        "mode": "mhc_post",
        "traits": {"num_tokens": 192, "hc_mult": 4, "hidden_size": 5120},
    }


@pytest.mark.parametrize("fused_norm", [True, False])
def test_mhc_apply_pre_generator_selection_matches_operation_api(
    monkeypatch: pytest.MonkeyPatch, fused_norm: bool
) -> None:
    from tokenspeed_kernel.ops import residual as residual_ops

    captured: dict[str, object] = {}

    def capture(source: str):
        def select(family, mode, signature, **kwargs):
            captured[source] = (family, mode, signature, kwargs["traits"])
            raise RuntimeError("selection captured")

        return select

    monkeypatch.setattr(residual_ops, "select_kernel", capture("operation"))
    monkeypatch.setattr(residual_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(residual_generator, "select_kernel", capture("generator"))

    residual = torch.empty((192, 4, 5120), dtype=torch.bfloat16, device="meta")
    pre = torch.empty((192, 4), dtype=torch.float32, device="meta")
    norm_weight = torch.empty((5120,), dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError, match="selection captured"):
        residual_ops.mhc_apply_pre(
            residual,
            pre,
            norm_weight=norm_weight if fused_norm else None,
            norm_eps=1e-20 if fused_norm else None,
        )
    request = _request("mhc_apply_pre", fused_norm=fused_norm)
    if not fused_norm:
        del request.parameters["rms_eps"]
    with pytest.raises(RuntimeError, match="selection captured"):
        residual_generator.prepare_mhc_apply_pre(request, None)

    assert captured["generator"] == captured["operation"]


@pytest.mark.parametrize(
    ("prepare", "mode", "overrides", "message"),
    [
        (
            residual_generator.prepare_mhc_mixes,
            "mhc_mixes",
            {"hc_mult": 2},
            "hc_mult 4",
        ),
        (
            residual_generator.prepare_mhc_post,
            "mhc_post",
            {"model_profile": "kimi_k3_tp8"},
            "mHC model_profile",
        ),
        (
            residual_generator.prepare_mhc_apply_pre,
            "mhc_apply_pre",
            {"fused_norm": False},
            "rms_eps exactly when fused_norm",
        ),
    ],
)
def test_mhc_generators_reject_unimplemented_cases(
    prepare, mode: str, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(BenchmarkCaseError, match=message):
        prepare(_request(mode, **overrides), None)
