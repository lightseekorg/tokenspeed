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
        "traits": {
            "num_tokens": 192,
            "buffer_offsets_fit_int32": True,
            "hc_mult": 4,
            "hidden_size": 5120,
        },
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("inject", [True, False])
def test_hyperconnection_mix_generator_selection_matches_operation_api(
    monkeypatch: pytest.MonkeyPatch, inject: bool
) -> None:
    from tokenspeed_kernel.ops import residual as residual_ops

    captured: list[tuple] = []

    def select(family, mode, signature, **kwargs):
        captured.append((family, mode, signature, kwargs["traits"]))
        raise RuntimeError("selection captured")

    monkeypatch.setattr(residual_ops, "select_kernel", select)
    monkeypatch.setattr(residual_generator, "load_builtin_kernels", lambda: None)

    hc_count, hidden_size, lowrank, tokens = 4, 256, 32, 16
    wide = hc_count * hidden_size
    normalized = torch.empty((tokens, wide), dtype=torch.bfloat16, device="cuda")
    projection = torch.empty(
        (lowrank + (hc_count if inject else 0), wide),
        dtype=torch.bfloat16,
        device="cuda",
    )
    up = torch.empty((wide, lowrank), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="selection captured"):
        residual_ops.gated_residual_mix(
            normalized,
            projection,
            up,
            hc_count,
            hidden_size,
            lowrank,
            projection_rows=lowrank + (hc_count if inject else 0),
            weights_independent=True,
        )
    request = BenchmarkRequest(
        family="residual",
        mode="hyperconnection_mix",
        parameters={
            "model_profile": "qwen38_flash_tp2",
            "tokens": tokens,
            "hc_count": hc_count,
            "hidden_size": hidden_size,
            "lowrank": lowrank,
            "inject": inject,
            "dtype": "bfloat16",
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )
    with pytest.raises(RuntimeError, match="selection captured"):
        residual_generator.prepare_hyperconnection_mix(request, None)

    assert captured[0] == captured[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("inject", [True, False])
def test_hyperconnection_mix_generator_invokes_with_padded_storage(
    monkeypatch: pytest.MonkeyPatch, inject: bool
) -> None:
    from tokenspeed_kernel.ops import residual as residual_ops

    hc_count, hidden_size, lowrank, tokens = 4, 32, 5, 16
    projection_rows = lowrank + (hc_count if inject else 0)
    select = residual_ops.select_gated_residual_mix_kernel
    selected_weights: list[torch.Tensor] = []

    def capture_selection(normalized, projection_weight, *args, **kwargs):
        selected_weights.append(projection_weight)
        return select(normalized, projection_weight, *args, **kwargs)

    monkeypatch.setattr(
        residual_ops, "select_gated_residual_mix_kernel", capture_selection
    )
    request = _request(
        "hyperconnection_mix",
        model_profile="qwen38_flash_tp2",
        tokens=tokens,
        hc_count=hc_count,
        hidden_size=hidden_size,
        lowrank=lowrank,
        inject=inject,
    )
    prepared = residual_generator.prepare_hyperconnection_mix(request, None)
    mixed, inject_logits = prepared.invocation.invoke()

    assert mixed.shape == (tokens, hidden_size)
    if inject:
        assert inject_logits.shape == (tokens, hc_count)
    else:
        assert inject_logits is None
    prepared_weight, invoked_weight = selected_weights
    assert invoked_weight is prepared_weight
    assert prepared_weight.shape == (
        (projection_rows + 7) // 8 * 8,
        hc_count * hidden_size,
    )
    assert torch.count_nonzero(prepared_weight[projection_rows:]) == 0


def _boundary_request(**updates: object) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="residual",
        mode="post_norm_residual_pre_norm",
        parameters={
            "model_profile": "muse_spark_tp8",
            "tokens": 3,
            "hidden_size": 256,
            "post_weighted": True,
            "pre_weighted": True,
            "outputs": "bf16",
            "post_eps": 1e-8,
            "pre_eps": 1e-5,
            "gate_eps": 1e-3,
            "gate_temperature": 0.3,
            "weight_offset": 1.0,
            "dtype": "bfloat16",
            **updates,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"outputs": "fp32"}, "outputs"),
        ({"post_eps": 0.0}, "post_eps"),
        ({"pre_weighted": 1}, "pre_weighted"),
        ({"model_profile": "kimi_k3_tp8"}, "model_profile"),
    ],
)
def test_post_norm_residual_pre_norm_rejects_invalid_parameters(updates, match) -> None:
    with pytest.raises(BenchmarkCaseError, match=match):
        residual_generator.prepare_post_norm_residual_pre_norm(
            _boundary_request(**updates), None
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    ("updates", "outputs"),
    [
        ({"outputs": "bf16"}, (torch.float32, torch.bfloat16)),
        (
            {"outputs": "fp32_and_bf16", "post_weighted": False},
            (torch.float32, torch.float32, torch.bfloat16),
        ),
        (
            {"outputs": "fp32_and_bf16", "pre_weighted": False},
            (torch.float32, None, None),
        ),
    ],
)
def test_post_norm_residual_pre_norm_returns_the_selected_entrys_outputs(
    updates, outputs
) -> None:
    from tokenspeed_kernel.platform import current_platform

    prepared = residual_generator.prepare_post_norm_residual_pre_norm(
        _boundary_request(**updates), current_platform()
    )
    results = prepared.invocation.invoke()

    assert prepared.registration.mode == "post_norm_residual_pre_norm"
    assert tuple(None if r is None else r.dtype for r in results) == outputs
    assert all(r is None or r.shape == (3, 256) for r in results)
