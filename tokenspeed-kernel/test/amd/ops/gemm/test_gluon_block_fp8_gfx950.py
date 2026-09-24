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

"""Focused layout and dispatch tests for the gfx950 block-FP8 GEMM."""

from __future__ import annotations

import importlib

import pytest
import torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import spec_matches_shape_traits, spec_matches_traits
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8 import (
    GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    pack_gluon_fp8_blockscale_weight,
    supports_gluon_fp8_blockscale_largem,
)

_CANDIDATE = "gluon_mm_fp8_blockscale_largem_gfx950"
_PRIMARY_FLAG = "TOKENSPEED_EXPERIMENTAL_GLUON_FP8_BLOCKSCALE"


def test_packing_is_lossless_and_owns_storage() -> None:
    n, k = 128, 256
    weight = torch.linspace(-4, 4, n * k).reshape(n, k).to(torch.float8_e4m3fn)
    packed = pack_gluon_fp8_blockscale_weight(weight)
    restored = (
        packed.view(k // 128, n // 64, 128, 64)
        .permute(1, 3, 0, 2)
        .contiguous()
        .view(n, k)
    )
    assert packed.shape == weight.shape
    assert packed.data_ptr() != weight.data_ptr()
    torch.testing.assert_close(restored.float(), weight.float(), rtol=0, atol=0)


@pytest.mark.parametrize("m", [8144, 8192])
def test_only_production_rows_have_private_dispatch(m: int) -> None:
    assert supports_gluon_fp8_blockscale_largem(m, 1024, 4096)
    assert not supports_gluon_fp8_blockscale_largem(848, 1024, 4096)
    assert not supports_gluon_fp8_blockscale_largem(m, 1024, 512)


def test_registration_does_not_accept_a_crossed_projection_shape() -> None:
    spec = KernelRegistry.get().get_by_name(_CANDIDATE)
    assert spec is not None
    assert spec_matches_shape_traits(spec, {"m": 8192, "n": 1024, "k": 4096})
    assert spec_matches_shape_traits(spec, {"m": 8192, "n": 6144, "k": 4096})
    assert not spec_matches_shape_traits(spec, {"m": 8192, "n": 1024, "k": 512})
    assert not spec_matches_traits(spec, {"a_scales_inner_stride_one": False})
    assert not spec_matches_traits(spec, {"b_scales_inner_stride_one": False})


@pytest.mark.parametrize(
    ("enabled", "shape", "expected_layout"),
    [
        (False, (6144, 4096), None),
        (True, (6144, 4096), GLUON_BLOCK_FP8_WEIGHT_LAYOUT),
        (True, (1024, 4096), GLUON_BLOCK_FP8_WEIGHT_LAYOUT),
    ],
)
def test_plan_packs_once_for_selected_layout(
    enabled: bool,
    shape: tuple[int, int],
    expected_layout: str | None,
    monkeypatch: pytest.MonkeyPatch,
    mi350_platform,
) -> None:
    monkeypatch.setattr(Platform, "_instance", mi350_platform)
    monkeypatch.setenv(_PRIMARY_FLAG, "1" if enabled else "0")
    gemm = importlib.import_module("tokenspeed_kernel.ops.gemm")
    registry = KernelRegistry.get()
    monkeypatch.setattr(registry, "get_by_name", lambda _name: object())
    n, k = shape
    weight = torch.empty(shape, device="meta", dtype=torch.float8_e4m3fn)
    scales = torch.empty((n // 128, k // 128), device="meta")
    plan = gemm.prepare_fp8_linear(weight, scales, (128, 128))
    assert plan.prepared_weight_layout == expected_layout
    if expected_layout is not None:
        assert plan.override == _CANDIDATE
        assert plan.prepared_weight.shape == weight.shape
        assert plan.state_dict() == {}
        assert plan.prepared_weight_is_current(weight)
        assert "prepared_weight" in plan._non_persistent_buffers_set
    else:
        assert plan.prepared_weight is None


def test_refresh_reuses_one_buffer_and_invalidates_stale_weight() -> None:
    gemm = importlib.import_module("tokenspeed_kernel.ops.gemm")
    canonical = torch.nn.Parameter(
        torch.zeros((128, 256), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    packed = pack_gluon_fp8_blockscale_weight(canonical)
    plan = gemm._PreparedFp8Linear(
        override=_CANDIDATE,
        block_size=(128, 128),
        prepared_weight=packed,
        prepared_weight_source=canonical,
        prepared_weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
        eligible_rows=frozenset({8144, 8192}),
    )
    packed_ptr = packed.data_ptr()
    with torch.no_grad():
        canonical.copy_(torch.ones_like(canonical))
    assert not plan.prepared_weight_is_current(canonical)
    assert gemm.refresh_fp8_linear_weight(plan, canonical)
    assert plan.prepared_weight_is_current(canonical)
    assert plan.prepared_weight.data_ptr() == packed_ptr
    assert plan.state_dict() == {}
    assert gemm.invalidate_fp8_linear_weight(plan)
    assert not plan.prepared_weight_is_current(canonical)


def test_mm_passes_private_layout_only_to_registered_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemm = importlib.import_module("tokenspeed_kernel.ops.gemm")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeKernel:
        name = _CANDIDATE

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return torch.empty((2, 128), dtype=torch.bfloat16)

    monkeypatch.setattr(gemm, "select_kernel", lambda *args, **kwargs: FakeKernel())
    gemm.mm(
        torch.empty((2, 128), dtype=torch.float8_e4m3fn),
        torch.empty((128, 128), dtype=torch.float8_e4m3fn),
        A_scales=torch.ones((2, 1)),
        B_scales=torch.ones((1, 1)),
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override=_CANDIDATE,
        weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    assert len(calls) == 1
    assert calls[0][1]["weight_layout"] == GLUON_BLOCK_FP8_WEIGHT_LAYOUT
