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
    unpack_gluon_fp8_blockscale_weight,
)

_CANDIDATE = "gluon_mm_fp8_blockscale_largem_gfx950"
_DECODE = "gluon_mm_fp8_blockscale_decode_gfx950"
_PRIMARY_FLAG = "TOKENSPEED_EXPERIMENTAL_GLUON_FP8_BLOCKSCALE"


def test_packing_is_lossless_and_owns_storage() -> None:
    n, k = 128, 256
    weight = torch.linspace(-4, 4, n * k).reshape(n, k).to(torch.float8_e4m3fn)
    packed = pack_gluon_fp8_blockscale_weight(weight)
    restored = unpack_gluon_fp8_blockscale_weight(packed)
    assert packed.shape == weight.shape
    assert packed.data_ptr() != weight.data_ptr()
    torch.testing.assert_close(restored.float(), weight.float(), rtol=0, atol=0)


@pytest.mark.parametrize("m", [129, 848, 8144, 8192])
def test_prefill_rows_have_packed_gluon_dispatch(m: int) -> None:
    assert supports_gluon_fp8_blockscale_largem(m, 1024, 4096)
    assert not supports_gluon_fp8_blockscale_largem(128, 1024, 4096)
    assert not supports_gluon_fp8_blockscale_largem(m, 1024, 512)
    assert supports_gluon_fp8_blockscale_largem(1, 4096, 512)


def test_registration_does_not_accept_a_crossed_projection_shape() -> None:
    spec = KernelRegistry.get().get_by_name(_CANDIDATE)
    assert spec is not None
    assert spec_matches_shape_traits(spec, {"m": 8192, "n": 1024, "k": 4096})
    assert spec_matches_shape_traits(spec, {"m": 129, "n": 1024, "k": 4096})
    assert not spec_matches_shape_traits(spec, {"m": 128, "n": 1024, "k": 4096})
    assert spec_matches_shape_traits(spec, {"m": 1, "n": 4096, "k": 512})
    assert spec_matches_shape_traits(spec, {"m": 8192, "n": 6144, "k": 4096})
    assert not spec_matches_shape_traits(spec, {"m": 8192, "n": 1024, "k": 512})
    assert not spec_matches_traits(spec, {"a_scales_inner_stride_one": False})

    decode = KernelRegistry.get().get_by_name(_DECODE)
    assert decode is not None
    assert spec_matches_shape_traits(decode, {"m": 1, "n": 1024, "k": 4096})
    assert spec_matches_shape_traits(decode, {"m": 128, "n": 4096, "k": 1536})
    assert not spec_matches_shape_traits(decode, {"m": 129, "n": 1024, "k": 4096})
    assert not spec_matches_shape_traits(decode, {"m": 1, "n": 4096, "k": 512})
    assert not spec_matches_traits(spec, {"b_scales_inner_stride_one": False})


@pytest.mark.parametrize(
    ("enabled", "resident_scope", "shape", "expected_layout"),
    [
        (False, True, (6144, 4096), None),
        (True, False, (6144, 4096), None),
        (True, True, (6144, 4096), GLUON_BLOCK_FP8_WEIGHT_LAYOUT),
        (True, True, (1024, 4096), GLUON_BLOCK_FP8_WEIGHT_LAYOUT),
    ],
)
def test_plan_only_packs_after_explicit_promotion(
    enabled: bool,
    resident_scope: bool,
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
    plan = gemm.prepare_fp8_linear(
        weight, scales, (128, 128), packed_resident=resident_scope
    )
    assert plan.prepared_weight_layout is None
    assert plan.state_dict() == {}
    packed = gemm.promote_fp8_linear_weight(plan, weight)
    if expected_layout is not None:
        assert packed.shape == weight.shape
        assert plan.packed_resident
        assert plan.prepared_weight_layout == expected_layout
        assert plan.state_dict() == {}
        with pytest.raises(RuntimeError, match="already packed"):
            gemm.promote_fp8_linear_weight(plan, packed)
    else:
        assert packed is None
        assert not plan.packed_resident


def test_refresh_reuses_resident_buffer_and_export_is_canonical() -> None:
    gemm = importlib.import_module("tokenspeed_kernel.ops.gemm")
    canonical = torch.nn.Parameter(
        torch.zeros((128, 256), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    packed = pack_gluon_fp8_blockscale_weight(canonical)
    plan = gemm._PreparedFp8Linear(
        override=None,
        block_size=(128, 128),
        pack_candidate=True,
        packed_resident=True,
        prepared_weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    packed_ptr = packed.data_ptr()
    plan.packed_weight_ptr = packed_ptr
    plan.packed_valid = True
    plan.packed_intact = True
    with torch.no_grad():
        canonical.copy_(torch.ones_like(canonical))
    assert gemm.refresh_fp8_linear_weight(plan, canonical, packed)
    assert packed.data_ptr() == packed_ptr
    torch.testing.assert_close(
        gemm.export_fp8_linear_weight(plan, packed).float(),
        canonical.float(),
        rtol=0,
        atol=0,
    )
    assert plan.state_dict() == {}
    assert gemm.invalidate_fp8_linear_weight(plan)
    assert not plan.packed_valid
    with pytest.raises(RuntimeError, match="not ready for export"):
        gemm.export_fp8_linear_weight(plan, packed)
    assert gemm.rebind_fp8_linear_weight(plan, packed)
    assert plan.packed_valid
    assert gemm.invalidate_fp8_linear_weight(plan)
    plan.packed_intact = False
    with pytest.raises(RuntimeError, match="invalid"):
        gemm.rebind_fp8_linear_weight(plan, packed)


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

    fake = FakeKernel()
    monkeypatch.setattr(gemm, "select_kernel", lambda *args, **kwargs: fake)
    args = (
        torch.empty((2, 128), dtype=torch.float8_e4m3fn),
        torch.empty((128, 128), dtype=torch.float8_e4m3fn),
    )
    kwargs = dict(
        A_scales=torch.ones((2, 1)),
        B_scales=torch.ones((1, 1)),
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override=_CANDIDATE,
        weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    gemm.mm(*args, **kwargs)
    assert len(calls) == 1
    assert calls[0][1]["weight_layout"] == GLUON_BLOCK_FP8_WEIGHT_LAYOUT
    fake.name = "triton_mm_fp8_blockscale"
    with pytest.raises(ValueError, match="does not match"):
        gemm.mm(*args, **kwargs)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(4, 1024, 4096), (4, 4096, 512), (129, 1024, 4096), (8144, 1024, 4096)],
)
def test_packed_plan_never_drops_to_canonical_dispatch(
    m: int, n: int, k: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemm = importlib.import_module("tokenspeed_kernel.ops.gemm")
    calls: list[dict[str, object]] = []

    def capture_mm(*args, **kwargs):
        calls.append(kwargs)
        return torch.empty((m, n), dtype=torch.float16)

    monkeypatch.setattr(gemm, "mm", capture_mm)
    plan = gemm._PreparedFp8Linear(
        override=None,
        block_size=(128, 128),
        pack_candidate=True,
        packed_resident=True,
        prepared_weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    weight = torch.empty((n, k), dtype=torch.float8_e4m3fn)
    plan.packed_valid = True
    plan.packed_weight_ptr = weight.data_ptr()
    gemm.fp8_linear(
        plan,
        torch.empty((m, k), dtype=torch.float16),
        weight,
        torch.empty((n // 128, k // 128)),
        out_dtype=torch.float16,
    )
    assert calls[0]["weight_layout"] == GLUON_BLOCK_FP8_WEIGHT_LAYOUT
    assert calls[0]["override"] == (_CANDIDATE if m >= 129 or k == 512 else _DECODE)


def test_packed_plan_checks_storage_and_reload_state() -> None:
    gemm = importlib.import_module("tokenspeed_kernel.ops.gemm")
    first = torch.empty((128, 256), dtype=torch.float8_e4m3fn)
    second = torch.empty_like(first)
    plan = gemm._PreparedFp8Linear(
        override=None,
        block_size=(128, 128),
        pack_candidate=True,
        packed_resident=True,
        prepared_weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    plan.packed_valid = True
    plan.packed_intact = True
    plan.packed_weight_ptr = first.data_ptr()
    with pytest.raises(RuntimeError, match="not ready"):
        gemm.fp8_linear(plan, torch.empty((4, 256)), second, torch.empty((1, 2)))
    assert gemm.rebind_fp8_linear_weight(plan, second)
    assert plan.packed_weight_ptr == second.data_ptr()
    with pytest.raises(RuntimeError, match="lost its storage layout"):
        gemm.rebind_fp8_linear_weight(plan, second.float())
    assert plan.packed_weight_ptr == second.data_ptr()
    assert gemm.invalidate_fp8_linear_weight(plan)
    with pytest.raises(RuntimeError, match="not ready"):
        gemm.fp8_linear(plan, torch.empty((4, 256)), second, torch.empty((1, 2)))


def test_refresh_pack_failure_keeps_previous_packed_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemm = importlib.import_module("tokenspeed_kernel.ops.gemm")
    packed_module = importlib.import_module("tokenspeed_kernel_amd.ops.gfx950.gemm.fp8")
    canonical = torch.zeros((128, 256), dtype=torch.float8_e4m3fn)
    packed = pack_gluon_fp8_blockscale_weight(canonical)
    plan = gemm._PreparedFp8Linear(
        override=None,
        block_size=(128, 128),
        pack_candidate=True,
        packed_resident=True,
        prepared_weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    plan.packed_weight_ptr = packed.data_ptr()
    plan.packed_valid = True
    plan.packed_intact = True
    assert gemm.invalidate_fp8_linear_weight(plan)

    def fail_pack(*args, **kwargs):
        raise RuntimeError("pack failed")

    monkeypatch.setattr(packed_module, "pack_gluon_fp8_blockscale_weight", fail_pack)
    with pytest.raises(RuntimeError, match="pack failed"):
        gemm.refresh_fp8_linear_weight(plan, torch.ones_like(canonical), packed)
    torch.testing.assert_close(
        unpack_gluon_fp8_blockscale_weight(packed).float(),
        canonical.float(),
        rtol=0,
        atol=0,
    )
    assert plan.packed_valid and plan.packed_intact
    torch.testing.assert_close(
        gemm.export_fp8_linear_weight(plan, packed).float(),
        canonical.float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("restore_fails", [False, True])
def test_refresh_copy_failure_restores_or_marks_weight_fatal(
    restore_fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemm = importlib.import_module("tokenspeed_kernel.ops.gemm")
    old = torch.zeros((128, 256), dtype=torch.float8_e4m3fn)
    packed = pack_gluon_fp8_blockscale_weight(old)
    plan = gemm._PreparedFp8Linear(
        override=None,
        block_size=(128, 128),
        pack_candidate=True,
        packed_resident=True,
        prepared_weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    plan.packed_weight_ptr = packed.data_ptr()
    plan.packed_valid = True
    plan.packed_intact = True
    original_copy = torch.Tensor.copy_
    copies_to_resident = 0

    def injected_copy(destination, source, *args, **kwargs):
        nonlocal copies_to_resident
        if destination.data_ptr() == packed.data_ptr():
            copies_to_resident += 1
            if copies_to_resident == 1:
                original_copy(destination.flatten()[:64], source.flatten()[:64])
                raise RuntimeError("injected partial packed copy")
            if restore_fails:
                raise RuntimeError("injected restore failure")
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", injected_copy)
    expected_error = (
        gemm.PackedFp8WeightCorruptionError if restore_fails else RuntimeError
    )
    with pytest.raises(expected_error):
        gemm.refresh_fp8_linear_weight(plan, torch.ones_like(old), packed)
    assert copies_to_resident == 2
    if restore_fails:
        assert not plan.packed_valid
        assert not plan.packed_intact
        with pytest.raises(RuntimeError, match="not ready for export"):
            gemm.export_fp8_linear_weight(plan, packed)
    else:
        assert plan.packed_valid and plan.packed_intact
        torch.testing.assert_close(
            unpack_gluon_fp8_blockscale_weight(packed).float(),
            old.float(),
            rtol=0,
            atol=0,
        )
