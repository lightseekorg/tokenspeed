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

"""Behavioral CPU checks for the shape-derived FP32 launch policy."""

from __future__ import annotations

import importlib

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import _gemm_format_signature
from tokenspeed_kernel.ops.gemm.gluon import gluon_mm_fp8_blockscale_gfx950
from tokenspeed_kernel.selection import select_kernel

mm = importlib.import_module("tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.mm")


@pytest.mark.parametrize("k,decode", [(3072, True), (3104, False)])
def test_public_fp32_workspace_and_argument_forwarding(monkeypatch, k, decode):
    registration = importlib.import_module("tokenspeed_kernel.ops.gemm.gluon")
    a = torch.empty((17, k), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((128, k), device="meta", dtype=torch.float8_e4m3fn)
    sa = torch.empty((17, (k + 127) // 128), device="meta")
    sb = torch.empty((1, (k + 127) // 128), device="meta")
    out = torch.empty((17, 128), device="meta", dtype=torch.bfloat16)
    alpha = torch.empty((), device="meta")
    block_size = (128, 128)
    counters = torch.empty(
        mm.gluon_mm_mxfp8_decode_num_counters_gfx950(17, 128, k),
        device="meta",
        dtype=torch.int32,
    )

    def allocate(size, *, dtype, device):
        assert decode
        assert size == counters.numel() and dtype == counters.dtype
        assert device == a.device
        return counters

    # Only the unavailable GPU launch and its device workspace are replaced.
    def launch(aa, bb, ascales, bscales, dtype, **kwargs):
        assert aa is a and bb is b and ascales is sa and bscales is sb
        assert dtype == out.dtype and kwargs["out"] is out
        assert kwargs["alpha"] is alpha and kwargs["block_size"] is block_size
        assert kwargs.get("counters") is (counters if decode else None)
        return out

    monkeypatch.setattr(registration, "zeroed_state", allocate)
    monkeypatch.setattr(
        mm,
        (
            "launch_gluon_mm_mxfp8_decode_gfx950"
            if decode
            else "launch_gluon_mm_mxfp8_gfx950"
        ),
        launch,
    )
    assert (
        gluon_mm_fp8_blockscale_gfx950(
            a, b, sa, sb, out.dtype, alpha=alpha, block_size=block_size, out=out
        )
        is out
    )


@pytest.mark.parametrize(
    "missing_scale,block_size",
    [("activation", [128, 128]), ("weight", [128, 128]), (None, [1, 32])],
)
def test_public_fp32_requires_canonical_scales(missing_scale, block_size):
    a = torch.empty((1, 512), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((128, 512), device="meta", dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 4), device="meta")
    with pytest.raises(ValueError, match="both scales and block_size"):
        gluon_mm_fp8_blockscale_gfx950(
            a,
            b,
            None if missing_scale == "activation" else scale,
            None if missing_scale == "weight" else scale,
            torch.bfloat16,
            alpha=None,
            block_size=block_size,
        )


@pytest.mark.parametrize(
    "shape,expected,hint",
    [
        ((1, 1024, 4096), (64, 64, 4, 8), 0),
        ((64, 1024, 4096), (64, 64, 4, 8), 2),
        ((65, 1024, 4096), (64, 64, 4, 6), 2),
        ((128, 1024, 4096), (64, 64, 4, 8), 2),
        ((129, 4096, 512), (64, 64, 4, 1), 2),
        ((16, 4096, 1536), (64, 64, 4, 3), 2),
        ((300, 256, 384), (64, 64, 4, 1), 2),
        ((896, 6144, 4096), (128, 128, 8, 1), 2),
        ((1616, 1024, 4096), (128, 128, 8, 2), 2),
        ((1616, 2048, 4096), (128, 128, 8, 1), 2),
        ((3536, 4096, 512), (256, 256, 8, 1), 2),
    ],
)
def test_fp32_shape_policy_boundaries(shape, expected, hint):
    config = mm._fp32_launch_config(*shape)
    assert tuple(config) == expected
    assert mm._fp32_pipeline_waves_per_eu(*shape, config) == hint


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 16, 288),
        (37, 208, 800),
        (129, 144, 1056),
        (641, 1152, 576),
        (641, 1152, 5120),
    ],
)
def test_fp32_ragged_partitions_have_work_and_legal_wave_tiles(m, n, k):
    config = mm._fp32_launch_config(m, n, k)
    assert ((k + 255) // 256) // config.splits >= 2
    assert config.waves in (4, 8)


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((128, 1024, 4096), (4, 64)),
        ((129, 1024, 4096), (16, 256)),
        ((128, 256, 4096), (16, 256)),
        ((128, 1024, 576), (16, 256)),
    ],
)
def test_reducer_family_boundaries(shape, expected):
    assert mm._fp32_reducer_tile(*shape) == expected


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize(
    "shape,output_dtype,expected",
    [
        ((1, 1024, 4096), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((17, 130, 3072), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((17, 128, 3104), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((129, 4096, 512), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((896, 6144, 4096), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((37, 208, 800), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((4, 128, 256), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((4, 130, 256), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((256, 130, 256), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((257, 4096, 512), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((1, 15, 128), torch.bfloat16, "triton_mm_fp8_blockscale"),
        ((1, 16, 128), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((1, 16, 96), torch.bfloat16, "triton_mm_fp8_blockscale"),
        ((63, 32768, 2048), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((64, 32768, 2048), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((1, 1048560, 2048), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((1, 1048576, 2048), torch.bfloat16, "gluon_mm_fp8_blockscale_gfx950"),
        ((4, 128, 512), torch.float16, "triton_mm_fp8_blockscale"),
    ],
)
def test_registry_fp32_contract_and_fallback(shape, output_dtype, expected, online):
    m, n, k = shape
    a = torch.empty(
        (m, k), device="meta", dtype=torch.bfloat16 if online else torch.float8_e4m3fn
    )
    b = torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn)
    sa = None if online else torch.empty((m, (k + 127) // 128), device="meta")
    sb = torch.empty(((n + 127) // 128, (k + 127) // 128), device="meta")
    signature = _gemm_format_signature(a, b, sa, sb, output_dtype, "mxfp8", [128, 128])
    selected = select_kernel(
        "gemm",
        "mm",
        signature,
        traits={
            "m": m,
            "n": n,
            "k": k,
            "b_layout": "NK",
            "a_inner_stride_one": True,
            "a_scales_inner_stride_one": True,
            "b_inner_stride_one": True,
            "b_scales_inner_stride_one": True,
            "block_scale_layout": "canonical",
            "out_dtype": output_dtype,
            "out_inner_stride_one": True,
            "pdl_enabled": False,
        },
    )
    assert selected.name == expected


@pytest.mark.parametrize(
    "trait,expected",
    [
        ("a_inner_stride_one", "triton_mm_fp8_blockscale"),
        ("b_inner_stride_one", "triton_mm_fp8_blockscale"),
        ("out_inner_stride_one", "triton_mm_fp8_blockscale"),
        ("a_scales_inner_stride_one", "gluon_mm_fp8_blockscale_gfx950"),
        ("b_scales_inner_stride_one", "gluon_mm_fp8_blockscale_gfx950"),
    ],
)
def test_registry_value_and_scale_stride_boundaries(trait, expected):
    a = torch.empty((4, 512), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((128, 512), device="meta", dtype=torch.float8_e4m3fn)
    sa = torch.empty((4, 4), device="meta")
    sb = torch.empty((1, 4), device="meta")
    signature = _gemm_format_signature(
        a, b, sa, sb, torch.bfloat16, "mxfp8", [128, 128]
    )
    traits = {
        "m": 4,
        "n": 128,
        "k": 512,
        "b_layout": "NK",
        "a_inner_stride_one": True,
        "b_inner_stride_one": True,
        "a_scales_inner_stride_one": True,
        "b_scales_inner_stride_one": True,
        "block_scale_layout": "canonical",
        "out_dtype": torch.bfloat16,
        "out_inner_stride_one": True,
        "pdl_enabled": False,
    }
    traits[trait] = False
    assert select_kernel("gemm", "mm", signature, traits=traits).name == expected
