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

from types import SimpleNamespace

import pytest
import torch
from utils import is_cdna5

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for gfx1250 MXFP8 GEMV tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8 import (  # noqa: E402
    decode_mm as mxfp8_mm,
)
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8 import (  # noqa: E402
    gluon_mm_fp8_blockscale_gfx1250,
    gluon_mm_mxfp8_ue8m0_gfx1250,
)

# Inputs dequantize exactly; leave a small margin above BF16's worst-case
# relative rounding error (2**-8) when comparing against FP32 references.
_BF16_ATOL = 4e-3
_BF16_RTOL = 4e-3


def _random_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape, device="cuda") * 0.25).to(torch.float8_e4m3fn)


def _ue8m0_dequantize(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    decoded = scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
    return values.float() * decoded[:, : values.shape[1]]


def _fp32_block_dequantize(
    values: torch.Tensor,
    scales: torch.Tensor,
    row_block: int,
) -> torch.Tensor:
    rows = scales.repeat_interleave(row_block, dim=0)
    expanded = rows.repeat_interleave(128, dim=1)
    return values.float() * expanded[: values.shape[0], : values.shape[1]]


@pytest.mark.parametrize("m,n,k", [(1, 128, 256), (16, 256, 288)])
def test_mxfp8_ue8m0_gemv_matches_dequantized_reference(
    m: int,
    n: int,
    k: int,
) -> None:
    torch.manual_seed(0)
    a = _random_fp8((m, k))
    b = _random_fp8((n, k))
    a_scales = torch.randint(124, 130, (m, k // 32), device="cuda", dtype=torch.uint8)
    b_scales = torch.randint(124, 130, (n, k // 32), device="cuda", dtype=torch.uint8)

    actual = gluon_mm_mxfp8_ue8m0_gfx1250(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=[1, 32],
    )
    expected = (
        _ue8m0_dequantize(a, a_scales)
        @ _ue8m0_dequantize(
            b,
            b_scales,
        ).T
    )

    torch.testing.assert_close(
        actual.float(), expected, atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 128, 128), (1, 128, 256), (16, 256, 512), (1, 128, 4096)],
)
def test_fp8_blockscale_gemv_matches_dequantized_reference(
    m: int,
    n: int,
    k: int,
) -> None:
    torch.manual_seed(1)
    a = _random_fp8((m, k))
    b = _random_fp8((n, k))
    a_scales = torch.rand((m, k // 128), device="cuda") + 0.5
    b_scales = torch.rand((n // 128, k // 128), device="cuda") + 0.5

    actual = gluon_mm_fp8_blockscale_gfx1250(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=[128, 128],
    )
    expected = (
        _fp32_block_dequantize(a, a_scales, 1)
        @ _fp32_block_dequantize(b, b_scales, 128).T
    )

    torch.testing.assert_close(
        actual.float(), expected, atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize(
    "contract,n,k,split_k",
    [("ue8m0", 576, 5120, 5), ("fp32", 128, 4096, 2)],
)
def test_mxfp8_split_k_matches_direct(
    monkeypatch,
    contract: str,
    n: int,
    k: int,
    split_k: int,
) -> None:
    torch.manual_seed(2)
    m = 1
    a = _random_fp8((m, k))
    b = _random_fp8((n, k))
    if contract == "ue8m0":
        implementation = gluon_mm_mxfp8_ue8m0_gfx1250
        block_size = [1, 32]
        a_scales = torch.randint(
            124, 130, (m, k // 32), device="cuda", dtype=torch.uint8
        )
        b_scales = torch.randint(
            124, 130, (n, k // 32), device="cuda", dtype=torch.uint8
        )
    else:
        implementation = gluon_mm_fp8_blockscale_gfx1250
        block_size = [128, 128]
        a_scales = torch.rand((m, k // 128), device="cuda") + 0.5
        b_scales = torch.rand((n // 128, k // 128), device="cuda") + 0.5

    monkeypatch.setattr(
        mxfp8_mm,
        "_select_split_k",
        lambda selected_contract, selected_m, selected_n, selected_k: 1,
    )
    expected = implementation(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=block_size,
    )
    monkeypatch.setattr(
        mxfp8_mm,
        "_select_split_k",
        lambda selected_contract, selected_m, selected_n, selected_k: split_k,
    )
    backing = torch.empty((m, n + 17), device="cuda", dtype=torch.bfloat16)
    out = backing[:, :n]
    actual = implementation(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=block_size,
        out=out,
    )

    assert actual is out
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize(
    "contract,m,n,k,expected",
    [
        ("fp32", 1, 1536, 4096, 4),
        ("fp32", 15, 1536, 4096, 4),
        ("fp32", 16, 1536, 4096, 1),
        ("fp32", 8, 2048, 7168, 7),
        ("fp32", 16, 2048, 7168, 1),
        ("ue8m0", 1, 576, 5120, 5),
        ("ue8m0", 16, 1792, 5120, 5),
        ("ue8m0", 1, 4096, 1280, 1),
    ],
)
def test_mxfp8_split_k_selection(
    contract: str,
    m: int,
    n: int,
    k: int,
    expected: int,
) -> None:
    assert mxfp8_mm._select_split_k(contract, m, n, k) == expected


@pytest.mark.parametrize(
    "contract,m,n,k,split_k,expected",
    [
        ("fp32", 1, 2048, 7168, 7, mxfp8_mm._TDM_FUSION_PAIR_2W),
        ("fp32", 15, 2048, 7168, 7, mxfp8_mm._TDM_FUSION_PAIR_2W),
        ("fp32", 1, 2048, 7168, 1, mxfp8_mm._TDM_FUSION_NONE),
        ("fp32", 16, 2048, 7168, 1, mxfp8_mm._TDM_FUSION_NONE),
        ("fp32", 1, 1536, 4096, 4, mxfp8_mm._TDM_FUSION_NONE),
        ("ue8m0", 1, 1792, 5120, 5, mxfp8_mm._TDM_FUSION_NONE),
    ],
)
def test_mxfp8_tdm_fusion_selection(
    contract: str,
    m: int,
    n: int,
    k: int,
    split_k: int,
    expected: int,
) -> None:
    assert mxfp8_mm._select_tdm_fusion(contract, m, n, k, split_k) == expected


def test_fp8_blockscale_fused_tdm_matches_single_wave(monkeypatch) -> None:
    torch.manual_seed(3)
    m, n, k = 1, 2048, 7168
    a = _random_fp8((m, k))
    b = _random_fp8((n, k))
    a_scales = torch.rand((m, k // 128), device="cuda") + 0.5
    b_scales = torch.rand((n // 128, k // 128), device="cuda") + 0.5

    compiled_calls = []
    kernel = mxfp8_mm._fp8_blockscale_gemv_kernel
    run = kernel.run

    def record_compiled(*args, **kwargs):
        compiled = run(*args, **kwargs)
        compiled_calls.append(compiled)
        return compiled

    monkeypatch.setattr(kernel, "run", record_compiled)
    monkeypatch.setattr(
        mxfp8_mm,
        "_select_tdm_fusion",
        lambda contract, selected_m, selected_n, selected_k, split_k: mxfp8_mm._TDM_FUSION_NONE,
    )
    expected = gluon_mm_fp8_blockscale_gfx1250(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=[128, 128],
    )
    single_wave = compiled_calls[-1]

    monkeypatch.setattr(
        mxfp8_mm,
        "_select_tdm_fusion",
        lambda contract, selected_m, selected_n, selected_k, split_k: mxfp8_mm._TDM_FUSION_PAIR_2W,
    )
    actual = gluon_mm_fp8_blockscale_gfx1250(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=[128, 128],
    )
    fused = compiled_calls[-1]

    torch.testing.assert_close(
        actual.float(), expected.float(), atol=_BF16_ATOL, rtol=_BF16_RTOL
    )

    def count_tdm_instructions(compiled) -> int:
        return sum(
            "tensor_load_to_lds" in line and not line.lstrip().startswith((";", "."))
            for line in compiled.asm["amdgcn"].splitlines()
        )

    assert "bn16" in single_wave.name
    assert "tdmf0" in single_wave.name
    assert "bn32" in fused.name
    assert "tdmf1" in fused.name
    assert count_tdm_instructions(fused) < count_tdm_instructions(single_wave)


@pytest.mark.parametrize(
    "contract,n,k,expected_wmma",
    [
        ("ue8m0", 16, 5120, "v_wmma_scale_f32_16x16x128_f8f6f4"),
        ("fp32", 128, 7168, "v_wmma_f32_16x16x128_fp8_fp8"),
    ],
)
def test_gfx1250_gemv_uses_native_pipeline(
    monkeypatch,
    contract: str,
    n: int,
    k: int,
    expected_wmma: str,
) -> None:
    m = 1
    a = _random_fp8((m, k))
    b = _random_fp8((n, k))
    if contract == "ue8m0":
        kernel = mxfp8_mm._mxfp8_ue8m0_gemv_kernel
        implementation = gluon_mm_mxfp8_ue8m0_gfx1250
        block_size = [1, 32]
        a_scales = torch.full((m, k // 32), 127, device="cuda", dtype=torch.uint8)
        b_scales = torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8)
    else:
        kernel = mxfp8_mm._fp8_blockscale_gemv_kernel
        implementation = gluon_mm_fp8_blockscale_gfx1250
        block_size = [128, 128]
        a_scales = torch.ones((m, k // 128), device="cuda")
        b_scales = torch.ones((n // 128, k // 128), device="cuda")

    compiled_calls = []
    run = kernel.run

    def record_compiled(*args, **kwargs):
        compiled = run(*args, **kwargs)
        compiled_calls.append(compiled)
        return compiled

    monkeypatch.setattr(kernel, "run", record_compiled)
    implementation(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=block_size,
    )

    assert len(compiled_calls) == 1
    compiled = compiled_calls[0]
    assert "tensor_load_to_lds" in compiled.asm["amdgcn"]
    assert expected_wmma in compiled.asm["amdgcn"]
    assert "buffer_store_b128" in compiled.asm["amdgcn"]
    assert compiled.asm["ttgir"].count("scf.for") == 1


def test_mxfp8_ue8m0_gemv_preserves_strided_out() -> None:
    torch.manual_seed(2)
    m, n, k = 4, 128, 256
    a = _random_fp8((m, k))
    b = _random_fp8((n, k))
    a_scales = torch.full((m, k // 32), 127, device="cuda", dtype=torch.uint8)
    b_scales = torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8)
    backing = torch.empty((m, n + 17), device="cuda", dtype=torch.bfloat16)
    out = backing[:, :n]

    actual = gluon_mm_mxfp8_ue8m0_gfx1250(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=[1, 32],
        out=out,
    )

    assert actual is out
    torch.testing.assert_close(
        out.float(),
        a.float() @ b.float().T,
        atol=_BF16_ATOL,
        rtol=_BF16_RTOL,
    )


def test_mxfp8_launch_metadata_models_reread_traffic() -> None:
    m, n, k = 4, 128, 256
    metadata = mxfp8_mm._ue8m0_launch_metadata(
        (n // 16,),
        SimpleNamespace(name="mxfp8"),
        {"M": m, "N": n, "K": k, "SPLIT_K": 1},
    )

    assert metadata["name"] == "mxfp8"
    assert metadata["flops8"] == 2 * m * n * k
    assert metadata["bytes"] == (
        n * k
        + (n // 16) * m * k
        + n * (k // 32)
        + (n // 16) * m * (k // 32)
        + m * n * 2
    )
    assert (
        mxfp8_mm._mxfp8_ue8m0_gemv_kernel.launch_metadata
        is mxfp8_mm._ue8m0_launch_metadata
    )


def test_mxfp8_split_k_launch_metadata_models_partials() -> None:
    m, n, k, split_k = 4, 576, 5120, 5
    producer = mxfp8_mm._ue8m0_launch_metadata(
        (n // 16 * split_k,),
        SimpleNamespace(name="mxfp8_split"),
        {"M": m, "N": n, "K": k, "SPLIT_K": split_k},
    )
    reduction = mxfp8_mm._split_reduce_launch_metadata(
        (m * ((n + 127) // 128),),
        SimpleNamespace(name="mxfp8_reduce"),
        {"M": m, "N": n, "SPLIT_K": split_k},
    )

    assert producer["flops8"] == 2 * m * n * k
    assert producer["bytes"] == (
        n * k
        + (n // 16) * m * k
        + n * (k // 32)
        + (n // 16) * m * (k // 32)
        + split_k * m * n * 4
    )
    assert reduction == {
        "name": "mxfp8_reduce",
        "flops32": m * n * (split_k - 1),
        "bytes": split_k * m * n * 4 + m * n * 2,
    }


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"alpha": torch.tensor(1.0)}, "does not support alpha"),
        ({"block_size": [128, 128]}, "requires block_size"),
    ],
)
def test_mxfp8_ue8m0_gemv_rejects_unsupported_arguments(kwargs, match: str) -> None:
    m, n, k = 1, 128, 256
    a = _random_fp8((m, k))
    b = _random_fp8((n, k))
    a_scales = torch.full((m, k // 32), 127, device="cuda", dtype=torch.uint8)
    b_scales = torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8)
    call_kwargs = {"block_size": [1, 32], **kwargs}

    with pytest.raises(ValueError, match=match):
        gluon_mm_mxfp8_ue8m0_gfx1250(
            a,
            b,
            a_scales,
            b_scales,
            torch.bfloat16,
            **call_kwargs,
        )
