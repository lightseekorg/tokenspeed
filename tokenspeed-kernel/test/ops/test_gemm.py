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

import importlib
from unittest.mock import Mock

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.gemm import (
    linear_attnres_partials,
    linear_attnres_partials_available,
)
from tokenspeed_kernel.ops.gemm.triton import triton_bmm_fp8_blockscale


def test_mm_rejects_bad_out_layout() -> None:
    a = torch.empty((4, 8), dtype=torch.bfloat16)
    b = torch.empty((16, 8), dtype=torch.bfloat16)
    out = torch.empty((16, 4), dtype=torch.bfloat16).transpose(0, 1)

    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        tokenspeed_kernel.mm(a, b, out=out)


def test_mm_reference_rejects_out_dtype_mismatch() -> None:
    a = torch.empty((4, 8), dtype=torch.float32)
    b = torch.empty((16, 8), dtype=torch.float32)
    out = torch.empty((4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="torch_mm out= requires out_dtype"):
        tokenspeed_kernel.mm(a, b, out=out, override="torch_mm")


def test_bmm_rejects_batch_mismatch() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((3, 16, 8), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="batch mismatch"):
        tokenspeed_kernel.bmm(a, b)


def test_bmm_rejects_rank2_weights() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((16, 8), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"B with shape \[B, N, K\]"):
        tokenspeed_kernel.bmm(a, b)


def test_bmm_rejects_bad_out_layout() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((2, 16, 8), dtype=torch.bfloat16)
    out = torch.empty((2, 16, 4), dtype=torch.bfloat16).transpose(1, 2)

    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        tokenspeed_kernel.bmm(a, b, out=out)


def test_bmm_reference_rejects_out_dtype_mismatch() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.float32)
    b = torch.empty((2, 16, 8), dtype=torch.float32)
    out = torch.empty((2, 4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="torch_bmm out= requires out_dtype"):
        tokenspeed_kernel.bmm(a, b, out=out, override="torch_bmm")


@pytest.mark.parametrize("ue8m0_scales", [False, True], ids=["float", "ue8m0"])
def test_triton_bmm_fp8_blockscale_partial_tiles_and_out(
    ue8m0_scales: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Triton FP8 BMM requires a GPU")

    torch.manual_seed(23)
    batch, m, n, k = 2, 5, 70, 96
    block_n, block_k = 32, 32
    a = torch.randn((batch, m, k), device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn((batch, n, k), device="cuda").to(torch.float8_e4m3fn)
    if ue8m0_scales:
        a_scales = torch.randint(
            124,
            130,
            (batch, m, k // block_k),
            device="cuda",
            dtype=torch.uint8,
        )
        b_scales = torch.randint(
            124,
            130,
            (batch, (n + block_n - 1) // block_n, k // block_k),
            device="cuda",
            dtype=torch.uint8,
        )
        a_scale_values = torch.exp2(a_scales.float() - 127.0)
        b_scale_values = torch.exp2(b_scales.float() - 127.0)
    else:
        a_scales = torch.rand(
            (batch, m, k // block_k), device="cuda", dtype=torch.float32
        )
        b_scales = torch.rand(
            (batch, (n + block_n - 1) // block_n, k // block_k),
            device="cuda",
            dtype=torch.float32,
        )
        a_scale_values = a_scales
        b_scale_values = b_scales

    backing = torch.empty((m, batch, n + 3), device="cuda", dtype=torch.float32)
    out = backing[..., :n].transpose(0, 1)
    actual = triton_bmm_fp8_blockscale(
        a,
        b,
        a_scales,
        b_scales,
        torch.float32,
        block_size=[block_n, block_k],
        out=out,
    )

    a_dequant = a.float() * a_scale_values.repeat_interleave(block_k, dim=-1)
    b_dequant = (
        b.float()
        * b_scale_values.repeat_interleave(block_n, dim=1).repeat_interleave(
            block_k, dim=2
        )[:, :n, :k]
    )
    expected = torch.bmm(a_dequant, b_dequant.transpose(1, 2))

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


def test_bmm_writes_head_major_strided_output(device: str) -> None:
    heads, tokens, k, n = 3, 1, 8, 16
    a = torch.randn(heads, tokens, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    backing = torch.empty(tokens, heads, n + 4, device=device, dtype=torch.bfloat16)
    out = backing[..., :n].transpose(0, 1)

    returned = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        out=out,
        override="torch_bmm",
    )

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=0, rtol=0)


def test_gluon_bmm_writes_head_major_strided_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    heads, tokens, k, n = 12, 1, 128, 512
    a_backing = torch.randn(tokens, heads, k, device=device, dtype=torch.bfloat16)
    a = a_backing.transpose(0, 1)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    backing = torch.empty(tokens, heads, n + 64, device=device, dtype=torch.bfloat16)
    out = backing[..., :n].transpose(0, 1)

    returned = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        out=out,
        override="gluon_bmm_a16w16_gfx950",
    )

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=0.25, rtol=0.01)


def test_gluon_bmm_allocates_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    a = torch.randn(12, 1, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(12, 128, 512, device=device, dtype=torch.bfloat16)

    output = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        override="gluon_bmm_a16w16_gfx950",
    )

    torch.testing.assert_close(output, torch.bmm(a, weight), atol=0.25, rtol=0.01)


def test_gluon_bmm_falls_back_for_fp32_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    a = torch.randn(12, 1, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(12, 128, 512, device=device, dtype=torch.bfloat16)

    output = tokenspeed_kernel.bmm(a, weight.transpose(1, 2), out_dtype=torch.float32)

    assert output.dtype == torch.float32


def test_decode_gemv_writes_preallocated_output() -> None:
    from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

    x = torch.randn(2, 8)
    weight = torch.randn(4, 8)
    out = torch.empty(2, 4)

    returned = decode_gemv(x, weight, out=out)

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, x @ weight.t())


def test_linear_attnres_partials_portable_composition() -> None:
    torch.manual_seed(13)
    hidden = torch.randn(2, 6, dtype=torch.bfloat16)
    weight = torch.randn(9, 6, dtype=torch.bfloat16)
    blocks = torch.randn(4, 2, 6, dtype=torch.bfloat16)
    scores = (
        torch.randn(6, dtype=torch.bfloat16),
        torch.randn(6, dtype=torch.bfloat16),
    )
    scratch = tuple(
        (
            torch.empty(2, dtype=torch.float32),
            torch.empty(2, dtype=torch.float32),
            torch.empty(2, 6, dtype=torch.float32),
        )
        for _ in range(2)
    )

    actual = linear_attnres_partials(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )

    torch.testing.assert_close(actual, torch.nn.functional.linear(hidden, weight))
    values = blocks.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-5)
    for score, outputs in zip(scores, scratch, strict=True):
        logits = torch.einsum("bth,h->bt", values, score.float()) * inverse_rms
        maxima = logits.max(dim=0).values
        unnormalized = torch.exp(logits - maxima)
        torch.testing.assert_close(outputs[0], maxima)
        torch.testing.assert_close(outputs[1], unnormalized.sum(dim=0))
        torch.testing.assert_close(
            outputs[2],
            torch.einsum("bt,bth->th", unnormalized, values),
        )


def test_linear_attnres_partials_decode_fallback_uses_gemv(monkeypatch) -> None:
    from tokenspeed_kernel.ops.gemm import triton_gemv

    hidden = torch.randn(1, 6, dtype=torch.bfloat16)
    weight = torch.randn(9, 6, dtype=torch.bfloat16)
    blocks = torch.randn(2, 1, 6, dtype=torch.bfloat16)
    scores = tuple(torch.randn(6, dtype=torch.bfloat16) for _ in range(2))
    scratch = tuple(
        (
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, 6, dtype=torch.float32),
        )
        for _ in range(2)
    )
    expected = torch.empty(1, 9, dtype=torch.bfloat16)
    gemv = Mock(return_value=expected)
    monkeypatch.setattr(triton_gemv, "decode_gemv", gemv)

    assert not linear_attnres_partials_available(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )

    actual = linear_attnres_partials(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )

    assert actual is expected
    gemv.assert_called_once_with(hidden, weight)


def test_linear_attnres_partials_cpu_skips_device_kernel_selection(
    monkeypatch,
) -> None:
    module = importlib.import_module(
        "tokenspeed_kernel.ops.gemm.linear_attnres_partials"
    )
    selector = Mock()
    monkeypatch.setattr(module, "select_kernel", selector)
    hidden = torch.randn(1, 6, dtype=torch.bfloat16)
    weight = torch.randn(9, 6, dtype=torch.bfloat16)
    blocks = torch.randn(2, 1, 6, dtype=torch.bfloat16)
    scores = tuple(torch.randn(6, dtype=torch.bfloat16) for _ in range(2))
    scratch = tuple(
        (
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, 6, dtype=torch.float32),
        )
        for _ in range(2)
    )

    assert not linear_attnres_partials_available(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )
    selector.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""),
    reason="gfx950 is required",
)
@pytest.mark.parametrize("tokens", [1, 2, 4])
@pytest.mark.parametrize("output_size", [3648, 6288])
def test_linear_attnres_partials_gfx950_matches_composition(
    tokens: int, output_size: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    hidden = (torch.randn(tokens, 7168, device="cuda", generator=generator) * 0.1).to(
        torch.bfloat16
    )
    weight = (
        torch.randn(output_size, 7168, device="cuda", generator=generator) * 0.01
    ).to(torch.bfloat16)
    blocks = (
        torch.randn(4, tokens, 7168, device="cuda", generator=generator) * 0.1
    ).to(torch.bfloat16)
    scores = tuple(
        (torch.randn(7168, device="cuda", generator=generator) * 0.02).to(
            torch.bfloat16
        )
        for _ in range(2)
    )
    scratch = tuple(
        (
            torch.empty(tokens, device="cuda", dtype=torch.float32),
            torch.empty(tokens, device="cuda", dtype=torch.float32),
            torch.empty(tokens, 7168, device="cuda", dtype=torch.float32),
        )
        for _ in range(2)
    )

    assert linear_attnres_partials_available(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-6,
    )

    actual = linear_attnres_partials(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-6,
        override="gluon_linear_attnres_partials_gfx950",
    )

    torch.testing.assert_close(
        actual,
        torch.nn.functional.linear(hidden, weight),
        atol=2e-2,
        rtol=2e-2,
    )
    values = blocks.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-6)
    for score, outputs in zip(scores, scratch, strict=True):
        logits = torch.einsum("bth,h->bt", values, score.float()) * inverse_rms
        maxima = logits.max(dim=0).values
        unnormalized = torch.exp(logits - maxima)
        torch.testing.assert_close(outputs[0], maxima, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(
            outputs[1], unnormalized.sum(dim=0), atol=2e-4, rtol=2e-4
        )
        torch.testing.assert_close(
            outputs[2],
            torch.einsum("bt,bth->th", unnormalized, values),
            atol=2e-4,
            rtol=2e-4,
        )

    with pytest.raises(ValueError, match="output size must be divisible by 16"):
        linear_attnres_partials(
            hidden,
            weight[:17],
            blocks,
            *scores,
            *scratch,
            eps=1e-6,
            override="gluon_linear_attnres_partials_gfx950",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is required")
def test_linear_attnres_partials_cuda_portable_strided_inputs() -> None:
    generator = torch.Generator(device="cuda").manual_seed(37)
    hidden = torch.randn(
        1, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        32, 65, device="cuda", dtype=torch.bfloat16, generator=generator
    )[:, :64]
    blocks = torch.randn(
        3, 1, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )[..., ::2]
    scores = tuple(
        torch.randn(128, device="cuda", dtype=torch.bfloat16, generator=generator)[::2]
        for _ in range(2)
    )
    scratch = tuple(
        (
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(1, 64, device="cuda", dtype=torch.float32),
        )
        for _ in range(2)
    )

    actual = linear_attnres_partials(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )

    torch.testing.assert_close(actual, torch.nn.functional.linear(hidden, weight))
    values = blocks.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-5)
    for score, outputs in zip(scores, scratch, strict=True):
        logits = torch.einsum("bth,h->bt", values, score.float()) * inverse_rms
        maxima = logits.max(dim=0).values
        unnormalized = torch.exp(logits - maxima)
        torch.testing.assert_close(outputs[0], maxima)
        torch.testing.assert_close(outputs[1], unnormalized.sum(dim=0))
        torch.testing.assert_close(
            outputs[2],
            torch.einsum("bt,bth->th", unnormalized, values),
        )
