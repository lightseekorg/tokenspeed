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

"""GFX1250 hc=4 mHC mixing coefficients and their WMMA prenorm projection."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.residual import torch_mhc_mixes
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel.ops.quantization import quantize_fp8  # noqa: E402
from tokenspeed_kernel.ops.residual import mhc_apply_pre_quantize_fp8  # noqa: E402
from tokenspeed_kernel.ops.residual.gluon import (  # noqa: E402
    gluon_mhc_mixes_gfx1250,
    gluon_mhc_pre_gfx1250,
)
from tokenspeed_kernel.ops.residual.triton import (  # noqa: E402
    compute_mhc_num_splits,
    triton_mhc_apply_pre,
)
from tokenspeed_kernel.selection import select_kernel  # noqa: E402
from tokenspeed_kernel.signature import (  # noqa: E402
    dense_tensor_format,
    format_signature,
)
from tokenspeed_kernel_amd.ops.gfx1250.mhc import (  # noqa: E402
    gluon_mhc_mixes_project_gfx1250,
    gluon_mhc_mixes_reduce_gfx1250,
    launch_gluon_mhc_mixes_project_gfx1250,
    use_gluon_mhc_mixes_project_gfx1250,
)

_RMS_EPS = 1e-6
_HC_EPS = 1e-6
_SINKHORN_ITERS = 20


def _operands(tokens: int, hidden: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(
        tokens, 4, hidden, device="cuda", generator=generator
    ).bfloat16()
    weight = torch.randn(24, 4 * hidden, device="cuda", generator=generator) * 0.01
    scale = torch.tensor([0.7, 1.1, 0.5], device="cuda")
    base = torch.randn(24, device="cuda", generator=generator)
    return residual, weight, scale, base


def _reference(residual, weight, scale, base):
    return torch_mhc_mixes(
        residual,
        weight.double(),
        scale.double(),
        base.double(),
        _RMS_EPS,
        _HC_EPS,
        _SINKHORN_ITERS,
    )


def _run(residual, weight, scale, base):
    return gluon_mhc_mixes_gfx1250(
        residual, weight, scale, base, _RMS_EPS, _HC_EPS, _SINKHORN_ITERS
    )


def _assert_matches_reference(actual, residual, weight, scale, base) -> None:
    for got, want in zip(actual, _reference(residual, weight, scale, base)):
        assert got.dtype == torch.float32
        torch.testing.assert_close(got.double(), want, atol=1e-4, rtol=1e-4)


def _splits(tokens: int, hidden: int) -> int:
    return compute_mhc_num_splits(
        torch.device("cuda"), 64, 4 * hidden, max(1, -(-tokens // 64))
    )


# On 256 CUs these token counts reach 80, 64, 32 and fewer K splits, so both
# 64-wide K tiles (64 or more splits) and 128-wide ones run; 320 is the
# cheapest 128-wide case. 4 * 1000 does not tile into 64-wide K tiles, so that
# width keeps the Triton GEMM.
@pytest.mark.parametrize("tokens", [1, 96, 192, 320, 2048, 8192])
@pytest.mark.parametrize("hidden", [5120, 1000])
def test_mhc_mixes_matches_reference(tokens: int, hidden: int) -> None:
    operands = _operands(tokens, hidden, seed=tokens + hidden)
    splits = _splits(tokens, hidden)
    assert use_gluon_mhc_mixes_project_gfx1250(4 * hidden, splits) == (hidden != 1000)
    _assert_matches_reference(_run(*operands), *operands)


def test_projection_matches_fp64() -> None:
    tokens = 2048
    residual, weight, _, _ = _operands(tokens, 5120, seed=tokens)
    x = residual.view(tokens, -1)
    splits = _splits(tokens, 5120)
    assert use_gluon_mhc_mixes_project_gfx1250(x.shape[1], splits)
    out_mul = torch.empty(splits, tokens, 24, device="cuda")
    out_sqrsum = torch.empty(splits, tokens, device="cuda")
    launch_gluon_mhc_mixes_project_gfx1250(x, weight, out_mul, out_sqrsum, splits)
    parts = x.double().view(tokens, splits, -1)
    torch.testing.assert_close(
        out_mul.double(),
        torch.einsum("tsk,nsk->stn", parts, weight.double().view(24, splits, -1)),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        out_sqrsum.double(), parts.square().sum(-1).T, atol=0, rtol=1e-5
    )


def test_mhc_mixes_selects_gfx1250_kernel() -> None:
    kernel = select_kernel(
        "residual",
        "mhc_mixes",
        format_signature(residual=dense_tensor_format(torch.bfloat16)),
    )
    assert kernel.name == "gluon_mhc_mixes_gfx1250"


@pytest.mark.parametrize("tokens", [1, 96, 2048])
def test_mhc_pre_matches_reference(tokens: int) -> None:
    residual, weight, scale, base = _operands(tokens, 4096, seed=tokens)
    layer_input, post, comb = gluon_mhc_pre_gfx1250(
        residual, weight, scale, base, _RMS_EPS, _HC_EPS, _SINKHORN_ITERS, None, None
    )
    pre, post_ref, comb_ref = _reference(residual, weight, scale, base)
    torch.testing.assert_close(post.squeeze(-1).double(), post_ref, atol=1e-4, rtol=0)
    torch.testing.assert_close(comb.double(), comb_ref, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        layer_input.double(),
        (pre[:, :, None] * residual.double()).sum(1),
        atol=1e-2,
        rtol=1e-2,
    )


def test_mhc_pre_selects_gfx1250_kernel() -> None:
    kernel = select_kernel(
        "residual",
        "mhc_pre",
        format_signature(
            residual=dense_tensor_format(torch.bfloat16),
            fn=dense_tensor_format(torch.float32),
            hc_scale=dense_tensor_format(torch.float32),
            hc_base=dense_tensor_format(torch.float32),
        ),
        traits={"num_tokens": 8, "hc_mult": 4, "hidden_size": 4096},
    )
    assert kernel.name == "gluon_mhc_pre_gfx1250"


@pytest.mark.parametrize("tokens", [1, 5, 80, 2048])
def test_mhc_apply_pre_quantize_fp8_matches_unfused(tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    residual = torch.randn(
        tokens, 4, 5120, device="cuda", generator=generator
    ).bfloat16()
    pre = torch.rand(tokens, 4, device="cuda", generator=generator)
    weight = torch.rand(5120, device="cuda", generator=generator).bfloat16()
    out, codes, scales = mhc_apply_pre_quantize_fp8(residual, pre, weight, _RMS_EPS)
    expected = triton_mhc_apply_pre(
        residual, pre, norm_weight=weight, norm_eps=_RMS_EPS
    )
    # The RMS reduction order differs, which moves a rare value by one ulp.
    torch.testing.assert_close(out, expected, atol=0, rtol=2**-7)
    assert (out != expected).double().mean().item() < 1e-3
    expected_codes, expected_scales = quantize_fp8(
        out,
        granularity="token_group",
        group_size=32,
        scale_encoding="ue8m0",
        enable_pdl=False,
        override=None,
        solution=None,
    )
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


def test_mhc_apply_pre_fp8_selects_gfx1250_kernel() -> None:
    kernel = select_kernel(
        "residual",
        "mhc_apply_pre_fp8_group32",
        format_signature(
            residual=dense_tensor_format(torch.bfloat16),
            pre=dense_tensor_format(torch.float32),
        ),
        traits={"buffer_offsets_fit_int32": True, "hc_mult": 4},
    )
    assert kernel.name == "gluon_mhc_apply_pre_fp8_group32_gfx1250"


def test_mhc_mixes_cuda_graph_replay() -> None:
    tokens = 32
    residual, weight, scale, base = _operands(tokens, 5120, seed=5)
    _run(residual, weight, scale, base)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(residual, weight, scale, base)
    residual.copy_(_operands(tokens, 5120, seed=6)[0])
    graph.replay()
    _assert_matches_reference(out, residual, weight, scale, base)


def test_projection_varying_tokens_reuse_compiled_kernels() -> None:
    _, weight, scale, base = _operands(1, 5120, seed=0)

    def run(tokens: int) -> None:
        _run(_operands(tokens, 5120, seed=tokens)[0], weight, scale, base)

    # Warm both tile widths (64-wide K tiles at 64 or more splits, 128-wide
    # below) and the three reduction layouts (splits in registers, also over
    # lanes beyond 256 tokens, and also over waves at or below).
    for tokens in (1, 848, 8192):
        run(tokens)
    with assert_no_triton_compile(
        gluon_mhc_mixes_project_gfx1250, gluon_mhc_mixes_reduce_gfx1250
    ):
        for tokens in (6, 192, 600, 2048, 4097):
            run(tokens)
