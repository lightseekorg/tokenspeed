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
"""Split-K FP8 GEMM with row and channel scales, and its operand
conversions, on gfx950."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("gfx950 required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import mm, prepare_fp8_channel_weight
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.thirdparty.sglang.fp8_small_m_gemm import (
    _fp8_rowmajor_partial,
    _quantize_rows_kernel,
    _sum_scaled,
)

SPLIT_K = "triton_mm_fp8_channel_splitk"
# (N, K) and the row counts its tuning table serves.
TUNED = {
    (4608, 8192): 256,
    (2304, 8192): 256,
    (1024, 8192): 256,
}


def quantize_rows(x):
    """The amax-division token quantizer, as the FP8 projections call it."""
    return quantize_fp8(
        x,
        granularity="token",
        scale_encoding="float32",
        enable_pdl=False,
        required_features=frozenset({"token_fp8_amax_div"}),
    )


def reference_rows(x):
    """CPU reference (correctly rounded FP32 division): scale = max(row amax,
    1e-20) / 448, codes = x / scale; returned on x's device."""
    values = x.float().cpu()
    scale = values.abs().amax(1).clamp_min(1e-20) / 448.0
    codes = (values / scale[:, None]).to(torch.float8_e4m3fn)
    return codes.to(x.device), scale.to(x.device)


def project(rows, scales, weight, weight_scale):
    return mm(
        rows,
        weight.t(),
        A_scales=scales,
        B_scales=weight_scale,
        out_dtype=torch.bfloat16,
        quant="fp8",
        override=SPLIT_K,
    )


def bf16_half_ulp(x):
    """Half a BF16 unit in the last place of each element of ``x`` (BF16)."""
    _, exponent = torch.frexp(x.float())
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), exponent - 9)


@pytest.mark.parametrize("m", [1, 17, 64, 256])
def test_row_quantizer_matches_the_torch_reference(m):
    """Codes and scales equal the CPU reference bit for bit, including a zero
    row (the 1e-20 floor, all-zero codes) and the row maximum (448)."""
    torch.manual_seed(m)
    x = (torch.randn(m, 8192, device="cuda") * 3).bfloat16()
    x[0].zero_()
    if m > 1:
        x[1, 7] = -1.0e4
    codes, scales = quantize_rows(x)
    expected_codes, expected_scales = reference_rows(x)
    assert torch.equal(scales.reshape(-1), expected_scales)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert not codes[0].float().any()
    if m > 1:
        assert codes[1, 7].float() == -448.0


def test_weight_conversion_matches_the_torch_reference():
    """Per-channel scale = max(channel amax, 1e-12) x FP32 1 / 448 and codes =
    channel / scale, bit for bit against a CPU reference; a zero channel keeps
    zero codes."""
    weight = (torch.randn(64, 8192, device="cuda") * 0.02).bfloat16()
    weight[3].zero_()
    codes, scales = prepare_fp8_channel_weight(weight)
    values = weight.float().cpu()
    one_448th = torch.tensor(1 / 448.0, dtype=torch.float32)
    expected_scales = values.abs().amax(1).clamp_min(1e-12) * one_448th
    expected = (values / expected_scales[:, None]).to(torch.float8_e4m3fn)
    assert torch.equal(scales.cpu(), expected_scales)
    assert torch.equal(codes.cpu().view(torch.uint8), expected.view(torch.uint8))
    assert not codes[3].float().any()


@pytest.mark.parametrize("n,k", list(TUNED))
@pytest.mark.parametrize("m", [1, 16, 17, 64, 96, 128, 129, 192, 256])
def test_fp8_channel_splitk_is_within_fp32_accumulation_error(m, n, k):
    """mm's split-K FP8 product is within the error of its K + 1 FP32
    roundings (K - 1 additions, two scale multiplies) and one BF16 rounding
    of the exact product: (K + 1) x 2^-24 x sum |x_i w_i| plus half a BF16
    ulp. Nearly every element (all but a few cancelling sums) is the
    correctly rounded exact product or a BF16 neighbour of it."""
    if m > TUNED[(n, k)]:
        pytest.skip("past this shape's tuned rows")
    torch.manual_seed(m)
    weight_codes, weight_scale = prepare_fp8_channel_weight(
        (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    )
    rows, scales = reference_rows(torch.randn(m, k, device="cuda").bfloat16())
    a = rows.double() * scales.double()[:, None]
    b = weight_codes.double() * weight_scale.double()[:, None]
    exact = a @ b.T
    magnitude = a.abs() @ b.abs().T
    actual = project(rows, scales, weight_codes, weight_scale)
    assert actual.dtype == torch.bfloat16 and actual.shape == (m, n)
    rounded = exact.to(torch.bfloat16)
    half_ulp = bf16_half_ulp(rounded).double()
    # (K + 1) u is the first-order bound; 1.01 covers its higher-order terms.
    bound = half_ulp + 1.01 * (k + 1) * 2.0**-24 * magnitude
    assert torch.all((actual.double() - exact).abs() <= bound)
    # Within one BF16 ulp of the rounded exact product (that value or a
    # neighbour) for at least 99.5 % of the elements.
    near = (actual.double() - rounded.double()).abs() <= 2 * half_ulp
    assert near.float().mean() >= 0.995


@pytest.mark.parametrize(
    "n,k,m", [(512, 8192, 17), (4096, 8192, 129), (4608, 8192, 257)]
)
def test_fp8_channel_splitk_refuses_untuned_problems(n, k, m):
    """Pinned by name, the split-K GEMM refuses a shape or row tile its table
    lacks instead of running another kernel."""
    weight, weight_scale = prepare_fp8_channel_weight(
        torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    )
    rows, scales = reference_rows(torch.zeros(m, k, device="cuda").bfloat16())
    from tokenspeed_kernel.selection import NoKernelFoundError

    with pytest.raises(NoKernelFoundError):
        project(rows, scales, weight, weight_scale)


def test_fp8_channel_splitk_row_count_does_not_recompile():
    """Row counts inside one row tile share the compiled GEMM, reduction and
    row quantizer."""
    n, k = 4608, 8192
    weight, weight_scale = prepare_fp8_channel_weight(
        (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    )
    source = torch.randn(256, k, device="cuda").bfloat16()
    for m in (17, 33, 65, 129):
        project(*quantize_rows(source[:m]), weight, weight_scale)
    with assert_no_triton_compile(
        _fp8_rowmajor_partial, _sum_scaled, _quantize_rows_kernel
    ):
        for m in range(17, 257):
            project(*quantize_rows(source[:m]), weight, weight_scale)


@pytest.mark.parametrize("m", [17, 64, 256])
def test_unpinned_fp8_mm_keeps_triton_mm_fp8_scaled(m):
    """Ranked selection does not move other FP8 callers onto the split-K
    GEMM, which is not batch-invariant: at a shape its table serves, an mm
    without override= launches the kernels triton_mm_fp8_scaled launches and
    gives its bytes."""
    weight, weight_scale = prepare_fp8_channel_weight(
        (torch.randn(4608, 8192, device="cuda") * 0.02).bfloat16()
    )
    rows, scales = reference_rows(torch.randn(m, 8192, device="cuda").bfloat16())
    kwargs = dict(
        A_scales=scales, B_scales=weight_scale, out_dtype=torch.bfloat16, quant="fp8"
    )

    def launched(**override):
        # Warm first so the profiled call launches only the GEMM's kernels.
        mm(rows, weight.t(), **override, **kwargs)
        torch.cuda.synchronize()
        activities = [torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as profile:
            result = mm(rows, weight.t(), **override, **kwargs)
            torch.cuda.synchronize()
        names = {e.name for e in profile.events() if e.device_type.name == "CUDA"}
        return result, names

    selected, selected_kernels = launched()
    expected, expected_kernels = launched(override="triton_mm_fp8_scaled")
    assert expected_kernels and selected_kernels == expected_kernels
    assert torch.equal(selected, expected)


def quantize_rows_1d(x):
    """``quantize_rows`` with ``[M]`` scales, as the split-K partials take them."""
    codes, scales = quantize_rows(x)
    return codes, scales.reshape(-1)


def _split_k_case(heads, m, seed):
    """The partials, row and column scales and BF16 product of one split-K
    projection with ``heads`` = (query, KV, gate) heads of 64."""
    from tokenspeed_kernel.ops.gemm import mm_split_k_partials

    hq, hkv, hg = heads
    n = (hq + 2 * hkv + hg) * 64
    torch.manual_seed(seed)
    weight, weight_scale = prepare_fp8_channel_weight(
        (torch.randn(n, 8192, device="cuda") * 0.02).bfloat16()
    )
    rows, scales = quantize_rows_1d(torch.randn(m, 8192, device="cuda").bfloat16())
    partials = mm_split_k_partials(
        rows, weight.t(), A_scales=scales, B_scales=weight_scale
    ).partials
    return partials, scales, weight_scale, project(rows, scales, weight, weight_scale)


# Muse Spark's Q/K/V/gate projection at TP4 and TP8.
_QKVG_HEADS = [(32, 4, 32), (16, 2, 16)]


@pytest.mark.parametrize("heads", _QKVG_HEADS)
@pytest.mark.parametrize("m", [17, 24, 32, 33, 48, 64, 65, 96, 128, 129, 200, 256])
@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("style", ["neox", "gptj", "none"])
@pytest.mark.parametrize("with_norm", [True, False])
def test_split_k_prologue_writes_the_bytes_of_the_bf16_projection(
    heads, m, cache_dtype, style, with_norm
):
    """gqa_prologue_split_k over the partials gives the query, gate and K/V
    cache bytes of the split-K GEMM's BF16 rows through gqa_prologue."""
    from tokenspeed_kernel.ops.attention.prologue import (
        HeadKVCache,
        HeadNorm,
        RopeStyle,
        Rotary,
        SplitKQKVG,
        gqa_prologue,
        gqa_prologue_split_k,
    )

    hq, hkv, hg = heads
    partials, scales, weight_scale, projected = _split_k_case(heads, m, m)
    q, k, v, gate = projected.split((hq * 64, hkv * 64, hkv * 64, hg * 64), dim=-1)
    norm = (
        HeadNorm(
            torch.randn(64, device="cuda").bfloat16(),
            torch.randn(64, device="cuda").bfloat16(),
            0.0,
            1e-5,
            round_before_rope=False,
        )
        if with_norm
        else None
    )
    rotary = (
        None
        if style == "none"
        else Rotary(
            torch.randn(4096, 64, device="cuda"),
            torch.randint(0, 4096, (m,), device="cuda"),
            RopeStyle(style),
            None,
        )
    )
    slots = torch.randperm(1024, device="cuda")[:m].int()

    def cache():
        fill = torch.full((1024, hkv, 64), 0.5, device="cuda").to(cache_dtype)
        return HeadKVCache(fill, fill.clone(), None, slots)

    expected_cache, actual_cache = cache(), cache()
    expected = gqa_prologue(
        q,
        k,
        v,
        norm=norm,
        rotary=rotary,
        cache=expected_cache,
        query_dtype=torch.bfloat16,
        return_kv=False,
        solution="triton",
        override=None,
    )
    actual = gqa_prologue_split_k(
        SplitKQKVG(partials, scales, weight_scale),
        num_q_heads=hq,
        norm=norm,
        rotary=rotary,
        cache=actual_cache,
        query_dtype=torch.bfloat16,
        solution=None,
        override=None,
    )
    as_bytes = lambda t: t.contiguous().view(torch.uint8)  # noqa: E731
    assert torch.equal(as_bytes(actual.q), as_bytes(expected.q))
    assert torch.equal(as_bytes(actual.gate), as_bytes(gate))
    assert torch.equal(as_bytes(actual_cache.k_cache), as_bytes(expected_cache.k_cache))
    assert torch.equal(as_bytes(actual_cache.v_cache), as_bytes(expected_cache.v_cache))


def test_split_k_prologue_row_count_does_not_recompile():
    """Decode row counts (3-16 running requests of 8 verify rows) share the
    split-K prologue's compiled kernel."""
    from tokenspeed_kernel.ops.attention.prologue import (
        HeadKVCache,
        RopeStyle,
        Rotary,
        SplitKQKVG,
        gqa_prologue_split_k,
    )
    from tokenspeed_kernel.ops.attention.prologue.triton import (
        _gqa_prologue_split_k_kernel,
    )

    def run(m):
        partials, scales, weight_scale, _ = _split_k_case((32, 4, 32), m, 0)
        fill = torch.zeros((1024, 4, 64), device="cuda").to(torch.float8_e4m3fn)
        gqa_prologue_split_k(
            SplitKQKVG(partials, scales, weight_scale),
            num_q_heads=32,
            norm=None,
            rotary=Rotary(
                torch.randn(4096, 64, device="cuda"),
                torch.arange(m, device="cuda"),
                RopeStyle.NEOX,
                None,
            ),
            cache=HeadKVCache(
                fill, fill.clone(), None, torch.arange(m, device="cuda").int()
            ),
            query_dtype=torch.bfloat16,
            solution=None,
            override=None,
        )

    # One warm launch per split count and token block (32 / 64 / 128 rows).
    for m in (24, 72, 128):
        run(m)
    with assert_no_triton_compile(_gqa_prologue_split_k_kernel):
        for m in (24, 32, 40, 48, 56, 64, 72, 80, 96, 104, 112, 120, 128):
            run(m)


def test_split_k_prologue_declines_split_counts_it_does_not_reduce():
    """No prologue spells out _sum_scaled's order for 8 slices; asking for it
    raises instead of reducing in another order."""
    from tokenspeed_kernel.ops.attention.prologue import (
        HeadKVCache,
        SplitKQKVG,
        gqa_prologue_split_k,
    )
    from tokenspeed_kernel.selection import NoKernelFoundError

    partials = torch.zeros((8, 32, 72 * 64), device="cuda")
    fill = torch.zeros((1024, 4, 64), device="cuda").bfloat16()
    with pytest.raises(NoKernelFoundError):
        gqa_prologue_split_k(
            SplitKQKVG(
                partials,
                torch.ones(32, device="cuda"),
                torch.ones(72 * 64, device="cuda"),
            ),
            num_q_heads=32,
            norm=None,
            rotary=None,
            cache=HeadKVCache(
                fill, fill.clone(), None, torch.arange(32, device="cuda").int()
            ),
            query_dtype=torch.bfloat16,
            solution=None,
            override=None,
        )


@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("m", [17, 32, 64, 96, 128])
def test_split_k_partials_finish_to_the_gemm_bits(m, n):
    """The unreduced partials, summed in _sum_scaled's order, scaled and
    rounded once, give the split-K GEMM's BF16 output bit for bit. [1024,
    8192] (TP4 pre-expert columns) splits K eight ways, [4096, 8192] four."""
    from tokenspeed_kernel.ops.gemm import mm_split_k_partials

    torch.manual_seed(m)
    k = 8192
    weight, weight_scale = prepare_fp8_channel_weight(
        (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    )
    rows, scales = quantize_rows_1d(torch.randn(m, k, device="cuda").bfloat16())
    result = mm_split_k_partials(
        rows, weight.t(), A_scales=scales, B_scales=weight_scale
    )
    assert result is not None
    p = result.partials
    if n == 1024:
        assert p.shape == (8, m, n)
        total = ((p[0] + p[2]) + (p[4] + p[6])) + ((p[1] + p[3]) + (p[5] + p[7]))
    else:
        assert p.shape == (4, m, n)
        total = (p[0] + p[2]) + (p[1] + p[3])
    finished = (total * result.row_scales[:, None] * result.col_scales).bfloat16()
    assert torch.equal(finished, project(rows, scales, weight, weight_scale))


def test_split_k_partials_refuse_unserved_problems():
    """Problems off the tuned table, or past 256 rows, keep the full GEMM: no
    kernel serves their partials."""
    from tokenspeed_kernel.ops.gemm import mm_split_k_partials
    from tokenspeed_kernel.ops.gemm.triton_fp8_channel_splitk import (
        triton_mm_fp8_channel_splitk_partials,
    )

    for n, m in ((2048, 32), (1024, 257)):
        weight, weight_scale = prepare_fp8_channel_weight(
            (torch.randn(n, 8192, device="cuda") * 0.02).bfloat16()
        )
        rows, scales = quantize_rows_1d(torch.randn(m, 8192, device="cuda").bfloat16())
        assert (
            mm_split_k_partials(
                rows, weight.t(), A_scales=scales, B_scales=weight_scale
            )
            is None
        )
        with pytest.raises(ValueError):
            triton_mm_fp8_channel_splitk_partials(
                rows, weight.t(), scales, weight_scale
            )
