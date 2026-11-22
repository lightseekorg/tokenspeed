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
"""The skinny FP8 GEMM (gluon_mm_fp8_skinny_gfx950) and its per-row FP8 code."""

import pytest
import torch
from utils import assert_no_triton_compile, compiled_kernels, is_cdna4

if not is_cdna4():
    pytest.skip("gfx950 required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import mm, prepare_fp8_channel_weight  # noqa: E402
from tokenspeed_kernel.ops.quantization import quantize_fp8  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8.skinny import (  # noqa: E402
    gluon_mm_fp8_skinny_gfx950,
    gluon_mm_fp8_skinny_reduce_gfx950,
    launch_gluon_mm_fp8_skinny_gfx950,
    supports_gluon_mm_fp8_skinny_gfx950,
)

_K = 8192
# The two configured weight shapes ([N, 8192]) and the first row each takes.
_SHAPES = (4608, 1024)
_FIRST = {4608: 17, 1024: 17}
_NAME = "gluon_mm_fp8_skinny_gfx950"


def _rows_fp8(x):
    return quantize_fp8(
        x.contiguous(),
        granularity="token",
        scale_encoding="float32",
        enable_pdl=False,
        required_features=frozenset({"token_fp8_amax_div"}),
    )


def _skinny(codes, scales, weight, weight_scales):
    return mm(
        codes,
        weight.t(),
        A_scales=scales,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="fp8",
        override=_NAME,
    )


def _bits(x):
    return x.view(torch.int16)


def test_row_code_is_the_ieee_division_definition():
    """Scale fl32(amax / 448) and codes RNE(fl32(x / scale)), both by exact
    division, saturated at 448: computed on the host in FP64 and rounded once
    to FP32, they are the kernel's bits."""
    torch.manual_seed(0)
    x = torch.randn(48, _K, device="cuda").bfloat16()
    x[3] = 0  # an all-zero row takes the floor scale and zero codes
    codes, scales = _rows_fp8(x)
    host = x.float().cpu()
    scale = torch.clamp((host.abs().amax(1).double() / 448.0).float(), min=1e-12)
    expected = (host.double() / scale.double()[:, None]).float().clamp(-448, 448)
    assert torch.equal(scales.view(-1).cpu(), scale)
    assert torch.equal(
        codes.cpu().view(torch.uint8),
        expected.to(torch.float8_e4m3fn).view(torch.uint8),
    )


@pytest.mark.parametrize("n", _SHAPES)
@pytest.mark.parametrize("m", ["first", 48, 64, 65, 96, 128])
def test_matches_the_fp64_product_of_its_operands(n, m):
    """Against FP64 ``(A * a_s) @ (B * b_s).T`` of the same codes, the errors
    are the MFMA's accumulation and the one BF16 rounding of the output.
    BF16 keeps 8 significant bits, so rounding to nearest is within half a
    step, 2^-8 of the value (rtol). The FP8 MFMA's FP32 accumulation is not
    exact per product: its partial sums measured within 2^-15.7 of the
    partial's largest magnitude, which near-zero outputs see as absolute
    error, so atol is 2^-12 of the largest output (16x that)."""
    m = _FIRST[n] if m == "first" else m
    torch.manual_seed(n + m)
    weight, weight_scales = prepare_fp8_channel_weight(
        (torch.randn(n, _K, device="cuda") * 0.02).bfloat16()
    )
    codes, scales = _rows_fp8(torch.randn(m, _K, device="cuda").bfloat16())
    out = _skinny(codes, scales, weight, weight_scales)
    reference = (codes.double() * scales.double().view(-1, 1)) @ (
        weight.double() * weight_scales.double().view(-1, 1)
    ).T
    atol = 2**-12 * reference.abs().max().item()
    torch.testing.assert_close(out.double(), reference, atol=atol, rtol=2**-8)


@pytest.mark.parametrize("n", _SHAPES)
@pytest.mark.parametrize("m", ["first", 64, 65, 128])
def test_small_integer_codes_are_bit_exact(n, m):
    """Codes in -2..2 with unit scales make every product and partial sum an
    exact small integer, so the output is the FP64 product rounded once to
    BF16: the tiling, the split order and the scaling are exact."""
    m = _FIRST[n] if m == "first" else m
    torch.manual_seed(n * m)
    codes = torch.randint(-2, 3, (m, _K), device="cuda").float().to(torch.float8_e4m3fn)
    weight = (
        torch.randint(-2, 3, (n, _K), device="cuda").float().to(torch.float8_e4m3fn)
    )
    ones_m = torch.ones(m, device="cuda")
    ones_n = torch.ones(n, device="cuda")
    out = _skinny(codes, ones_m, weight, ones_n)
    reference = (codes.double() @ weight.double().T).to(torch.bfloat16)
    assert torch.equal(_bits(out), _bits(reference))


@pytest.mark.parametrize("n", _SHAPES)
def test_deterministic_and_row_batch_invariant(n):
    """Repeated launches give the same bits, and every row's bits are the same
    whatever rows share its launch, across both row tiles."""
    torch.manual_seed(n)
    weight, weight_scales = prepare_fp8_channel_weight(
        (torch.randn(n, _K, device="cuda") * 0.02).bfloat16()
    )
    codes, scales = _rows_fp8(torch.randn(128, _K, device="cuda").bfloat16())
    full = _skinny(codes, scales, weight, weight_scales)
    assert torch.equal(
        _bits(full), _bits(_skinny(codes, scales, weight, weight_scales))
    )
    for m in (_FIRST[n], 40, 64, 65, 100):
        part = _skinny(
            codes[:m].contiguous(), scales[:m].contiguous(), weight, weight_scales
        )
        assert torch.equal(_bits(part), _bits(full[:m])), m


@pytest.mark.parametrize("n", _SHAPES)
def test_row_counts_compile_nothing_new(n):
    """M is a runtime argument: one binary per row tile serves the window."""
    weight, weight_scales = prepare_fp8_channel_weight(
        torch.randn(n, _K, device="cuda").bfloat16()
    )
    codes, scales = _rows_fp8(torch.randn(128, _K, device="cuda").bfloat16())
    for m in (_FIRST[n], 65):
        _skinny(codes[:m].contiguous(), scales[:m].contiguous(), weight, weight_scales)
    with assert_no_triton_compile(
        gluon_mm_fp8_skinny_gfx950, gluon_mm_fp8_skinny_reduce_gfx950
    ):
        for m in [
            m for m in (18, 31, 32, 34, 47, 63, 64, 66, 97, 127, 128) if m > _FIRST[n]
        ]:
            _skinny(
                codes[:m].contiguous(), scales[:m].contiguous(), weight, weight_scales
            )


def test_refuses_problems_without_a_config():
    """Shapes and row counts outside the measured configs are refused."""
    assert supports_gluon_mm_fp8_skinny_gfx950(17, 4608, _K)
    assert not supports_gluon_mm_fp8_skinny_gfx950(16, 4608, _K)
    assert supports_gluon_mm_fp8_skinny_gfx950(17, 1024, _K)
    assert not supports_gluon_mm_fp8_skinny_gfx950(16, 1024, _K)
    assert not supports_gluon_mm_fp8_skinny_gfx950(129, 1024, _K)
    assert not supports_gluon_mm_fp8_skinny_gfx950(48, 2304, _K)
    codes = torch.zeros(48, _K, device="cuda", dtype=torch.float8_e4m3fn)
    weight = torch.zeros(2304, _K, device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="no config"):
        launch_gluon_mm_fp8_skinny_gfx950(
            codes,
            torch.ones(48, device="cuda"),
            weight,
            torch.ones(2304, device="cuda"),
            out=None,
        )


@pytest.mark.parametrize("n", _SHAPES)
@pytest.mark.parametrize("m", [33, 63, 127])
def test_writes_no_row_past_m(n, m):
    """Into a larger NaN-filled buffer, rows at and past M keep their bits: M
    fills neither its row tile (64 or 128) nor its 16-row reduce tile, and
    63 and 127 are one below a tile. Rows below M match the plain launch."""
    torch.manual_seed(m)
    weight, weight_scales = prepare_fp8_channel_weight(
        (torch.randn(n, _K, device="cuda") * 0.02).bfloat16()
    )
    codes, scales = _rows_fp8(torch.randn(m, _K, device="cuda").bfloat16())
    buffer = torch.full((m + 16, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    before = buffer[m:].clone()
    out = _skinny(codes, scales, weight, weight_scales)
    mm(
        codes,
        weight.t(),
        A_scales=scales,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        out=buffer[:m],
        quant="fp8",
        override="gluon_mm_fp8_skinny_gfx950",
    )
    assert torch.equal(_bits(buffer[:m]), _bits(out))
    assert torch.equal(_bits(buffer[m:]), _bits(before))


@pytest.mark.parametrize("n", _SHAPES)
def test_startup_warmup_compiles_every_row_tile(n):
    """One call at the shape's first row under warm-up compiles both row tiles
    and the reduce, so no row count of the window compiles while serving."""
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8 import skinny

    weight, weight_scales = prepare_fp8_channel_weight(
        torch.randn(n, _K, device="cuda").bfloat16()
    )
    codes, scales = _rows_fp8(torch.randn(128, _K, device="cuda").bfloat16())
    for kernel in (gluon_mm_fp8_skinny_gfx950, gluon_mm_fp8_skinny_reduce_gfx950):
        kernel.device_caches.clear()
    skinny._warmed.clear()
    first = _FIRST[n]
    with kernel_warmup(enabled=True):
        _skinny(
            codes[:first].contiguous(),
            scales[:first].contiguous(),
            weight,
            weight_scales,
        )
    with assert_no_triton_compile(
        gluon_mm_fp8_skinny_gfx950, gluon_mm_fp8_skinny_reduce_gfx950
    ):
        for m in range(first, 129, 7):
            _skinny(
                codes[:m].contiguous(), scales[:m].contiguous(), weight, weight_scales
            )


@pytest.mark.parametrize("n", _SHAPES)
def test_selected_inside_its_window_and_declined_outside(n):
    """An FP8 call with row and channel scales selects this kernel at the
    configured shape and 17-128 rows; outside that window, for another shape,
    for a per-tensor activation scale, or for an offset or strided operand it
    declines (no raise) to the scaled FP8 GEMM. An online-quantized call
    (BF16 rows, no activation scales) takes it; a BF16 call never does."""
    from tokenspeed_kernel.ops.gemm import _mm_selected_kernel_name as mm_kernel_name

    torch.manual_seed(n)
    weight, weight_scales = prepare_fp8_channel_weight(
        (torch.randn(n, _K, device="cuda") * 0.02).bfloat16()
    )
    codes, scales = _rows_fp8(torch.randn(130, _K, device="cuda").bfloat16())

    def name(m, features, a=None, a_s=None, w=weight, ws=weight_scales):
        return mm_kernel_name(
            codes[:m].contiguous() if a is None else a,
            w.t(),
            A_scales=scales[:m].contiguous() if a_s is None else a_s,
            B_scales=ws,
            out_dtype=torch.bfloat16,
            quant="fp8",
            features=features,
        )

    streaming = frozenset({"weight_streaming"})
    for m in (17, 48, 128):
        assert name(m, None) == name(m, streaming) == _NAME
        out = mm(
            codes[:m].contiguous(),
            weight.t(),
            A_scales=scales[:m].contiguous(),
            B_scales=weight_scales,
            out_dtype=torch.bfloat16,
            quant="fp8",
        )
        named = _skinny(
            codes[:m].contiguous(), scales[:m].contiguous(), weight, weight_scales
        )
        assert torch.equal(_bits(out), _bits(named))
    for m in (16, 129):
        assert name(m, None) == "triton_mm_fp8_scaled"
        assert name(m, streaming) is None
    other, other_scales = prepare_fp8_channel_weight(
        torch.randn(2304, _K, device="cuda").bfloat16()
    )
    assert name(48, None, w=other, ws=other_scales) == "triton_mm_fp8_scaled"
    # A_scales 4 bytes off 16-byte alignment, a row-strided A, and a
    # row-strided weight copy (every other row of a wider one).
    wide = torch.zeros(n, 2 * _K, dtype=weight.dtype, device="cuda")
    wide[:, :_K] = weight
    # An [M, 1] column view of [M, 2] scales: per row, but not contiguous.
    paired = torch.zeros(130, 2, device="cuda")
    paired[:, :1] = scales.reshape(-1, 1)
    column = paired[:48, :1]
    for odd in (
        {"a_s": scales[1:49]},
        {"a": codes[:96:2]},
        {"w": wide[:, :_K]},
        {"a_s": column},
    ):
        assert name(48, None, **odd) == "triton_mm_fp8_scaled"
        assert name(48, streaming, **odd) is None
    # A per-tensor activation scale (a static input scale) with per-channel
    # weight scales is not this kernel's problem: it declines to the scaled
    # FP8 GEMM, whose output mm then returns.
    tensor_scale = scales[:64].amax().reshape(1)
    assert name(64, None, a_s=tensor_scale) == "triton_mm_fp8_scaled"
    assert name(64, streaming, a_s=tensor_scale) is None
    static_rows = codes[:64].contiguous()
    by_mm = mm(
        static_rows,
        weight.t(),
        A_scales=tensor_scale,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="fp8",
    )
    by_triton = mm(
        static_rows,
        weight.t(),
        A_scales=tensor_scale,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="fp8",
        override="triton_mm_fp8_scaled",
    )
    assert torch.equal(_bits(by_mm), _bits(by_triton))
    # An online-quantized call (BF16 rows, no activation scales): mm quantizes
    # the rows per token first, so the kernel serves it, as it serves those
    # quantized rows passed in.
    rows_bf16 = torch.randn(64, _K, device="cuda").bfloat16()
    online = dict(
        A_scales=None, B_scales=weight_scales, out_dtype=torch.bfloat16, quant="fp8"
    )
    assert mm_kernel_name(rows_bf16, weight.t(), **online, features=None) == _NAME
    by_online = mm(rows_bf16, weight.t(), **online)
    online_codes, online_scales = quantize_fp8(rows_bf16, granularity="token")
    by_codes = mm(
        online_codes,
        weight.t(),
        A_scales=online_scales,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="fp8",
    )
    assert torch.equal(_bits(by_online), _bits(by_codes))
    # A BF16 call (quant None) never selects this FP8 kernel.
    plain = mm_kernel_name(
        torch.randn(48, _K, device="cuda").bfloat16(),
        torch.randn(n, _K, device="cuda").bfloat16(),
        A_scales=None,
        B_scales=None,
        out_dtype=torch.bfloat16,
        quant=None,
        features=None,
    )
    assert plain != _NAME


@pytest.mark.parametrize("n", _SHAPES)
def test_binaries_do_not_depend_on_where_the_tensors_live(n):
    """Triton keys a binary on a tensor's 16-byte alignment and, on AMD, on
    whether its storage fits in 2 GB. The kernels specialize no pointer, so
    the binaries warmed from small fresh tensors serve a weight copy, codes,
    scales and an output that are views at the far end of storages over 2 GB,
    bit for bit."""
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8 import skinny

    torch.manual_seed(n)
    weight, weight_scales = prepare_fp8_channel_weight(
        (torch.randn(n, _K, device="cuda") * 0.02).bfloat16()
    )
    codes, scales = _rows_fp8(torch.randn(128, _K, device="cuda").bfloat16())
    for kernel in (gluon_mm_fp8_skinny_gfx950, gluon_mm_fp8_skinny_reduce_gfx950):
        kernel.device_caches.clear()
    skinny._warmed.clear()
    with kernel_warmup(enabled=True):
        _skinny(
            codes[:48].contiguous(), scales[:48].contiguous(), weight, weight_scales
        )

    over_2gb = 2**31 + 2**20

    def far_end(t):
        # A view holding ``t`` in the last elements of a storage over 2 GB.
        count = over_2gb // t.element_size() + t.numel()
        arena = torch.empty(count, dtype=t.dtype, device="cuda")
        assert arena.untyped_storage().size() > 2**31
        view = arena[count - t.numel() :].view(t.shape)
        view.copy_(t)
        return view

    far_weight, far_weight_scales = far_end(weight), far_end(weight_scales)
    with assert_no_triton_compile(
        gluon_mm_fp8_skinny_gfx950, gluon_mm_fp8_skinny_reduce_gfx950
    ):
        for m in (_FIRST[n], 64, 65, 128):
            far_codes, far_scales = far_end(codes[:m]), far_end(scales[:m])
            out = far_end(torch.empty(m, n, dtype=torch.bfloat16, device="cuda"))
            mm(
                far_codes,
                far_weight.t(),
                A_scales=far_scales,
                B_scales=far_weight_scales,
                out_dtype=torch.bfloat16,
                out=out,
                quant="fp8",
                override="gluon_mm_fp8_skinny_gfx950",
            )
            expected = _skinny(
                codes[:m].contiguous(), scales[:m].contiguous(), weight, weight_scales
            )
            assert torch.equal(_bits(out), _bits(expected))
            del far_codes, far_scales, out


@pytest.mark.parametrize("n", _SHAPES)
def test_runs_the_unscaled_fp8_mfma(n):
    """Every compiled binary multiplies with the unscaled FP8 MFMA, not the
    scaled one with unit scales."""
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8 import skinny

    gluon_mm_fp8_skinny_gfx950.device_caches.clear()
    skinny._warmed.clear()
    skinny.warm_gluon_mm_fp8_skinny_gfx950(n, _K, torch.device("cuda"))
    # The compiled binaries of the warm hook's compile-ahead calls; their
    # AMDGCN is what runs.
    binaries = compiled_kernels(gluon_mm_fp8_skinny_gfx950)
    assert len(binaries) == len(skinny.SKINNY_FP8_ROW_TILES)
    for compiled in binaries:
        isa = compiled.asm["amdgcn"]
        assert "v_mfma_f32_32x32x64_f8f6f4" in isa
        assert "v_mfma_scale" not in isa


@pytest.mark.parametrize("n", _SHAPES)
def test_graph_replay_matches_eager(n):
    """Captured in a HIP graph, one row count inside each row tile replays the
    eager bits, with new inputs copied into the captured buffers."""
    torch.manual_seed(n)
    weight, weight_scales = prepare_fp8_channel_weight(
        (torch.randn(n, _K, device="cuda") * 0.02).bfloat16()
    )
    for m in (48, 100):
        codes, scales = _rows_fp8(torch.randn(m, _K, device="cuda").bfloat16())
        _skinny(codes, scales, weight, weight_scales)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _skinny(codes, scales, weight, weight_scales)
        new_codes, new_scales = _rows_fp8(torch.randn(m, _K, device="cuda").bfloat16())
        codes.copy_(new_codes)
        scales.copy_(new_scales)
        graph.replay()
        torch.cuda.synchronize()
        eager = _skinny(new_codes, new_scales, weight, weight_scales)
        assert torch.equal(_bits(captured), _bits(eager))


def _fp64_error_bound(codes, scales, weight, weight_scales):
    """The FP64 product and the per-element error allowed in BF16: half a BF16
    ulp of the exact value (rounding the output once) plus the worst-case FP32
    summation error over K, ``K * 2^-24 * sum |a_k b_k|``."""
    a = codes.double() * scales.double().reshape(-1, 1)
    b = weight.double() * weight_scales.double().reshape(-1, 1)
    exact = a @ b.t()
    magnitude = a.abs() @ b.abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(exact.abs().clamp_min(2.0**-126))) - 7)
    return exact, ulp / 2 + a.shape[1] * 2.0**-24 * magnitude


@pytest.mark.parametrize("n", _SHAPES)
@pytest.mark.parametrize("m", [17, 33, 64, 65, 128])
def test_same_rounding_class_as_the_scaled_fp8_gemm(n, m):
    """Against the FP64 product, the skinny kernel and triton_mm_fp8_scaled
    each stay within one BF16 rounding of the output plus FP32 summation
    error, so callers that switch kernels at these shapes keep their
    precision."""
    torch.manual_seed(1000 * m + n)
    weight, weight_scales = prepare_fp8_channel_weight(
        (torch.randn(n, _K, device="cuda") * 0.02).bfloat16()
    )
    codes, scales = _rows_fp8(torch.randn(m, _K, device="cuda").bfloat16())
    exact, bound = _fp64_error_bound(codes, scales, weight, weight_scales)
    for override in (_NAME, "triton_mm_fp8_scaled"):
        out = mm(
            codes,
            weight.t(),
            A_scales=scales,
            B_scales=weight_scales,
            out_dtype=torch.bfloat16,
            quant="fp8",
            override=override,
        )
        error = (out.double() - exact).abs()
        assert bool((error <= bound).all()), (override, float((error / bound).max()))
