from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import (
    _online_quantize_mxfp8,
    mm,
    prepare_mxfp8_weight_scales,
)
from tokenspeed_kernel.ops.gemm.flashinfer import swizzle_mxfp8_scale
from tokenspeed_kernel.ops.quantization.triton import (
    _fp8_group32_ue8m0_quantize_swizzled,
    _quantize_fp8_group32_swizzled,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.selection import kernel_override
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="MiniMax-M3 MXFP8 checkpoint support targets NVIDIA GPUs.",
)


@pytest.mark.skipif(
    not current_platform().is_blackwell_plus,
    reason="Packed UE8M0 activation scales are consumed on Blackwell.",
)
@pytest.mark.parametrize("m,k", [(1, 128), (5, 384), (7, 640)])
def test_packed_ue8m0_quant_pdl_matches_reference(device: str, m: int, k: int) -> None:
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    torch.manual_seed(0)
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)

    def quantize_packed_ue8m0(enable_pdl: bool) -> tuple[torch.Tensor, torch.Tensor]:
        q, scales = quantize_fp8(
            x,
            granularity="token_group",
            group_size=128,
            scale_encoding="ue8m0",
            enable_pdl=enable_pdl,
        )
        # Repack the canonical [m, k // 128] UE8M0 exponents into the
        # four-exponents-per-word layout, zero-filling tail bytes.
        if scales.dtype != torch.uint8:
            scales = (torch.log2(scales) + 127).to(torch.uint8)
        packs = (k // 128 + 3) // 4
        padded = torch.zeros((m, packs * 4), dtype=torch.uint8, device=device)
        padded[:, : k // 128] = scales
        return q, padded.view(torch.int32)

    outputs = [quantize_packed_ue8m0(enable_pdl) for enable_pdl in (False, True)]

    q, packed_scales = outputs[0]
    torch.cuda.synchronize()
    assert torch.equal(outputs[1][0], q)
    assert torch.equal(outputs[1][1], packed_scales)

    groups = x.float().view(m, k // 128, 128)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    absmax = groups.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2(torch.clamp(absmax / fp8_info.max, min=1e-10)))
    expected_q = torch.clamp(
        groups / torch.exp2(exponent).unsqueeze(-1),
        fp8_info.min,
        fp8_info.max,
    ).to(torch.float8_e4m3fn)
    assert torch.equal(q, expected_q.view(m, k))

    packs = (k // 128 + 3) // 4
    expected_bytes = torch.zeros((m, packs, 4), device=device, dtype=torch.int64)
    biased_exponents = torch.clamp(exponent + 127, 0, 255).to(torch.int64)
    expected_bytes.view(m, -1)[:, : k // 128] = biased_exponents
    actual_bytes = torch.stack(
        [(packed_scales.to(torch.int64) >> shift) & 0xFF for shift in (0, 8, 16, 24)],
        dim=-1,
    )
    assert torch.equal(actual_bytes, expected_bytes)


@pytest.mark.parametrize("override", [None, "triton_mm_fp8_blockscale"])
def test_triton_mxfp8_1x32_raw_ue8m0_weight(device: str, override: str | None) -> None:
    torch.manual_seed(0)
    m, n, k = 19, 128, 128
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.2
    b = (torch.randn(n, k, device=device) * 0.2).to(torch.float8_e4m3fn)
    b_scales = torch.empty(n, k // 32, device=device, dtype=torch.uint8)
    for group in range(k // 32):
        b_scales[:, group] = 126 + group % 3

    out = mm(
        a,
        b,
        B_scales=b_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
        override=override,
    )

    scales = torch.exp2(b_scales.float() - 127.0).repeat_interleave(32, dim=1)
    # Compare against the operands mm() actually quantizes; UE8M0 scales are
    # stored as biased exponent bytes.
    q_a, a_scales = _online_quantize_mxfp8(
        a, [1, 32], "ue8m0", enable_pdl=False, kernel_name=None
    )
    activation_scales = torch.exp2(a_scales.float() - 127.0).repeat_interleave(
        32, dim=1
    )
    ref = (q_a.float() * activation_scales) @ (b.float() * scales).t()
    torch.testing.assert_close(out.float(), ref, atol=1e-3, rtol=5e-3)


def _has_flashinfer_mxfp8() -> bool:
    try:
        from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_mxfp8
    except ImportError:
        return False
    return has_flashinfer_mxfp8()


requires_flashinfer_mxfp8 = pytest.mark.skipif(
    not _has_flashinfer_mxfp8(),
    reason="flashinfer mm_mxfp8 requires SM100/103 and a flashinfer build with the API",
)


def _quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp8_quantize

    q, s = mxfp8_quantize(x, is_sf_swizzled_layout=False)
    return q, s.view(x.shape[0], x.shape[1] // 32)


def _dequantize_mxfp8(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.float() * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)


@requires_flashinfer_mxfp8
@pytest.mark.parametrize("m", [4, 16, 512])
@pytest.mark.parametrize("n,k", [(2304, 6144), (6144, 2048), (128, 6144)])
def test_flashinfer_mxfp8_matches_triton_on_identical_operands(
    device: str, m: int, n: int, k: int
) -> None:
    torch.manual_seed(0)
    a_q, a_s = _quantize_mxfp8(torch.randn(m, k, device=device, dtype=torch.bfloat16))
    b_q, b_s = _quantize_mxfp8(
        torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    )

    outs = {}
    for name in ("flashinfer_mm_mxfp8", "triton_mm_fp8_blockscale"):
        outs[name] = mm(
            a_q,
            b_q,
            A_scales=a_s,
            B_scales=b_s,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[1, 32],
            override=name,
        )
    torch.testing.assert_close(
        outs["flashinfer_mm_mxfp8"].float(),
        outs["triton_mm_fp8_blockscale"].float(),
        atol=8e-3,
        rtol=2e-2,
    )


@requires_flashinfer_mxfp8
def test_flashinfer_mxfp8_selected_with_online_quant(device: str) -> None:
    from flashinfer import autotune
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import (
        ScaleFormat,
        format_signature,
        tensor_format,
    )

    torch.manual_seed(0)
    m, n, k = 16, 256, 512
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_q, b_s = _quantize_mxfp8(
        torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    )

    # Selection with no override resolves to the flashinfer kernel; a
    # solution pin recovers the Triton fallback.
    fp8 = torch.float8_e4m3fn
    sig = format_signature(
        a=tensor_format(
            "mxfp8",
            fp8,
            scale=ScaleFormat(
                storage_dtype=torch.uint8, granularity="block", block_shape=(1, 32)
            ),
        ),
        b=tensor_format(
            "mxfp8",
            fp8,
            scale=ScaleFormat(
                storage_dtype=torch.uint8, granularity="block", block_shape=(1, 32)
            ),
        ),
    )
    assert select_kernel("gemm", "mm", sig).name == "flashinfer_mm_mxfp8"
    assert (
        select_kernel("gemm", "mm", sig, solution="triton").name
        == "triton_mm_fp8_blockscale"
    )

    # Production layout: bf16 activations (online ue8m0 quant inside mm),
    # weight scales pre-swizzled at load time.
    # Exercise candidate selection: rc2 admitted invalid narrow persistent tiles.
    with autotune(tuning_buckets=[m]):
        out = mm(
            a,
            b_q,
            B_scales=swizzle_mxfp8_scale(b_s, n, k),
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[1, 32],
        )
    # mm() quantizes the activation online with the same FlashInfer quantizer;
    # dequantizing that result keeps quantization noise out of the comparison.
    a_q, a_s = _quantize_mxfp8(a)
    ref = _dequantize_mxfp8(a_q, a_s) @ _dequantize_mxfp8(b_q, b_s).t()
    torch.testing.assert_close(out.float(), ref, atol=1e-3, rtol=1e-2)


@requires_flashinfer_mxfp8
def test_flashinfer_mxfp8_square_weight_orientation(device: str) -> None:
    # A square [N, K] weight (M3's dense-MLP gate_up_proj is 6144x6144) must
    # be read as N-major; a transposed read produces a different result, so
    # exact agreement with the Triton kernel proves the orientation.
    torch.manual_seed(0)
    m, n = 8, 256
    k = n
    a_q, a_s = _quantize_mxfp8(torch.randn(m, k, device=device, dtype=torch.bfloat16))
    b_q, b_s = _quantize_mxfp8(
        torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
    )
    assert not torch.equal(
        b_q.view(torch.uint8), b_q.t().contiguous().view(torch.uint8)
    )

    outs = {}
    for name in ("flashinfer_mm_mxfp8", "triton_mm_fp8_blockscale"):
        outs[name] = mm(
            a_q,
            b_q,
            A_scales=a_s,
            B_scales=b_s,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[1, 32],
            override=name,
        )
    torch.testing.assert_close(
        outs["flashinfer_mm_mxfp8"].float(),
        outs["triton_mm_fp8_blockscale"].float(),
        atol=8e-3,
        rtol=2e-2,
    )


@requires_flashinfer_mxfp8
def test_swizzle_mxfp8_scale_matches_flashinfer_layout(device: str) -> None:
    from flashinfer import mxfp8_quantize

    torch.manual_seed(0)
    for m, k in [(4, 512), (19, 2048), (300, 6144)]:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        _, s_lin = mxfp8_quantize(x, is_sf_swizzled_layout=False)
        _, s_128 = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        assert torch.equal(swizzle_mxfp8_scale(s_lin.view(m, k // 32), m, k), s_128)


@requires_flashinfer_mxfp8
class TestPreparedMxfp8:
    @pytest.fixture(autouse=True)
    def enable_pdl(self):
        previous = pdl_enabled()
        pdl_enabled(True)
        try:
            yield
        finally:
            pdl_enabled(previous)

    @pytest.mark.parametrize(
        "rows,width,strided",
        [(0, 96, False), (1, 1280, True), (17, 5120, True), (129, 96, False)],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_quantization_matches_canonical(self, rows, width, strided, dtype):
        x = torch.randn(
            rows, width + (32 if strided else 0), device="cuda", dtype=dtype
        )[:, :width]
        if rows:
            x[0, :32] = -0.0
            x[0, 32:64] = 1e-7
            x[0, 64:96] = 448.0 / 64
        codes, scales = _quantize_fp8_group32_swizzled(x)
        reference_codes, reference_scales = _online_quantize_mxfp8(
            x, [1, 32], "ue8m0", enable_pdl=False, kernel_name=None
        )
        torch.testing.assert_close(
            codes.view(torch.uint8), reference_codes.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            scales,
            swizzle_mxfp8_scale(reference_scales, rows, width),
            rtol=0,
            atol=0,
        )

    def test_quantization_graph_and_dynamic_rows(self):
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)[:, :128]
        _quantize_fp8_group32_swizzled(x[:17])
        with assert_no_triton_compile(_fp8_group32_ue8m0_quantize_swizzled):
            for rows in (1, 3, 16, 64):
                _quantize_fp8_group32_swizzled(x[::2][:rows])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            codes, scales = _quantize_fp8_group32_swizzled(x[:17])
        for magnitude in (1.0, 1e-8, 0.0):
            x.normal_().mul_(magnitude)
            graph.replay()
            expected_codes, expected_scales = _online_quantize_mxfp8(
                x[:17], [1, 32], "ue8m0", enable_pdl=False, kernel_name=None
            )
            torch.testing.assert_close(
                codes.view(torch.uint8),
                expected_codes.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                scales, swizzle_mxfp8_scale(expected_scales, 17, 128), rtol=0, atol=0
            )

    @pytest.mark.parametrize("rows", [1, 129])
    def test_prepacked_mm_reuses_scales_without_repacking(self, monkeypatch, rows):
        weight = torch.randn(256, 384, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.randint(120, 126, (256, 12), device="cuda", dtype=torch.uint8)
        prepared = prepare_mxfp8_weight_scales(weight, scales)
        x = torch.randn(rows, 384, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(256, device="cuda", dtype=x.dtype)
        codes, input_scales = _online_quantize_mxfp8(
            x, [1, 32], "ue8m0", enable_pdl=False, kernel_name=None
        )
        expected = mm(
            codes,
            weight,
            A_scales=input_scales,
            B_scales=scales,
            bias=bias,
            out_dtype=x.dtype,
            quant="mxfp8",
            block_size=[1, 32],
            override="flashinfer_mm_mxfp8",
        )

        def unexpected_repack(*args, **kwargs):
            pytest.fail("prepacked mm repacked scales during forward")

        monkeypatch.setattr(
            "tokenspeed_kernel.ops.gemm.flashinfer.swizzle_mxfp8_scale",
            unexpected_repack,
        )
        actual = mm(
            x,
            weight,
            B_scales=prepared,
            bias=bias,
            out_dtype=x.dtype,
            quant="mxfp8",
            block_size=[1, 32],
            prepacked_scales=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        out = torch.empty_like(expected)
        assert (
            mm(
                x,
                weight,
                B_scales=prepared,
                bias=bias,
                out=out,
                out_dtype=x.dtype,
                quant="mxfp8",
                block_size=[1, 32],
                prepacked_scales=True,
            )
            is out
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "case",
        ["pdl_disabled", "gemm_override", "quantizer_override", "small_n", "small_k"],
    )
    def test_prepare_declines_when_native_kernel_does_not_apply(self, case):
        n, k = {"small_n": (64, 128), "small_k": (256, 96)}.get(case, (256, 128))
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 123, device="cuda", dtype=torch.uint8)
        if case == "pdl_disabled":
            pdl_enabled(False)
        gemm = "triton_mm_fp8_blockscale" if case == "gemm_override" else None
        quantizer = (
            "triton_quantize_fp8_group32_ue8m0"
            if case == "quantizer_override"
            else None
        )
        with kernel_override("gemm", "mm", gemm), kernel_override(
            "quantization", "fp8_with_scale", quantizer
        ):
            assert prepare_mxfp8_weight_scales(weight, scales) is None

    def test_prepacked_mm_rejects_canonical_scales(self):
        weight = torch.randn(256, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.full((256, 4), 123, device="cuda", dtype=torch.uint8)
        x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="prepare_mxfp8_weight_scales"):
            mm(
                x,
                weight,
                B_scales=scales,
                out_dtype=x.dtype,
                quant="mxfp8",
                block_size=[1, 32],
                prepacked_scales=True,
            )

    @pytest.mark.parametrize("rows", [1, 129])
    def test_prepacked_mm_takes_quantized_codes_without_weight_repacking(
        self, monkeypatch, rows
    ):
        from tokenspeed_kernel.ops.gemm import flashinfer as gemm_flashinfer

        weight = torch.randn(256, 384, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.randint(120, 126, (256, 12), device="cuda", dtype=torch.uint8)
        prepared = prepare_mxfp8_weight_scales(weight, scales)
        x = torch.randn(rows, 384, device="cuda", dtype=torch.bfloat16)
        codes, input_scales = _online_quantize_mxfp8(
            x, [1, 32], "ue8m0", enable_pdl=False, kernel_name=None
        )
        expected = mm(
            codes,
            weight,
            A_scales=input_scales,
            B_scales=scales,
            out_dtype=x.dtype,
            quant="mxfp8",
            block_size=[1, 32],
            override="flashinfer_mm_mxfp8",
        )
        swizzle = gemm_flashinfer.swizzle_mxfp8_scale

        def activation_only(sf, m, k):
            assert m == rows, "prepacked mm repacked the weight scales"
            return swizzle(sf, m, k)

        monkeypatch.setattr(gemm_flashinfer, "swizzle_mxfp8_scale", activation_only)
        actual = mm(
            codes,
            weight,
            A_scales=input_scales,
            B_scales=prepared,
            out_dtype=x.dtype,
            quant="mxfp8",
            block_size=[1, 32],
            prepacked_scales=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("argument", ["alpha", "solution", "override"])
    def test_prepacked_mm_rejects_arguments_it_cannot_honor(self, argument):
        weight = torch.randn(256, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.full((256, 4), 123, device="cuda", dtype=torch.uint8)
        x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        extra = {
            "alpha": {"alpha": torch.ones(1, device="cuda")},
            "solution": {"solution": "triton"},
            "override": {"override": "triton_mm_fp8_blockscale"},
        }[argument]
        with pytest.raises(ValueError, match="runs only flashinfer_mm_mxfp8"):
            mm(
                x,
                weight,
                B_scales=prepare_mxfp8_weight_scales(weight, scales),
                out_dtype=x.dtype,
                quant="mxfp8",
                block_size=[1, 32],
                prepacked_scales=True,
                **extra,
            )


@pytest.mark.parametrize("rows", [0, 1, 16, 129])
@pytest.mark.parametrize("hidden", [96, 1280])
def test_rmsnorm_group32_quantization_replay(rows, hidden):
    from tokenspeed_kernel.ops.layernorm import rmsnorm
    from tokenspeed_kernel.ops.quantization.triton import (
        triton_quantize_fp8_group32_ue8m0,
        triton_rmsnorm_fp8_group32_ue8m0,
    )

    x = torch.randn(rows, hidden + 32, device="cuda", dtype=torch.bfloat16)[:, :hidden]
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    triton_rmsnorm_fp8_group32_ue8m0(x, weight, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        normalized, codes, scales = triton_rmsnorm_fp8_group32_ue8m0(x, weight, 1e-6)
    for magnitude in [1.0, 1e-8, 0.0]:
        x.normal_().mul_(magnitude)
        graph.replay()
        if rows:
            expected = rmsnorm(x, weight, 1e-6, residual=None, out=None)
            torch.testing.assert_close(normalized, expected, rtol=0.008, atol=1e-7)
        reference = triton_quantize_fp8_group32_ue8m0(
            normalized, "token_group", 32, "ue8m0", False
        )
        for actual, expected in zip((codes, scales), reference):
            torch.testing.assert_close(
                actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
            )


def test_rmsnorm_group32_quantization_dynamic_rows():
    from tokenspeed_kernel.ops.quantization.triton import (
        _rmsnorm_fp8_group32_ue8m0,
        triton_rmsnorm_fp8_group32_ue8m0,
    )
    from utils import assert_no_triton_compile

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    triton_rmsnorm_fp8_group32_ue8m0(x[:17], weight, 1e-6)
    with assert_no_triton_compile(_rmsnorm_fp8_group32_ue8m0):
        for rows in (1, 16, 33, 64):
            triton_rmsnorm_fp8_group32_ue8m0(x[::2][:rows], weight, 1e-6)


def test_group32_infinite_amax_keeps_finite_codes_zero():
    from tokenspeed_kernel.ops.quantization.triton import (
        triton_quantize_fp8_group32_ue8m0,
    )

    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    x[0, 3] = torch.inf
    codes, scales = triton_quantize_fp8_group32_ue8m0(
        x, "token_group", 32, "ue8m0", False
    )
    finite_codes, finite_scales = triton_quantize_fp8_group32_ue8m0(
        x[1:], "token_group", 32, "ue8m0", False
    )
    # An infinite group gets the all-ones exponent and a zero reciprocal, so
    # its finite values quantize to +-0 rather than saturating with a flipped sign.
    assert scales[0, 0].item() == 255
    group = codes[0, :32].float()
    assert torch.isnan(group[3])
    assert (group[torch.arange(32, device="cuda") != 3] == 0).all()
    torch.testing.assert_close(
        codes[1:].view(torch.uint8), finite_codes.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(scales[1:], finite_scales, rtol=0, atol=0)


def test_group32_reciprocal_finite_bf16_constant_blocks():
    from tokenspeed_kernel.ops.quantization.triton import (
        triton_quantize_fp8_group32_ue8m0,
    )

    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    x = values[:, None].expand(-1, 32).contiguous()
    codes, scales = triton_quantize_fp8_group32_ue8m0(
        x, "token_group", 32, "ue8m0", False
    )
    amax = x.float().abs().amax(dim=1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax * (1.0 / 448.0))))
    reference = (x.float() / scale[:, None]).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        codes.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scales.view(torch.float8_e8m0fnu).float().reshape(-1), scale, rtol=0, atol=0
    )
