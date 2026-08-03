from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import mm
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="MiniMax-M3 MXFP8 checkpoint support targets NVIDIA GPUs.",
)


def test_triton_mxfp8_1x32_raw_ue8m0_weight(device: str) -> None:
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
        override="triton_mm_fp8_blockscale",
    )

    scales = torch.exp2(b_scales.float() - 127.0).repeat_interleave(32, dim=1)
    ref = a.float() @ (b.float() * scales).t()
    torch.testing.assert_close(out.float(), ref, atol=0.08, rtol=0.12)


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
    from tokenspeed_kernel.ops.gemm.fp8_utils import swizzle_mxfp8_scale
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
                storage_dtype=torch.float32, granularity="block", block_shape=(1, 32)
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
    out = mm(
        a,
        b_q,
        B_scales=swizzle_mxfp8_scale(b_s, n, k),
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
    )
    scales = torch.exp2(b_s.float() - 127.0).repeat_interleave(32, dim=1)
    ref = a.float() @ (b_q.float() * scales).t()
    rel = (torch.norm(out.float() - ref) / torch.norm(ref)).item()
    assert rel < 5e-2, f"rel_l2={rel}"


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
    from tokenspeed_kernel.ops.gemm.fp8_utils import swizzle_mxfp8_scale

    torch.manual_seed(0)
    for m, k in [(4, 512), (19, 2048), (300, 6144)]:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        _, s_lin = mxfp8_quantize(x, is_sf_swizzled_layout=False)
        _, s_128 = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        assert torch.equal(swizzle_mxfp8_scale(s_lin.view(m, k // 32), m, k), s_128)
