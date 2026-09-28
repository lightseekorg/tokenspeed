from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import mm
from tokenspeed_kernel.ops.gemm import _online_quantize_mxfp8
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="MiniMax-M3 MXFP8 checkpoint support targets NVIDIA GPUs.",
)


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
    q_a, a_scales = _online_quantize_mxfp8(a, [1, 32], "ue8m0", enable_pdl=False)
    activation_scales = torch.exp2(a_scales.float() - 127.0).repeat_interleave(
        32, dim=1
    )
    ref = (q_a.float() * activation_scales) @ (b.float() * scales).t()
    torch.testing.assert_close(out.float(), ref, atol=1e-3, rtol=5e-3)
