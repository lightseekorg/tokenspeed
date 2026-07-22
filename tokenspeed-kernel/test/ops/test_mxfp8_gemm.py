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
