from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import mm
from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8
from tokenspeed_kernel.ops.gemm.triton import w8a8_block_fp8_matmul_triton
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.thirdparty.triton.aiter_fp8_gemm import preshuffle_fp8_weight

pytestmark = pytest.mark.skipif(
    not (
        current_platform().is_amd
        and current_platform().arch_version == ArchVersion(9, 5)
    ),
    reason="AITER-derived FP8 GEMM targets gfx950",
)


@pytest.mark.parametrize("m", [4, 16, 64])
def test_aiter_preshuffled_fp8_matches_canonical(device: str, m: int) -> None:
    torch.manual_seed(89)
    n, k = 2048, 7168
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    weight = (torch.randn(n, k, device=device) * 0.1).to(torch.float8_e4m3fn)
    weight_scale = torch.rand(n // 128, k // 128, device=device)
    x_q, x_scale = per_token_group_quant_fp8(x, 128, column_major_scales=False)
    _, column_major_scale = per_token_group_quant_fp8(x, 128, column_major_scales=True)

    expected = w8a8_block_fp8_matmul_triton(
        x_q,
        weight,
        x_scale,
        weight_scale,
        [128, 128],
        output_dtype=torch.bfloat16,
    )
    output = torch.empty_like(expected)
    actual = mm(
        x_q,
        preshuffle_fp8_weight(weight),
        A_scales=column_major_scale,
        B_scales=weight_scale,
        out=output,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override="triton_aiter_mm_fp8_blockscale_preshuffle_gfx950",
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, atol=3.2e-2, rtol=1e-2)
