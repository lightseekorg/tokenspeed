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

"""Tests for fused mHC pre-reduction + RMSNorm + group-32 UE8M0 FP8 quantization."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (
    stage_mxfp4_mega_moe_inputs,
)
from tokenspeed_kernel.ops.moe.deep_gemm import DeepGemmMegaMoeBuffers
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.residual import mhc_apply_pre_quantize_fp8
from tokenspeed_kernel.ops.residual.triton import (
    _mhc_apply_pre_fp8_group32_kernel,
    triton_mhc_apply_pre,
    triton_mhc_apply_pre_fp8_group32,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _min_shared_rows(tokens: int, shared_block_m: int) -> int:
    blocks = (tokens + shared_block_m - 1) // shared_block_m
    aligned_block = (shared_block_m + 127) // 128 * 128
    return blocks * aligned_block


def _make_mn_major_shared_sf(
    tokens: int, hidden: int, shared_block_m: int, *, extra_col_stride: int = 0
) -> torch.Tensor:
    col_stride = max(1, _min_shared_rows(tokens, shared_block_m) + extra_col_stride)
    backing = torch.full(
        (hidden // 128, col_stride), -1, dtype=torch.int32, device="cuda"
    )
    return backing[:, :tokens].transpose(0, 1)


@pytest.mark.parametrize("prefix", [(), (0,), (1,), (17,), (65,), (2, 3)])
@pytest.mark.parametrize("hidden", [128, 5120])
def test_triton_mhc_apply_pre_fp8_canonical_matches_unfused(
    prefix: tuple[int, ...], hidden: int
) -> None:
    torch.manual_seed(11)
    residual = (torch.randn(*prefix, 4, hidden, device="cuda") * 0.5).bfloat16()
    if residual.numel() > 0:
        # Exercise the 1e-4 amax floor on a zeroed stream group as well.
        residual[..., :64] = 0.0
    pre = torch.randn(*prefix, 4, device="cuda", dtype=torch.float32) * 0.4 + 0.25
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    norm_eps = 1e-6

    out, codes, scales = triton_mhc_apply_pre_fp8_group32(
        residual, pre, norm_weight, norm_eps
    )
    expected_out = triton_mhc_apply_pre(
        residual, pre, norm_weight=norm_weight, norm_eps=norm_eps
    )
    expected_codes, expected_scales = quantize_fp8(
        expected_out.reshape(-1, hidden),
        granularity="token_group",
        group_size=32,
        scale_encoding="ue8m0",
        enable_pdl=False,
        override=None,
        solution=None,
    )

    torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(
        codes.view(torch.uint8), expected_codes.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)

    if current_platform().is_nvidia:
        api_out, api_codes, api_scales = mhc_apply_pre_quantize_fp8(
            residual, pre, norm_weight, norm_eps
        )
        torch.testing.assert_close(api_out, expected_out, atol=0, rtol=0)
        torch.testing.assert_close(
            api_codes.view(torch.uint8),
            expected_codes.view(torch.uint8),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(api_scales, expected_scales, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [0, 1, 17, 64, 133])
@pytest.mark.parametrize("hidden", [128, 5120])
@pytest.mark.parametrize("shared_block_m", [64, 128, 256])
def test_mhc_apply_pre_fp8_direct_staging_matches_stage_mxfp4(
    tokens: int, hidden: int, shared_block_m: int
) -> None:
    torch.manual_seed(23)
    residual = (torch.randn(tokens, 4, hidden, device="cuda") * 0.7).bfloat16()
    if tokens >= 2:
        residual[1, :, :128] = 0.0
    pre = torch.randn(tokens, 4, device="cuda", dtype=torch.float32) * 0.5 + 0.1
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    norm_eps = 1e-6

    y_fp8_out = torch.empty((tokens, hidden), dtype=torch.float8_e4m3fn, device="cuda")
    y_routed_sf_out = torch.empty(
        (tokens, hidden // 128), dtype=torch.int32, device="cuda"
    )
    y_shared_sf_out = _make_mn_major_shared_sf(
        tokens, hidden, shared_block_m, extra_col_stride=16
    )

    out, ret_fp8, ret_sf = mhc_apply_pre_quantize_fp8(
        residual,
        pre,
        norm_weight,
        norm_eps,
        y_fp8_out=y_fp8_out,
        y_routed_sf_out=y_routed_sf_out,
        y_shared_sf_out=y_shared_sf_out,
        shared_sf_block_m=shared_block_m,
    )
    assert ret_fp8 is y_fp8_out
    assert ret_sf is y_routed_sf_out

    expected_out = triton_mhc_apply_pre(
        residual, pre, norm_weight=norm_weight, norm_eps=norm_eps
    )
    ref_fp8 = torch.empty_like(y_fp8_out)
    ref_routed_sf = torch.empty_like(y_routed_sf_out)
    ref_shared_sf = _make_mn_major_shared_sf(
        tokens, hidden, shared_block_m, extra_col_stride=16
    )
    dummy_ids = torch.zeros((tokens, 2), dtype=torch.int32, device="cuda")
    dummy_weights = torch.ones((tokens, 2), dtype=torch.float32, device="cuda")
    dummy_idx_out = torch.empty((tokens, 2), dtype=torch.int64, device="cuda")
    dummy_w_out = torch.empty((tokens, 2), dtype=torch.float32, device="cuda")
    stage_mxfp4_mega_moe_inputs(
        expected_out,
        dummy_weights,
        dummy_ids,
        ref_fp8,
        ref_routed_sf,
        dummy_idx_out,
        dummy_w_out,
        ref_shared_sf,
        shared_block_m,
        None,
    )

    torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(
        y_fp8_out.view(torch.uint8), ref_fp8.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(y_routed_sf_out, ref_routed_sf, atol=0, rtol=0)
    torch.testing.assert_close(
        y_shared_sf_out.contiguous(), ref_shared_sf.contiguous(), atol=0, rtol=0
    )


def test_mhc_apply_pre_fp8_direct_validation_and_routing_only_view() -> None:
    tokens, hidden = 4, 256
    residual = torch.randn(tokens, 4, hidden, device="cuda", dtype=torch.bfloat16)
    pre = torch.randn(tokens, 4, device="cuda", dtype=torch.float32)
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    y_fp8 = torch.empty((tokens, hidden), dtype=torch.float8_e4m3fn, device="cuda")
    y_routed_sf = torch.empty((tokens, hidden // 128), dtype=torch.int32, device="cuda")
    y_shared_sf = _make_mn_major_shared_sf(tokens, hidden, 64)

    with pytest.raises(ValueError, match="Direct|Invalid|shared_sf_block_m"):
        mhc_apply_pre_quantize_fp8(
            residual,
            pre,
            weight,
            1e-6,
            y_fp8_out=y_fp8,
            y_routed_sf_out=y_routed_sf,
            y_shared_sf_out=None,
            shared_sf_block_m=64,
        )
    with pytest.raises(ValueError, match="shared_sf_block_m must be zero"):
        mhc_apply_pre_quantize_fp8(residual, pre, weight, 1e-6, shared_sf_block_m=64)
    with pytest.raises(ValueError, match="Invalid MegaMoE FP8 destination targets"):
        mhc_apply_pre_quantize_fp8(
            residual,
            pre,
            weight,
            1e-6,
            y_fp8_out=y_fp8,
            y_routed_sf_out=y_routed_sf,
            y_shared_sf_out=y_shared_sf,
            shared_sf_block_m=256,
        )

    full = DeepGemmMegaMoeBuffers(
        x=y_fp8,
        x_sf=y_routed_sf,
        shared_x_sf=y_shared_sf,
        shared_block_m=64,
        topk_idx=torch.empty((tokens, 2), dtype=torch.int64, device="cuda"),
        topk_weights=torch.empty((tokens, 2), dtype=torch.float32, device="cuda"),
    )
    routing_only = full.routing_only()
    assert routing_only.x_quantized is None
    assert routing_only.routing_out[0] is full.topk_weights
    assert routing_only.routing_out[1] is full.topk_idx


def test_mhc_apply_pre_fp8_no_triton_recompile_across_shapes() -> None:
    hidden = 5120
    norm_eps = 1e-6
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)

    # Warm one canonical call and one direct-staging call.
    triton_mhc_apply_pre_fp8_group32(
        torch.randn(1, 4, hidden, device="cuda", dtype=torch.bfloat16),
        torch.ones(1, 4, device="cuda", dtype=torch.float32),
        weight,
        norm_eps,
    )
    triton_mhc_apply_pre_fp8_group32(
        torch.randn(1, 4, hidden, device="cuda", dtype=torch.bfloat16),
        torch.ones(1, 4, device="cuda", dtype=torch.float32),
        weight,
        norm_eps,
        y_fp8_out=torch.empty((1, hidden), dtype=torch.float8_e4m3fn, device="cuda"),
        y_routed_sf_out=torch.empty(
            (1, hidden // 128), dtype=torch.int32, device="cuda"
        ),
        y_shared_sf_out=_make_mn_major_shared_sf(1, hidden, 64),
        shared_sf_block_m=64,
    )

    with assert_no_triton_compile(_mhc_apply_pre_fp8_group32_kernel):
        for tokens, shared_block_m, extra_stride in (
            (2, 64, 0),
            (7, 128, 32),
            (19, 128, 64),
            (65, 256, 128),
            (130, 256, 0),
        ):
            residual = torch.randn(
                tokens, 4, hidden, device="cuda", dtype=torch.bfloat16
            )
            pre = torch.randn(tokens, 4, device="cuda", dtype=torch.float32)
            triton_mhc_apply_pre_fp8_group32(residual, pre, weight, norm_eps)
            triton_mhc_apply_pre_fp8_group32(
                residual,
                pre,
                weight,
                norm_eps,
                y_fp8_out=torch.empty(
                    (tokens, hidden), dtype=torch.float8_e4m3fn, device="cuda"
                ),
                y_routed_sf_out=torch.empty(
                    (tokens, hidden // 128), dtype=torch.int32, device="cuda"
                ),
                y_shared_sf_out=_make_mn_major_shared_sf(
                    tokens, hidden, shared_block_m, extra_col_stride=extra_stride
                ),
                shared_sf_block_m=shared_block_m,
            )


def test_mhc_apply_pre_fp8_selects_triton_for_direct_staging() -> None:
    sig = format_signature(
        residual=dense_tensor_format(torch.bfloat16),
        pre=dense_tensor_format(torch.float32),
    )
    direct = select_kernel(
        "residual",
        "mhc_apply_pre_fp8_group32",
        sig,
        traits={
            "buffer_offsets_fit_int32": True,
            "hc_mult": 4,
            "supports_direct_staging": True,
        },
    )
    assert direct.name == "triton_mhc_apply_pre_fp8_group32"
    large = select_kernel(
        "residual",
        "mhc_apply_pre_fp8_group32",
        sig,
        traits={
            "buffer_offsets_fit_int32": False,
            "hc_mult": 4,
            "supports_direct_staging": True,
        },
    )
    assert large.name == "triton_mhc_apply_pre_fp8_group32"


def test_mhc_apply_pre_fp8_large_offset_int64() -> None:
    hidden = 256
    # Row 2 starts past 2**31 BF16 elements (row 2 * stride = 2**31 + 1024).
    row_stride = (1 << 30) + 512
    total_elements = 2 * row_stride + 4 * hidden
    needed_bytes = total_elements * torch.bfloat16.itemsize
    if torch.cuda.mem_get_info()[0] < needed_bytes + (512 << 20):
        pytest.skip("requires >4.5 GiB free device memory")

    storage = torch.zeros(total_elements, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(31)
    row0 = (torch.randn(4, hidden, device="cuda") * 0.5 + 1.0).bfloat16()
    row1 = (torch.randn(4, hidden, device="cuda") * 0.5 - 2.0).bfloat16()
    row2 = (torch.randn(4, hidden, device="cuda") * 0.8 + 3.5).bfloat16()
    storage[: 4 * hidden].copy_(row0.reshape(-1))
    storage[row_stride : row_stride + 4 * hidden].copy_(row1.reshape(-1))
    storage[2 * row_stride : 2 * row_stride + 4 * hidden].copy_(row2.reshape(-1))

    residual_strided = torch.as_strided(
        storage, size=(3, 4, hidden), stride=(row_stride, hidden, 1)
    )
    residual_compact = torch.stack([row0, row1, row2], dim=0)
    pre = torch.tensor(
        [[0.4, 0.2, 0.3, 0.1], [0.1, 0.5, 0.2, 0.2], [0.25, 0.25, 0.25, 0.25]],
        device="cuda",
        dtype=torch.float32,
    )
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    y_fp8_got = torch.empty((3, hidden), dtype=torch.float8_e4m3fn, device="cuda")
    y_sf_got = torch.empty((3, hidden // 128), dtype=torch.int32, device="cuda")
    y_shared_got = _make_mn_major_shared_sf(3, hidden, 64)
    out_got, _, _ = triton_mhc_apply_pre_fp8_group32(
        residual_strided,
        pre,
        weight,
        1e-6,
        y_fp8_out=y_fp8_got,
        y_routed_sf_out=y_sf_got,
        y_shared_sf_out=y_shared_got,
        shared_sf_block_m=64,
    )

    y_fp8_ref = torch.empty_like(y_fp8_got)
    y_sf_ref = torch.empty_like(y_sf_got)
    y_shared_ref = _make_mn_major_shared_sf(3, hidden, 64)
    out_ref, _, _ = triton_mhc_apply_pre_fp8_group32(
        residual_compact,
        pre,
        weight,
        1e-6,
        y_fp8_out=y_fp8_ref,
        y_routed_sf_out=y_sf_ref,
        y_shared_sf_out=y_shared_ref,
        shared_sf_block_m=64,
    )
    torch.testing.assert_close(out_got, out_ref, atol=0, rtol=0)
    torch.testing.assert_close(
        y_fp8_got.view(torch.uint8), y_fp8_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(y_sf_got, y_sf_ref, atol=0, rtol=0)
    torch.testing.assert_close(
        y_shared_got.contiguous(), y_shared_ref.contiguous(), atol=0, rtol=0
    )

    # Bad dtype raises NoKernelFoundError rather than falling back silently when
    # direct targets are provided.
    with pytest.raises(NoKernelFoundError):
        mhc_apply_pre_quantize_fp8(
            residual_compact.float(),
            pre,
            weight,
            1e-6,
            y_fp8_out=y_fp8_ref,
            y_routed_sf_out=y_sf_ref,
            y_shared_sf_out=y_shared_ref,
            shared_sf_block_m=64,
        )
