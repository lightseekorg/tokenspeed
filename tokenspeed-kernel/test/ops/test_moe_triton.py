from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    _per_token_group_quant_8bit_raw,
    per_token_group_quant_fp8,
)
from tokenspeed_kernel.ops.moe.triton import _normalize_fp8_group_scale_layout


def test_normalize_fp8_group_scale_layout_handles_trtllm_flattened_scales():
    A = torch.empty((3, 2048))
    expected_scale_k = 16
    padded = torch.arange(expected_scale_k * 4, dtype=torch.float32)

    scales = _normalize_fp8_group_scale_layout(A, padded, expected_scale_k)

    assert scales.shape == (3, expected_scale_k)
    torch.testing.assert_close(scales, padded.view(expected_scale_k, 4)[:, :3].T)


def test_normalize_fp8_group_scale_layout_handles_total_count_padding():
    A = torch.empty((2, 768))
    expected_scale_k = 6
    padded = torch.arange(32, dtype=torch.float32)

    scales = _normalize_fp8_group_scale_layout(A, padded, expected_scale_k)

    assert scales.shape == (2, expected_scale_k)
    torch.testing.assert_close(
        scales,
        padded[: expected_scale_k * 4].view(expected_scale_k, 4)[:, :2].T,
    )


def test_normalize_fp8_group_scale_layout_transposes_column_major_scales():
    A = torch.empty((5, 2048))
    expected_scale_k = 16
    column_major = torch.arange(5 * expected_scale_k, dtype=torch.float32).view(
        expected_scale_k, 5
    )

    scales = _normalize_fp8_group_scale_layout(A, column_major, expected_scale_k)

    assert scales.shape == (5, expected_scale_k)
    torch.testing.assert_close(scales, column_major.T.contiguous())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_normalize_fp8_group_scale_layout_matches_raw_row_major_quantizer():
    for m, k in ((2, 768), (5, 768), (27, 768), (5, 2048)):
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        _, trtllm_scales = per_token_group_quant_fp8(x, 128)
        _, raw_scales = _per_token_group_quant_8bit_raw(
            x,
            128,
            dtype=torch.float8_e4m3fn,
            column_major_scales=False,
        )

        normalized = _normalize_fp8_group_scale_layout(
            x,
            trtllm_scales,
            expected_scale_k=k // 128,
        )

        torch.testing.assert_close(normalized, raw_scales)
