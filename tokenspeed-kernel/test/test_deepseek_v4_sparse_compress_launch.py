# Copyright (c) 2026 LightSeek Foundation

"""Launch-config tests for the fused sparse compress cache insert."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from tokenspeed_kernel.ops.attention.triton import deepseek_v4 as ops


def _launch(compress_ratio, overlap=False, wide_supported=True):
    recorded = {}

    class _Grid:
        def __getitem__(self, grid):
            def _call(*args, **kwargs):
                recorded["grid"] = grid
                recorded["num_warps"] = kwargs.get("num_warps")

            return _call

    m = 8
    with (
        patch.object(ops, "_deepseek_v4_fused_sparse_compress_cache_kernel", _Grid()),
        patch.object(
            ops, "_wide_compress_launch_supported", return_value=wide_supported
        ),
    ):
        ops.deepseek_v4_fused_sparse_compress_cache_insert(
            state_cache=torch.zeros(4, 8),
            token_to_req_indices=torch.zeros(m, dtype=torch.int32),
            positions=torch.zeros(m, dtype=torch.int32),
            compressor_slot_mapping=torch.zeros(m, dtype=torch.int32),
            block_table=torch.zeros(2, 4, dtype=torch.int32),
            compressor_block_size=64,
            rms_norm_weight=torch.ones(ops.DEEPSEEK_V4_HEAD_DIM),
            rms_norm_eps=1e-6,
            cos_sin_cache=torch.zeros(16, ops.DEEPSEEK_V4_ROPE_DIM),
            kv_cache_2d=torch.zeros(4, 8, dtype=torch.uint8),
            kv_slot_mapping=torch.zeros(m, dtype=torch.int32),
            kv_cache_block_size=64,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
    return recorded


def test_sparse_compress_uses_wide_launch_for_large_ratio():
    # Production HCA (ratio=128, overlap=False) reduces a 128-row window in
    # one CTA; the launch must request 16 warps there on supported targets.
    assert _launch(128)["num_warps"] == 16


def test_sparse_compress_keeps_narrow_launch_on_unsupported_target():
    # The wide launch is validated on sm100 only; other targets keep 4.
    assert _launch(128, wide_supported=False)["num_warps"] == 4


def test_sparse_compress_keeps_narrow_launch_for_small_ratio():
    assert _launch(4)["num_warps"] == 4


def test_sparse_compress_grid_is_one_program_per_token():
    assert _launch(128)["grid"] == (8,)


def test_sparse_compress_skips_empty_batch():
    mock = MagicMock()
    with patch.object(ops, "_deepseek_v4_fused_sparse_compress_cache_kernel", mock):
        ops.deepseek_v4_fused_sparse_compress_cache_insert(
            state_cache=torch.zeros(4, 8),
            token_to_req_indices=torch.zeros(0, dtype=torch.int32),
            positions=torch.zeros(0, dtype=torch.int32),
            compressor_slot_mapping=torch.zeros(0, dtype=torch.int32),
            block_table=torch.zeros(2, 4, dtype=torch.int32),
            compressor_block_size=64,
            rms_norm_weight=torch.ones(ops.DEEPSEEK_V4_HEAD_DIM),
            rms_norm_eps=1e-6,
            cos_sin_cache=torch.zeros(16, ops.DEEPSEEK_V4_ROPE_DIM),
            kv_cache_2d=torch.zeros(4, 8, dtype=torch.uint8),
            kv_slot_mapping=torch.zeros(0, dtype=torch.int32),
            kv_cache_block_size=64,
            compress_ratio=128,
            overlap=True,
        )
    mock.__getitem__.assert_not_called()
