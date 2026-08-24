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

"""Launch-config tests for the fused sparse compress cache insert."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton import dsv4 as ops


@pytest.fixture(autouse=True)
def _clear_wide_launch_capability_cache():
    ops._wide_compress_launch_supported.cache_clear()
    yield
    ops._wide_compress_launch_supported.cache_clear()


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
        patch.object(ops, "_dsv4_fused_sparse_compress_cache_kernel", _Grid()),
        patch.object(
            ops, "_wide_compress_launch_supported", return_value=wide_supported
        ),
    ):
        ops.dsv4_fused_sparse_compress_cache_insert(
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
    assert _launch(128)["num_warps"] == 16


def test_sparse_compress_keeps_narrow_launch_on_unsupported_target():
    assert _launch(128, wide_supported=False)["num_warps"] == 4


def test_sparse_compress_keeps_narrow_launch_for_small_ratio():
    assert _launch(4)["num_warps"] == 4


def test_sparse_compress_grid_is_one_program_per_token():
    assert _launch(128)["grid"] == (8,)


def test_sparse_compress_skips_empty_batch():
    mock = MagicMock()
    with patch.object(ops, "_dsv4_fused_sparse_compress_cache_kernel", mock):
        ops.dsv4_fused_sparse_compress_cache_insert(
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


def test_wide_launch_supports_sm100_and_caches_per_device(caplog):
    get_capability = MagicMock(return_value=(10, 0))
    with (
        patch.object(ops.torch.cuda, "is_available", return_value=True),
        patch.object(ops.torch.version, "hip", None),
        patch.object(ops.torch.cuda, "get_device_capability", get_capability),
        caplog.at_level("INFO", logger=ops.__name__),
    ):
        assert ops._wide_compress_launch_supported(0)
        assert ops._wide_compress_launch_supported(0)

    get_capability.assert_called_once_with(0)
    assert caplog.text.count("num_warps=16") == 1


def test_wide_launch_rejects_non_sm100():
    with (
        patch.object(ops.torch.cuda, "is_available", return_value=True),
        patch.object(ops.torch.version, "hip", None),
        patch.object(
            ops.torch.cuda, "get_device_capability", return_value=(9, 0)
        ) as get_capability,
    ):
        assert not ops._wide_compress_launch_supported(0)

    get_capability.assert_called_once_with(0)


@pytest.mark.parametrize(
    ("cuda_available", "hip_version"),
    [(False, None), (True, "6.4")],
)
def test_wide_launch_rejects_unavailable_or_hip_platform(cuda_available, hip_version):
    with (
        patch.object(ops.torch.cuda, "is_available", return_value=cuda_available),
        patch.object(ops.torch.version, "hip", hip_version),
        patch.object(ops.torch.cuda, "get_device_capability") as get_capability,
    ):
        assert not ops._wide_compress_launch_supported(0)

    get_capability.assert_not_called()


@pytest.mark.parametrize(
    "error_type", [AssertionError, RuntimeError, TypeError, ValueError]
)
def test_wide_launch_fails_closed_when_capability_query_raises(error_type):
    with (
        patch.object(ops.torch.cuda, "is_available", return_value=True),
        patch.object(ops.torch.version, "hip", None),
        patch.object(
            ops.torch.cuda,
            "get_device_capability",
            side_effect=error_type("capability unavailable"),
        ),
    ):
        assert not ops._wide_compress_launch_supported(0)
