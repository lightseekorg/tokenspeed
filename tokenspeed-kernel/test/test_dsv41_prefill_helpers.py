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

"""CPU behavioral tests for DSv4.1 prefill vectorization and MXFP8 warmup helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm, flash_mla
from tokenspeed_kernel.ops.gemm import flashinfer as flashinfer_gemm
from tokenspeed_kernel.ops.tuning import (
    get_autotune_max_num_tokens,
    kernel_warmup,
    set_autotune_max_num_tokens,
)


@pytest.mark.parametrize("history_rows", [(4,), (2, 5, 1), (1, 1, 3, 2)])
def test_pack_index_keys_matches_per_run_reference(monkeypatch, history_rows):
    """Vectorized _pack_index_keys produces identical pages and starts."""
    gathered = []

    def fake_gather(cache, pages, page_size):
        gathered.append((pages.clone(), page_size))
        return ("keys", "scales")

    monkeypatch.setattr(deep_gemm, "gather_index_pages", fake_gather)
    total_rows = sum(history_rows)
    width = 3
    page_table = torch.empty((total_rows, width), dtype=torch.int32)
    expected_pages = []
    expected_starts = []
    cursor = 0
    for run, count in enumerate(history_rows):
        row = torch.arange(
            1 + run * width, 1 + (run + 1) * width, dtype=torch.int32
        )
        page_table[cursor : cursor + count] = row
        expected_pages.append(row)
        expected_starts.extend([run * width * 64] * count)
        cursor += count

    result, pages, starts = deep_gemm._pack_index_keys(
        torch.empty(1, dtype=torch.uint8), page_table, history_rows
    )
    assert result == ("keys", "scales")
    assert torch.equal(pages, torch.cat(expected_pages))
    assert starts.dtype == torch.int32
    assert starts.tolist() == expected_starts
    assert len(gathered) == 1 and torch.equal(gathered[0][0], pages)


def test_flashmla_selected_attention_fp8_positions_nonnegative(monkeypatch):
    """positions_nonnegative=True validates contiguous int32 and skips clamping."""
    calls = []

    def fake_prefill(**kwargs):
        calls.append(kwargs)
        t = kwargs["q"].shape[0]
        g = kwargs["n_wv_group"]
        w = kwargs["q"].shape[1] * 512 // g
        return (
            torch.zeros((t, g, w), dtype=torch.float8_e4m3fn),
            torch.zeros((t, g, w // 128), dtype=torch.int32),
            None,
            None,
        )

    monkeypatch.setattr(
        flash_mla, "_FUSED_FLASH_MLA_API", SimpleNamespace(prefill=fake_prefill)
    )
    q = torch.zeros((4, 64, 512), dtype=torch.bfloat16)
    rope = torch.zeros((16, 64), dtype=torch.float32)
    sink = torch.zeros(64, dtype=torch.float32)
    kv = torch.zeros((8, 1, 512), dtype=torch.bfloat16)
    indices = torch.zeros((4, 64), dtype=torch.int32)

    # positions_nonnegative=True forwards the exact contiguous int32 tensor.
    pos_i32 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    flash_mla.flashmla_selected_attention_fp8(
        q,
        None,
        None,
        None,
        None,
        None,
        None,
        sink,
        512**-0.5,
        pos_i32,
        rope,
        8,
        kv,
        indices,
        positions_nonnegative=True,
    )
    assert calls[-1]["token_positions"] is pos_i32
    assert calls[-1]["indices"].dtype == torch.int32

    # positions_nonnegative=True rejects non-int32 and non-contiguous inputs.
    with pytest.raises(ValueError, match="contiguous int32"):
        flash_mla.flashmla_selected_attention_fp8(
            q,
            None,
            None,
            None,
            None,
            None,
            None,
            sink,
            512**-0.5,
            pos_i32.to(torch.int64),
            rope,
            8,
            kv,
            indices,
            positions_nonnegative=True,
        )
    non_contig = torch.arange(8, dtype=torch.int32)[::2]
    assert not non_contig.is_contiguous()
    with pytest.raises(ValueError, match="contiguous int32"):
        flash_mla.flashmla_selected_attention_fp8(
            q,
            None,
            None,
            None,
            None,
            None,
            None,
            sink,
            512**-0.5,
            non_contig,
            rope,
            8,
            kv,
            indices,
            positions_nonnegative=True,
        )

    # positions_nonnegative=False clamps negative padding positions to 0.
    pos_neg = torch.tensor([-5, 0, -1, 7], dtype=torch.int64)
    flash_mla.flashmla_selected_attention_fp8(
        q,
        None,
        None,
        None,
        None,
        None,
        None,
        sink,
        512**-0.5,
        pos_neg,
        rope,
        8,
        kv,
        indices,
        positions_nonnegative=False,
    )
    assert calls[-1]["token_positions"].dtype == torch.int32
    assert calls[-1]["token_positions"].tolist() == [0, 0, 0, 7]


def test_warm_mxfp8_buckets_warms_once_inside_kernel_warmup(monkeypatch):
    """_warm_mxfp8_buckets compiles every servable M bucket once during warmup
    with autotune(tune_mode=False) and skips outside warmup."""
    calls: list[tuple[int, int, int, int]] = []
    autotune_modes: list[bool] = []

    def fake_mm_mxfp8(a, b, a_scales, b_scales, *, out_dtype, backend):
        assert backend == "cute-dsl"
        calls.append((a.shape[0], a_scales.numel(), a.data_ptr(), a_scales.data_ptr()))
        return torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype)

    from contextlib import contextmanager

    @contextmanager
    def record_autotune(*, tune_mode, tuning_buckets, round_up):
        autotune_modes.append(tune_mode)
        yield

    monkeypatch.setattr(flashinfer_gemm, "mm_mxfp8", fake_mm_mxfp8)
    monkeypatch.setattr(
        flashinfer_gemm,
        "mxfp8_warmup_buckets",
        lambda max_m: tuple(m for m in (1, 2, 4, 8, 128, 256, 384, 512) if m <= max_m),
    )
    monkeypatch.setattr(flashinfer_gemm, "autotune", record_autotune)
    monkeypatch.setattr(flashinfer_gemm, "_WARMED_MXFP8_SHAPES", set())

    prev_max = get_autotune_max_num_tokens()
    set_autotune_max_num_tokens(512)
    try:
        k, n = 256, 128
        k_tiles = (k + 127) // 128
        b = torch.zeros((k, n), dtype=torch.float8_e4m3fn)
        b_scales = torch.full(((n + 127) // 128 * 128 * k_tiles * 4,), 127, dtype=torch.uint8)

        # Outside kernel_warmup: complete no-op.
        a_small = torch.zeros((128, k), dtype=torch.float8_e4m3fn)
        s_small = torch.full((128 * k_tiles * 4,), 127, dtype=torch.uint8)
        flashinfer_gemm._warm_mxfp8_buckets(a_small, b, s_small, b_scales, torch.bfloat16)
        assert calls == []

        # Inside kernel_warmup with A.shape[0] < max_m: warms all buckets != 128,
        # slicing A directly for <= 128 and temporary storage for > 128.
        with kernel_warmup(enabled=True):
            flashinfer_gemm._warm_mxfp8_buckets(
                a_small, b, s_small, b_scales, torch.bfloat16
            )
            # Repeat call for the same (device, n, k, dtype, out_dtype): no-op.
            flashinfer_gemm._warm_mxfp8_buckets(
                a_small, b, s_small, b_scales, torch.bfloat16
            )
        assert autotune_modes == [False]
        assert [c[0] for c in calls] == [1, 2, 4, 8, 256, 384, 512]
        # Buckets <= 128 slice A and A_scales without reallocating.
        for m, scale_len, a_ptr, s_ptr in calls:
            expected_scales = ((m + 127) // 128) * 128 * k_tiles * 4
            assert scale_len == expected_scales
            if m <= 128:
                assert a_ptr == a_small.data_ptr()
                assert s_ptr == s_small.data_ptr()
            else:
                assert a_ptr != a_small.data_ptr()
                assert s_ptr != s_small.data_ptr()

        # When A.shape[0] already covers max_m, every bucket slices A directly.
        calls.clear()
        autotune_modes.clear()
        b_other = torch.zeros((k, 256), dtype=torch.float8_e4m3fn)
        b_other_scales = torch.full(
            (((256 + 127) // 128) * 128 * k_tiles * 4,), 127, dtype=torch.uint8
        )
        a_large = torch.zeros((512, k), dtype=torch.float8_e4m3fn)
        s_large = torch.full((4 * 128 * k_tiles * 4,), 127, dtype=torch.uint8)
        with kernel_warmup(enabled=True):
            flashinfer_gemm._warm_mxfp8_buckets(
                a_large, b_other, s_large, b_other_scales, torch.bfloat16
            )
        assert [c[0] for c in calls] == [1, 2, 4, 8, 128, 256, 384]
        assert all(
            c[2] == a_large.data_ptr() and c[3] == s_large.data_ptr() for c in calls
        )
    finally:
        set_autotune_max_num_tokens(prev_max)
