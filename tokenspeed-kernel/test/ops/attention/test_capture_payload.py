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

"""Record production preserves exact payload values and unwritten storage."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda import (
    KdaReplayRecords,
    kda_fused_paged_verify_uses_split_producers,
    kda_verify_conv_update,
    try_kda_fused_paged_verify,
)

DIM = 128


def _stacks(layers, capacity, pad=0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    sources = [
        torch.randn(capacity + 3, DIM, device="cuda", dtype=torch.bfloat16, generator=g)
        for _ in range(3)
    ]
    key_gate = torch.randn(
        layers,
        capacity,
        1,
        1,
        2 * DIM + pad,
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )
    correction = torch.randn(
        layers,
        capacity,
        1,
        1,
        DIM + pad,
        device="cuda",
        dtype=torch.float32,
        generator=g,
    )
    return sources, key_gate, correction


def _capture(sources, key_gate, correction, rows):
    """T=1 and a zero checkpoint give U = sigmoid(0) * V exactly."""
    key, gate, value = (s[:rows].view(1, -1, 1, DIM) for s in sources)
    metadata = torch.zeros(rows, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(rows, device="cuda")
    stamps = torch.full(
        (correction.shape[0], 1, 2), -1, device="cuda", dtype=torch.int64
    )
    # Preserve source row/channel strides while packing the common producer input.
    tokens = key.shape[1]
    channel_stride = key.stride(-1)
    row_stride = max(3 * DIM * channel_stride, key.stride(1))
    conv_qkv = torch.empty_strided(
        (tokens, 3 * DIM),
        (row_stride, channel_stride),
        device="cuda",
        dtype=torch.bfloat16,
    )
    conv_qkv[:, :DIM].zero_()
    conv_qkv[:, DIM : 2 * DIM].copy_(key.view(tokens, DIM))
    conv_qkv[:, 2 * DIM :].copy_(value.view(tokens, DIM))
    reads = torch.ones(rows, device="cuda", dtype=torch.int32)
    records = KdaReplayRecords(
        original_qkv=None,
        correction=correction,
        key_gate=key_gate,
        metadata=metadata,
        stamps=stamps,
        query_start_loc=torch.arange(rows + 1, device="cuda", dtype=torch.int32),
    )
    conv_weights = torch.zeros(3 * DIM, 4, device="cuda", dtype=torch.bfloat16)
    conv_weights[:, -1] = 1
    conv_pool = torch.zeros(2, 3 * DIM, 3, device="cuda", dtype=torch.bfloat16)
    producers = {}
    if kda_fused_paged_verify_uses_split_producers(
        torch.bfloat16,
        store_states=False,
        recurrent_layout="v_major",
        num_heads=1,
        head_dim=DIM,
    ):
        producers["conv_qkv"] = kda_verify_conv_update(
            conv_qkv,
            conv_weights,
            conv_pool,
            reads,
            num_heads=1,
            head_dim=DIM,
            draft_token_num=1,
            recurrent_layout="v_major",
            bias=None,
        )
    output = try_kda_fused_paged_verify(
        conv_qkv,
        conv_weights,
        conv_pool,
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        torch.zeros(rows, 1, device="cuda", dtype=torch.bfloat16),
        torch.zeros(1, device="cuda"),
        torch.zeros(DIM, device="cuda"),
        state_pool=torch.zeros(2, 1, DIM, DIM, device="cuda"),
        state_scratch=None,
        read_indices=reads,
        write_indices=reads,
        num_heads=1,
        head_dim=DIM,
        draft_token_num=1,
        lower_bound=-5.0,
        recurrent_layout="v_major",
        store_states=False,
        replay_records=records,
        g_raw=gate.view(-1, DIM),
        **producers,
    )
    assert output is not None


def _activated(source):
    # Only the current-input convolution tap is nonzero.
    return torch.nn.functional.silu(source.float()).to(source.dtype)


@pytest.mark.parametrize("rows", [1, 8, 128, 1024])
@pytest.mark.parametrize("pad", [0, 16])
def test_matches_three_copies(rows, pad):
    """K, raw gate and the independent U oracle land in the selected layer."""
    src, kg, u = _stacks(4, rows, pad=pad, seed=rows)
    expected_kg, expected_u = kg.clone(), u.clone()
    expected_kg[2, :rows, 0, 0, :DIM].copy_(_activated(src[0][:rows]))
    expected_kg[2, :rows, 0, 0, DIM : 2 * DIM].copy_(src[1][:rows])
    expected_u[2, :rows, 0, 0, :DIM].copy_(_activated(src[2][:rows]).float() * 0.5)
    _capture(src, kg[2, ..., : 2 * DIM], u[2, ..., :DIM], rows)
    torch.testing.assert_close(kg, expected_kg, atol=0, rtol=0)
    torch.testing.assert_close(u, expected_u, atol=0, rtol=0)


def test_nothing_outside_the_written_rows_moves():
    """Unused slots, padding columns and other layers remain unchanged."""
    rows, capacity = 8, 24
    src, kg, u = _stacks(4, capacity, pad=16, seed=5)
    before = (kg.clone(), u.clone())
    _capture(src, kg[1, ..., : 2 * DIM], u[1, ..., :DIM], rows)
    for got, old, width in zip((kg, u), before, (2 * DIM, DIM), strict=True):
        for layer in (0, 2, 3):
            torch.testing.assert_close(got[layer], old[layer], atol=0, rtol=0)
        torch.testing.assert_close(got[1, rows:], old[1, rows:], atol=0, rtol=0)
        torch.testing.assert_close(
            got[1, ..., width:], old[1, ..., width:], atol=0, rtol=0
        )


def test_a_source_row_inside_a_wider_projection():
    """Producer records follow strided projection views without copying padding."""
    rows = 8
    wide = torch.randn(rows, 4 * DIM, device="cuda", dtype=torch.bfloat16)
    src = (wide[:, :DIM], wide[:, DIM : 2 * DIM], wide[:, 2 * DIM : 3 * DIM])
    _, kg, u = _stacks(1, rows)
    _capture(src, kg[0], u[0], rows)
    torch.testing.assert_close(kg[0, :, 0, 0, :DIM], _activated(src[0]), atol=0, rtol=0)
    torch.testing.assert_close(kg[0, :, 0, 0, DIM:], src[1], atol=0, rtol=0)
    torch.testing.assert_close(
        u[0, :, 0, 0], _activated(src[2]).float() * 0.5, atol=0, rtol=0
    )


def test_layouts_the_kernel_cannot_address_are_rejected():
    """Preserve the existing invalid-layout coverage at the records API."""
    rows = 8
    src, kg, u = _stacks(1, rows)
    wide = torch.randn(rows, 2 * DIM, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous|last stride 1"):
        _capture((wide[:, ::2], src[1], src[2]), kg[0], u[0], rows)
    with pytest.raises(ValueError, match="correction buffer"):
        _capture(src, kg[0], u[0, ..., :-1], rows)
    with pytest.raises(ValueError):
        _capture(tuple(s[: rows - 1] for s in src), kg[0], u[0], rows)
