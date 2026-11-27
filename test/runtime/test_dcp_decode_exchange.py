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

from __future__ import annotations

import pytest

from tokenspeed.runtime.layers.attention.dcp import comm


def test_decode_float_exchange_cap_bytes_sizes_dynamically_up_to_16mib():
    # Kimi K3.1 KVP=8 (local_heads=12, head_dim=576, topk=2048):
    # FP8 KV -> 2048*4 + 12*576*1 = 15104 B/row; BF16 KV -> 22016 B/row.
    fp8_row = 2048 * 4 + 12 * 576 * 1
    bf16_row = 2048 * 4 + 12 * 576 * 2

    # Small batches shrink below the 8 MiB default and stay 16-byte aligned.
    small = comm.decode_float_exchange_cap_bytes(96, float_row_bytes=fp8_row)
    assert small == 96 * fp8_row
    assert small % 16 == 0
    assert small < comm.DECODE_FLOAT_EXCHANGE_CAP_BYTES

    # High-concurrency KVP=8 decode (bs=96..128, width=5..8) exceeds the legacy
    # 8 MiB cap and sizes cleanly up to 16 MiB without silent NCCL fallback.
    for rows in (96 * 6, 112 * 5, 128 * 5, 128 * 6, 128 * 8):
        cap = comm.decode_float_exchange_cap_bytes(rows, float_row_bytes=fp8_row)
        assert cap == rows * fp8_row
        assert cap > comm.DECODE_FLOAT_EXCHANGE_CAP_BYTES
        assert cap <= comm.DECODE_FLOAT_EXCHANGE_MAX_BYTES
        assert cap % 16 == 0

    # Payloads beyond 16 MiB clamp to DECODE_FLOAT_EXCHANGE_MAX_BYTES.
    clamped = comm.decode_float_exchange_cap_bytes(128 * 6, float_row_bytes=bf16_row)
    assert clamped == comm.DECODE_FLOAT_EXCHANGE_MAX_BYTES

    # Unaligned per-row payloads round up to a 16-byte boundary.
    assert comm.decode_float_exchange_cap_bytes(3, float_row_bytes=5) == 16

    # Omitting float_row_bytes preserves the legacy 8 MiB default.
    assert (
        comm.decode_float_exchange_cap_bytes(768, float_row_bytes=None)
        == comm.DECODE_FLOAT_EXCHANGE_CAP_BYTES
    )


@pytest.mark.parametrize(
    ("rows", "row_bytes"),
    [(0, 15104), (-1, 15104), (64, 0), (64, -16)],
)
def test_decode_float_exchange_cap_bytes_rejects_nonpositive_inputs(rows, row_bytes):
    with pytest.raises(ValueError, match="positive decode_rows and float_row_bytes"):
        comm.decode_float_exchange_cap_bytes(rows, float_row_bytes=row_bytes)


def test_prepare_decode_exchange_forwards_dynamic_float_cap(monkeypatch):
    calls: list[tuple[tuple[int, ...], int, int, int]] = []

    def fake_prepare_small_exchange(group, cap_bytes, float_cap_bytes, timeout_ns):
        calls.append((group, cap_bytes, float_cap_bytes, timeout_ns))
        return True

    monkeypatch.setattr(comm, "prepare_small_exchange", fake_prepare_small_exchange)
    monkeypatch.setattr(comm.torch.distributed, "get_rank", lambda: 0)

    fp8_row = 2048 * 4 + 12 * 576 * 1
    assert comm.prepare_decode_exchange(
        (0, 1, 2, 3, 4, 5, 6, 7),
        120.0,
        value_heads=None,
        decode_rows=128 * 6,
        float_words=True,
        float_row_bytes=fp8_row,
    )
    assert comm.prepare_decode_exchange(
        (0, 1, 2, 3, 4, 5, 6, 7),
        120.0,
        value_heads=None,
        decode_rows=128 * 6,
        float_words=False,
    )
    assert calls == [
        (
            (0, 1, 2, 3, 4, 5, 6, 7),
            comm.DECODE_EXCHANGE_CAP_BYTES,
            128 * 6 * fp8_row,
            int(90 * 1e9),
        ),
        (
            (0, 1, 2, 3, 4, 5, 6, 7),
            comm.DECODE_EXCHANGE_CAP_BYTES,
            0,
            int(90 * 1e9),
        ),
    ]

    with pytest.raises(ValueError, match="float_row_bytes requires float_words=True"):
        comm.prepare_decode_exchange(
            (0, 1),
            120.0,
            value_heads=None,
            decode_rows=64,
            float_words=False,
            float_row_bytes=fp8_row,
        )
