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

"""CPU checks for the logical weights consumed by the fused NZ ABI."""

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime.layers.attention.dsa.ascend import (
    AscendDSAIndexer,
    pack_down_qa_panels,
    pack_qb_panels,
    pack_score_fp32,
)


def test_compact_qb_preserves_heads_channels_and_k():
    weights = torch.arange(20480 * 1536, dtype=torch.int32).reshape(20480, 1536)
    packed = pack_qb_panels(weights)
    for head in (0, 7, 8, 31, 32, 63):
        for channel in (0, 15, 127, 128, 143, 191, 192, 207, 319):
            for k in (0, 15, 16, 511, 512, 1023, 1024, 1535):
                row = head * 320 + channel
                offset = ((row // 128 * 3 + k // 512) * 32 + k % 512 // 16) * 2048
                offset += row % 128 // 16 * 256 + row % 16 * 16 + k % 16
                assert (
                    packed[offset // 16 % 16, offset // 256 * 16 + offset % 16]
                    == weights[row, k]
                )


def test_compact_qb_rejects_missing_heads():
    with pytest.raises(ValueError):
        pack_qb_panels(torch.empty((2560, 1536)))


@pytest.mark.parametrize("dtype", [torch.int32, torch.bfloat16])
def test_down_pack_preserves_qa_kv_and_indexer_coefficients(dtype):
    # Unique int32 coefficients detect misplaced rows. Reinterpreted BF16
    # coefficients also cover signed zeros and NaN payloads without rounding.
    source = torch.arange(2240 * 4096, dtype=torch.int32).reshape(2240, 4096)
    if dtype == torch.bfloat16:
        source = source.to(torch.int16).view(torch.bfloat16)
    packed = pack_down_qa_panels(source)
    original_bits = source.view(torch.int16) if dtype == torch.bfloat16 else source
    packed_bits = packed.view(torch.int16) if dtype == torch.bfloat16 else packed
    rows = sorted(
        {0, 7, 8, 15, 16, 1535, 1536, 1537, 1551, 1552, 2111, 2112, 2239}
        | {r for panel in range(1, 16) for r in (panel * 96 - 1, panel * 96)}
    )
    channels = (0, 15, 16, 511, 512, 513, 1023, 1024, 2047, 2048, 4032, 4095)
    for row in rows:
        for k in channels:
            if row < 1536:
                # QA N96/K512 panels, with [K16, N16, n, k] inner tiles.
                raw = row // 96 * (96 * 4096) + k // 512 * (96 * 512)
                raw += (k % 512 // 16 * 6 + row % 96 // 16) * 256
                raw += row % 16 * 16 + k % 16
            else:
                # KV and indexer rows retain their genuine NZ suffix.
                auxiliary_row = row - 1536
                raw = (
                    1536 * 4096
                    + ((auxiliary_row // 16 * 256 + k // 16) * 16 + k % 16) * 16
                    + auxiliary_row % 16
                )
            carrier_row = raw // 16 % 4096
            carrier_column = raw // 65536 * 16 + raw % 16
            assert packed_bits[carrier_row, carrier_column] == original_bits[row, k]


def test_score_pack_preserves_fp32_coefficient_bits():
    bits = torch.arange(64 * 4096, dtype=torch.int32) + 0x3F000000
    weights = bits.view(torch.float32).reshape(64, 4096)
    packed = pack_score_fp32(weights)
    assert packed.dtype == torch.float32
    actual = packed.view(torch.int32).flatten()
    for head in (0, 7, 8, 31, 63):
        for channel in (0, 15, 16, 511, 512, 4095):
            # Native NZ_KN: head blocks of 8, channel blocks of 16,
            # with the head lane contiguous inside each channel lane.
            offset = ((head // 8 * 256 + channel // 16) * 16 + channel % 16) * 8
            offset += head % 8
            assert actual[offset] == bits[head * 4096 + channel]


def test_indexer_keeps_fp32_score_weights():
    config = PretrainedConfig(index_n_heads=2, index_head_dim=4, index_use_rope=False)
    indexer = AscendDSAIndexer(
        config=config,
        hidden_size=8,
        q_lora_rank=4,
    )
    assert indexer.weights_proj.weight.dtype == torch.float32
    assert not indexer.use_rotary
