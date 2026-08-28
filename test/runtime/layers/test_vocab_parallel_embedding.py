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
import torch

from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 embedding lookup requires CUDA"
)
def test_fp8_tp_lookup_masks_remote_rows() -> None:
    embedding = VocabParallelEmbedding(
        num_embeddings=128,
        embedding_dim=4,
        params_dtype=torch.float8_e4m3fn,
        tp_rank=0,
        tp_size=2,
        tp_group=(0, 1),
    ).cuda()
    embedding.weight.data.copy_(
        torch.arange(64 * 4, device="cuda", dtype=torch.float32)
        .reshape(64, 4)
        .remainder(32)
        .to(torch.float8_e4m3fn)
    )
    token_ids = torch.tensor([0, 63, 64, 127], device="cuda")

    output = embedding(token_ids, reduce_results=False)

    assert output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(output[:2].float(), embedding.weight[[0, 63]].float())
    assert torch.count_nonzero(output[2:].float()) == 0


def test_masked_input_stays_inside_local_vocab_shard():
    input_ids = torch.tensor(
        [-1, 0, 75967, 75968, 151644, 151935, 151936, torch.iinfo(torch.int32).max]
    )

    masked_input, input_mask = get_masked_input_and_mask(
        input_ids,
        org_vocab_start_index=75968,
        org_vocab_end_index=151936,
        num_org_vocab_padding=0,
        added_vocab_start_index=151936,
        added_vocab_end_index=151936,
    )

    torch.testing.assert_close(
        masked_input,
        torch.tensor([0, 0, 0, 0, 75676, 75967, 0, 0]),
    )
    torch.testing.assert_close(
        input_mask,
        torch.tensor([True, True, True, False, False, False, True, True]),
    )
    assert masked_input.min() == 0
    assert masked_input.max() < 75968


def test_masked_input_handles_changing_token_counts():
    input_ids = torch.tensor(
        [
            151644,
            872,
            198,
            14880,
            110298,
            66017,
            82587,
            16,
            26939,
            20,
            3837,
            11622,
            107463,
            17992,
            17177,
            99859,
            1773,
            151645,
            198,
            151644,
            77091,
            198,
            151667,
            271,
            151668,
            271,
        ],
        dtype=torch.int32,
    )

    for token_count in (19, 26, 17, 26):
        current = input_ids[:token_count]
        masked_input, input_mask = get_masked_input_and_mask(
            current,
            org_vocab_start_index=0,
            org_vocab_end_index=75968,
            num_org_vocab_padding=0,
            added_vocab_start_index=151936,
            added_vocab_end_index=151936,
        )
        expected_mask = current >= 75968
        torch.testing.assert_close(
            masked_input,
            current.masked_fill(expected_mask, 0),
            check_dtype=False,
        )
        torch.testing.assert_close(input_mask, expected_mask)
