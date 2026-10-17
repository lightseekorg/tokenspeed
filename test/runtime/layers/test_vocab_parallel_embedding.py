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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 embedding lookup requires CUDA"
)
def test_fp8_single_rank_lookup_clamps_out_of_range_ids() -> None:
    """A replicated FP8 table keeps the clamp-then-lookup path: in-range ids
    read their rows and an id past the table reads the last row, not OOB."""
    embedding = VocabParallelEmbedding(
        num_embeddings=128,
        embedding_dim=4,
        params_dtype=torch.float8_e4m3fn,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    ).cuda()
    embedding.weight.data.copy_(
        torch.arange(128 * 4, device="cuda", dtype=torch.float32)
        .reshape(128, 4)
        .remainder(32)
        .to(torch.float8_e4m3fn)
    )
    token_ids = torch.tensor([0, 63, 127, 4000], device="cuda")

    output = embedding(token_ids)

    assert output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        output.float(), embedding.weight[[0, 63, 127, 127]].float()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused gather requires CUDA")
def test_bf16_single_rank_lookup_maps_ids_like_a_shard() -> None:
    """A replicated table maps ids as a TP shard does: original-vocab rows in
    place, added rows after the original vocab's padding, and every other id
    (negative, padding, past the table) to a zero row."""
    embedding = VocabParallelEmbedding(
        num_embeddings=131,
        embedding_dim=8,
        params_dtype=torch.bfloat16,
        org_num_embeddings=130,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
        padding_size=64,
    ).cuda()
    # Rows 0..129 original vocab, 130..191 its padding, 192 the added id 130.
    assert embedding.weight.shape[0] == 256
    embedding.weight.data.copy_(
        torch.arange(256, device="cuda", dtype=torch.float32)
        .unsqueeze(1)
        .expand(256, 8)
        .to(torch.bfloat16)
    )
    token_ids = torch.tensor([-1, 0, 129, 130, 131, 191, 4000], device="cuda")

    output = embedding(token_ids)

    zero = torch.zeros(8, device="cuda", dtype=torch.bfloat16)
    expected = torch.stack(
        [zero, embedding.weight[0], embedding.weight[129], embedding.weight[192]]
        + [zero] * 3
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="embedding lookup on CUDA")
def test_single_rank_table_outside_the_gather_kernel_keeps_the_lookup() -> None:
    """A dtype the fused gather is not registered for (float64) keeps the
    clamp-then-lookup path."""
    embedding = VocabParallelEmbedding(
        num_embeddings=64,
        embedding_dim=4,
        params_dtype=torch.float64,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    ).cuda()
    embedding.weight.data.copy_(
        torch.randn(embedding.weight.shape, device="cuda", dtype=torch.float64)
    )
    token_ids = torch.tensor([0, 63, 4000], device="cuda")

    output = embedding(token_ids)

    torch.testing.assert_close(output, embedding.weight[[0, 63, 63]], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="embedding lookup on CUDA")
@pytest.mark.parametrize("tp_rank,tp_size", [(0, 1), (0, 2), (1, 2)])
def test_column_strided_table_keeps_the_lookup(tp_rank: int, tp_size: int) -> None:
    """A table with strided columns (the gather kernel reads unit-stride rows)
    keeps the lookup path, on one rank and on a TP shard."""
    embedding = VocabParallelEmbedding(
        num_embeddings=128,
        embedding_dim=4,
        params_dtype=torch.bfloat16,
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_group=tuple(range(tp_size)),
    ).cuda()
    backing = torch.randn(embedding.weight.shape[0], 8, device="cuda").to(
        torch.bfloat16
    )
    embedding.weight = torch.nn.Parameter(backing[:, ::2], requires_grad=False)
    local_ids = torch.tensor([0, 5, 63], device="cuda")
    token_ids = local_ids + tp_rank * 64

    output = embedding(token_ids, reduce_results=False)

    torch.testing.assert_close(output, backing[local_ids, ::2], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused gather requires CUDA")
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_bf16_tp_lookup_matches_masked_reference(tp_rank: int) -> None:
    """The one-kernel shard gather equals mask-then-lookup on every rank."""
    embedding = VocabParallelEmbedding(
        num_embeddings=130,
        embedding_dim=8,
        params_dtype=torch.bfloat16,
        tp_rank=tp_rank,
        tp_size=2,
        tp_group=(0, 1),
        padding_size=64,
    ).cuda()
    embedding.weight.data.copy_(
        torch.randn(embedding.weight.shape, device="cuda").to(torch.bfloat16)
    )
    token_ids = torch.tensor([0, 63, 64, 65, 127, 128, 129, 191, 192], device="cuda")
    assert embedding._fused_shard_gather(token_ids)

    output = embedding(token_ids, reduce_results=False)

    shard = embedding.shard_indices
    masked, mask = get_masked_input_and_mask(
        token_ids,
        shard.org_vocab_start_index,
        shard.org_vocab_end_index,
        shard.num_org_vocab_padding,
        shard.added_vocab_start_index,
        shard.added_vocab_end_index,
    )
    expected = torch.nn.functional.embedding(masked, embedding.weight)
    expected.masked_fill_(mask.unsqueeze(-1), 0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert (
        torch.count_nonzero(expected) > 0 and torch.count_nonzero(~mask) < mask.numel()
    )


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
