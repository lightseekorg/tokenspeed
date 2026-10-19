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
from tokenspeed_kernel.ops.sampling import argmax
from tokenspeed_kernel.ops.sampling.triton.vocab_parallel_argmax import (
    _argmax_key_index_kernel,
    _pack_argmax_key_kernel,
    argmax_key_index,
    vocab_parallel_argmax_keys,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _sharded_argmax(logits: torch.Tensor, num_shards: int) -> torch.Tensor:
    """Run each shard's key packing, reduce with MAX as the all-reduce would,
    and decode the global index."""
    shard_width = logits.shape[1] // num_shards
    keys = []
    for shard in range(num_shards):
        out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        columns = logits[:, shard * shard_width : (shard + 1) * shard_width]
        vocab_parallel_argmax_keys(
            columns.contiguous(), vocab_offset=shard * shard_width, out=out
        )
        keys.append(out)
    reduced = torch.stack(keys).amax(dim=0)
    return argmax_key_index(reduced, out=torch.empty_like(reduced))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,shard_width", [(1, 1000), (16, 124160), (64, 4096)])
def test_matches_argmax_of_the_gathered_row(dtype, rows, shard_width) -> None:
    torch.manual_seed(0)
    logits = torch.randn(rows, 2 * shard_width, device="cuda").to(dtype)

    actual = _sharded_argmax(logits, num_shards=2)

    torch.testing.assert_close(actual, logits.argmax(dim=-1), rtol=0, atol=0)


def test_ties_and_signs_resolve_like_argmax() -> None:
    shard_width = 512
    logits = torch.full((4, 4 * shard_width), -3.0, device="cuda")
    # A tie across shards goes to the lower global index.
    logits[0, 3 * shard_width + 1] = 7.0
    logits[0, shard_width + 5] = 7.0
    # All-negative rows compare by value, not by raw bits.
    logits[1, 2 * shard_width] = -0.5
    logits[1, 7] = -0.75
    # Signed zeros tie.
    logits[2] = -1.0
    logits[2, 2 * shard_width + 3] = 0.0
    logits[2, shard_width + 9] = -0.0
    # A shard maximum in the last column of the last shard.
    logits[3, -1] = 1.0

    actual = _sharded_argmax(logits, num_shards=4)

    expected = [shard_width + 5, 2 * shard_width, shard_width + 9, 4 * shard_width - 1]
    assert actual.tolist() == expected


def test_row_count_does_not_recompile() -> None:
    logits = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    keys = torch.empty(64, dtype=torch.int64, device="cuda")
    index = torch.empty(64, dtype=torch.int32, device="cuda")
    vocab_parallel_argmax_keys(logits[:1], vocab_offset=2048, out=keys[:1])
    argmax_key_index(keys[:1], out=index[:1])
    with assert_no_triton_compile(_pack_argmax_key_kernel, _argmax_key_index_kernel):
        for rows in (3, 16, 17, 64):
            vocab_parallel_argmax_keys(
                logits[:rows], vocab_offset=2048, out=keys[:rows]
            )
            argmax_key_index(keys[:rows], out=index[:rows])


def test_nan_rows_follow_the_argmax_of_the_gathered_row() -> None:
    shard_width = 512
    logits = torch.randn(3, 2 * shard_width, device="cuda", dtype=torch.bfloat16)
    # NaN on the second shard only, after the first shard's maximum.
    logits[0, shard_width + 7] = float("nan")
    # Negative NaN on the first shard only.
    logits[1, 5] = -float("nan")
    # NaN everywhere.
    logits[2] = float("nan")

    actual = _sharded_argmax(logits, num_shards=2)

    torch.testing.assert_close(actual.long(), argmax(logits).long(), rtol=0, atol=0)
