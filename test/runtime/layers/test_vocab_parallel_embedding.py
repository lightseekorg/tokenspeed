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
