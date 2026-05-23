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

"""NCCL fallback for Batch-DP logits shape swap."""

from __future__ import annotations

import torch

from tokenspeed.runtime.distributed.comm_backend import CommBackend, Group
from tokenspeed.runtime.distributed.comm_ops import all_to_all_single


def swap_batch_vocab(
    local_logits: torch.Tensor,
    *,
    tp_size: int,
    pad_bs: int,
    num_tokens_per_req: int,
    vocab_size: int,
    rank: int,
    group: Group,
    backend: CommBackend | None = None,
) -> torch.Tensor:
    """Move logits from vocab shards to request shards.

    Each rank starts with local_logits[pad_bs * N, V_local] for the full
    padded batch and its local vocab slice, where V_local=V/TP. The result is
    [reqs_per_rank * N, V] for this rank's reqs_per_rank=pad_bs/TP requests.
    Returned row local_req * N + d is global request
    rank * reqs_per_rank + local_req at draft position d.
    """
    assert (
        pad_bs % tp_size == 0
    ), f"swap_batch_vocab: pad_bs={pad_bs} must be divisible by tp_size={tp_size}"
    assert (
        vocab_size % tp_size == 0
    ), f"swap_batch_vocab: vocab_size={vocab_size} must be divisible by tp_size={tp_size}"

    reqs_per_rank = pad_bs // tp_size
    v_local = vocab_size // tp_size
    n = num_tokens_per_req

    expected_shape = (pad_bs * n, v_local)
    assert tuple(local_logits.shape) == expected_shape, (
        f"swap_batch_vocab: local_logits shape {tuple(local_logits.shape)} "
        f"!= expected {expected_shape} (pad_bs={pad_bs}, N={n}, V/TP={v_local})"
    )

    recv = torch.empty_like(local_logits)
    all_to_all_single(recv, local_logits, rank, group, backend=backend)

    return (
        recv.view(tp_size, reqs_per_rank, n, v_local)
        .permute(1, 2, 0, 3)
        .contiguous()
        .view(reqs_per_rank * n, vocab_size)
    )
