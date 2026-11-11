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

"""EAGLE-4 low-rank candidate head kernels (SM90+, thread-block clusters); see
``csrc/eagle4_candidate.cu``. Both reproduce the torch ops they replace bit for
bit. ``tokenspeed_kernel.ops.sampling.eagle4_candidate_supports`` says which
heads they cover, and a launch outside it raises."""

import functools
from pathlib import Path

import torch

__all__ = ["eagle4_candidate_tail", "eagle4_shard_topk"]


@functools.cache
def _module():
    import tvm_ffi

    so_path = (
        Path(__file__).parent / "objs" / "eagle4_candidate" / "eagle4_candidate.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel eagle4_candidate library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def eagle4_shard_topk(
    scores: torch.Tensor,
    k: int,
    vocab_start: int,
    global_ids: torch.Tensor,
) -> None:
    """``torch.topk(scores, k, dim=-1, sorted=False).indices`` of a vocab shard's
    ``[M, N]`` bf16 scores, same set and order, offset by ``vocab_start`` into
    ``global_ids`` (``[M, k]`` int64, unit column stride). One cluster of 8
    CTAs per row; covers N up to 262144."""
    _module().eagle4_shard_topk(scores, k, vocab_start, global_ids)


def eagle4_candidate_tail(
    cand_logits: torch.Tensor,
    cand_tokens: torch.Tensor,
    top_k: int,
    logits_out: torch.Tensor,
    tokens_out: torch.Tensor,
    topk_probs: torch.Tensor | None,
    topk_tokens: torch.Tensor | None,
) -> None:
    """Everything after the candidate all-gather, in one launch.

    The same, bit for bit, as::

        logits, pos = torch.topk(gathered_logits, NC, sorted=False)  # R > 1 only
        tokens = torch.gather(gathered_tokens, -1, pos)
        logits = logits.float()
        top = torch.topk(torch.softmax(logits, -1), top_k, -1)  # top_k > 0 only
        top_tokens = torch.gather(tokens, -1, top.indices)

    where ``gathered_*`` concatenates the ``R`` rank blocks of the ``[R, M, NC]``
    views ``cand_logits`` (bf16) and ``cand_tokens`` (int64), any rank
    and row strides. Writes the contiguous ``logits_out`` (``[M, NC]`` fp32),
    ``tokens_out`` (``[M, NC]`` int64; not for R == 1), and with ``top_k`` the
    probabilities and tokens into ``topk_probs`` / ``topk_tokens`` (``[M,
    top_k]``). Covers ``NC`` in 32, 64, 128, 256, ``R * NC <= 1024`` and
    ``top_k <= 32``.
    """
    if topk_probs is None:  # top_k == 0: the kernel does not touch them
        topk_probs, topk_tokens = logits_out, tokens_out
    _module().eagle4_candidate_tail(
        cand_logits, cand_tokens, top_k, logits_out, tokens_out, topk_probs, topk_tokens
    )
