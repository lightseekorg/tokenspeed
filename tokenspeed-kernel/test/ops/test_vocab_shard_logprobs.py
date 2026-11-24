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

"""Selected-token logprobs from vocabulary shards equal the whole rows'.

Each simulated rank selects its shard's top-k (writing its final logits and
the halo of leading columns its neighbor reads), the ranks "gather" by
stacking, and the logprob parts and their sum must give bit for bit
``selected_token_logprobs`` on the whole rows the logits processor would
produce: gathered, cut to the real vocabulary, capped, then masked.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import tanh_softcap
from tokenspeed_kernel.ops.sampling import (
    vocab_shard_logprob_blocks,
    vocab_shard_logprob_layout,
    vocab_shard_logprobs,
    vocab_shard_topk,
    vocab_topk_merge,
)
from tokenspeed_kernel.ops.sampling.triton.logprobs import selected_token_logprobs
from tokenspeed_kernel.ops.sampling.triton.vocab_logprobs import (
    _vocab_shard_logprob_blocks_kernel,
    _vocab_shard_logprobs_kernel,
)
from utils import assert_no_triton_compile

K = 128
CAP = 30.0

# (ranks, shard columns, real vocabulary): a TP4 head padded inside its last
# shard (blocks straddle every boundary), an unpadded one, and eight narrow
# shards whose last holds no block start (its 300 columns are the halo).
GEOMETRIES = [(4, 50256, 200058), (4, 26624, 4 * 26624), (8, 1500, 7 * 1500 + 300)]


class _Shards:
    """``ranks`` vocabulary shards of ``raw`` and their selection outputs."""

    def __init__(self, raw, ranks, columns, vocab, softcap, mask):
        self.raw, self.ranks, self.columns, self.vocab = raw, ranks, columns, vocab
        self.softcap, self.mask = softcap, mask
        self.halo, self.width = vocab_shard_logprob_layout(columns, vocab)
        rows = raw.shape[0]
        self.values = [
            torch.full((rows, vocab), float("nan"), dtype=raw.dtype, device=raw.device)
            for _ in range(ranks)
        ]
        staged = []
        for r in range(ranks):
            start = r * columns
            real = min(columns, vocab - start)
            out = torch.empty(rows, 2 * K + self.halo, device=raw.device)
            vocab_shard_topk(
                raw[:, start : start + real],
                K,
                None,
                start,
                out[:, : 2 * K],
                softcap=softcap,
                softcap_form=None if softcap is None else "tanh",
                vocab_mask=mask,
                values_out=self.values[r],
                halo_out=out[:, 2 * K :],
            )
            staged.append(out)
        gathered = torch.stack(staged)
        self.ids = torch.empty(rows, K, dtype=torch.int32, device=raw.device)
        self.top = torch.empty(rows, K, device=raw.device)
        vocab_topk_merge(gathered[:, :, : 2 * K], K, None, self.ids, self.top)
        for r in range(ranks - 1):
            nxt = (r + 1) * columns
            reach = min(nxt + self.halo, vocab)
            self.values[r][:, nxt:reach] = gathered[
                r + 1, :, 2 * K : 2 * K + reach - nxt
            ]

    def logprobs(self, rows, tokens):
        parts = []
        for r in range(self.ranks):
            part = torch.full(
                (rows.shape[0], self.width), float("nan"), device=self.raw.device
            )
            vocab_shard_logprob_blocks(
                self.values[r],
                rows,
                tokens,
                self.top[:, 0],
                r * self.columns,
                self.columns,
                part,
            )
            parts.append(part)
        out = torch.empty(rows.shape[0], device=self.raw.device)
        vocab_shard_logprobs(
            torch.stack(parts),
            rows,
            tokens,
            self.top[:, 0],
            self.columns,
            self.vocab,
            out,
        )
        return out


def _bitmask(allowed: torch.Tensor) -> torch.Tensor:
    rows, vocab = allowed.shape
    padded = torch.zeros(rows, -(-vocab // 32) * 32, dtype=torch.bool)
    padded[:, :vocab] = allowed
    bits = padded.view(rows, -1, 32).to(torch.int64) << torch.arange(32)
    bits = bits.sum(-1)
    return torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)


def _case(rows, ranks, columns, vocab, dtype, softcap, masked, seed, device):
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(rows, ranks * columns, generator=gen) * 4
    # A peak per row, so the row maximum is a single token.
    peaks = torch.randint(0, vocab, (rows,), generator=gen)
    raw[torch.arange(rows), peaks] += 40
    raw = raw.to(device, dtype)
    allowed = torch.ones(rows, vocab, dtype=torch.bool)
    mask = None
    if masked:
        allowed = torch.rand(rows, vocab, generator=gen) < 0.3
        allowed[torch.arange(rows), peaks] = True
        mask = _bitmask(allowed).to(device)
    # The logits processor's whole rows, then the sampler's in-place mask.
    whole = raw[:, :vocab]
    whole = tanh_softcap(whole, softcap) if softcap is not None else whole.clone()
    whole = whole.masked_fill(~allowed.to(device), float("-inf"))
    shards = _Shards(raw, ranks, columns, vocab, softcap, mask)
    # The sampled token (the peak) in most rows, other tokens in the rest.
    tokens = torch.randint(0, vocab, (rows,), generator=gen)
    tokens[::2] = peaks[::2]
    return whole, shards, tokens.to(device, torch.int32)


@pytest.mark.parametrize("ranks,columns,vocab", GEOMETRIES)
@pytest.mark.parametrize("rows", [8, 64])
@pytest.mark.parametrize(
    "dtype,softcap",
    [(torch.float32, CAP), (torch.float32, None), (torch.bfloat16, None)],
)
@pytest.mark.parametrize("masked", [False, True])
def test_shard_logprobs_equal_whole_row_logprobs(
    ranks, columns, vocab, rows, dtype, softcap, masked, device: str
) -> None:
    whole, shards, tokens = _case(
        rows, ranks, columns, vocab, dtype, softcap, masked, rows + ranks, device
    )
    # Each shard's final logits are the whole rows' columns, byte for byte.
    for r in range(ranks):
        start = r * columns
        stop = min(start + columns, vocab)
        assert torch.equal(shards.values[r][:, start:stop], whole[:, start:stop])
    assert torch.equal(shards.top[:, 0], whole.float().max(dim=1).values)

    out = shards.logprobs(torch.arange(rows, dtype=torch.int32, device=device), tokens)

    assert torch.equal(out, selected_token_logprobs(whole, tokens))


@pytest.mark.parametrize("ranks,columns,vocab", GEOMETRIES[:1])
def test_tree_rows_score_their_path_rows(ranks, columns, vocab, device: str) -> None:
    """Output rows read the row ``rows`` names, as an accepted tree path's
    predictions score the path's own rows."""
    whole, shards, tokens = _case(
        16, ranks, columns, vocab, torch.float32, CAP, True, 5, device
    )
    path = torch.tensor([0, 2, 5, 3, 4, 5, 6, 7, 8, 11, 10, 11, 12, 13, 14, 15])
    path = path.to(device)

    out = shards.logprobs(path, tokens)

    assert torch.equal(out, selected_token_logprobs(whole[path], tokens))


def test_row_counts_and_tokens_never_recompile(device: str) -> None:
    ranks, columns, vocab = GEOMETRIES[0]
    whole, shards, tokens = _case(
        64, ranks, columns, vocab, torch.float32, CAP, False, 9, device
    )
    for rows in (16, 7):
        shards.logprobs(torch.arange(rows, device=device), tokens[:rows])
    with assert_no_triton_compile(
        _vocab_shard_logprob_blocks_kernel, _vocab_shard_logprobs_kernel
    ):
        for rows in (1, 8, 33, 64):
            out = shards.logprobs(torch.arange(rows, device=device), tokens[:rows])
            assert torch.equal(
                out, selected_token_logprobs(whole[:rows], tokens[:rows])
            )


def test_far_rows_index_past_int32(device: str) -> None:
    """A row whose values start past 2**31 elements: only the far row is
    filled and scored."""
    vocab, columns = 200058, 100029
    sources = 2**31 // vocab + 2
    values = torch.empty(sources, vocab, dtype=torch.bfloat16, device=device)
    gen = torch.Generator().manual_seed(1)
    row = (torch.randn(1, vocab, generator=gen) * 4).to(device, torch.bfloat16)
    values[-1] = row[0]
    rows = torch.tensor([sources - 1], device=device)
    tokens = torch.tensor([123456], device=device)
    row_max = torch.zeros(sources, device=device)
    row_max[-1] = row.float().max()
    _, width = vocab_shard_logprob_layout(columns, vocab)
    parts = torch.empty(2, 1, width, device=device)
    for r in range(2):
        vocab_shard_logprob_blocks(
            values, rows, tokens, row_max, r * columns, columns, parts[r]
        )
    out = torch.empty(1, device=device)
    vocab_shard_logprobs(parts, rows, tokens, row_max, columns, vocab, out)
    assert torch.equal(out, selected_token_logprobs(row, tokens))
