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

"""Batch-invariant DSA indexer top-k: a token's selection is its row's alone.

All-zero index keys make every candidate score exactly 0, the worst case for
tie resolution: the selection is then decided entirely by the tie-break, so a
kernel whose choice depends on the batch (row count, tiling, CTA split) shows
up immediately.
"""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
if not (torch.cuda.is_available() and platform.is_nvidia and platform.is_hopper_plus):
    pytest.skip("the DeepGEMM DSA top-k needs a Hopper+ GPU", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsa import (  # noqa: E402
    dsa_decode_topk,
    dsa_plan,
    dsa_prefill_topk,
)
from tokenspeed_kernel.ops.attention.dsa.flashinfer import (  # noqa: E402
    has_deterministic_decode_topk,
)

if not has_deterministic_decode_topk():
    pytest.skip("needs the lowest-index tie-break top-k", allow_module_level=True)

_HEADS = 32
_DIM = 128
_ROW_BYTES = _DIM + _DIM // 128 * 4
_PAGE = 64
_TOPK = 512
_INITIAL = 16
_LOCAL = 128


def _expected(length: int) -> set[int]:
    """Forced initial and local windows, then the lowest remaining columns."""
    forced = set(range(min(_INITIAL, length)))
    forced.update(range(max(0, length - _LOCAL), length))
    rest = [c for c in range(length) if c not in forced]
    return forced | set(rest[: max(0, min(_TOPK, length) - len(forced))])


def _prefill(lengths: list[int], max_logits_bytes: int) -> list[set[int]]:
    total = max(lengths)
    pages = -(-total // _PAGE)
    cache = torch.zeros(pages * _PAGE, _ROW_BYTES, dtype=torch.uint8, device="cuda")
    tokens = len(lengths)
    q = torch.randn(tokens, _HEADS, _DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, _HEADS, device="cuda")
    ends = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    rows, lens = dsa_prefill_topk(
        q,
        weights,
        torch.arange(total, dtype=torch.int64, device="cuda"),
        torch.zeros(tokens, dtype=torch.int32, device="cuda"),
        ends,
        topk=_TOPK,
        softmax_scale=1.0,
        batch_invariant=True,
        index_k_cache=cache,
        page_size=_PAGE,
        max_logits_bytes=max_logits_bytes,
        candidate_lens_cpu=torch.tensor(lengths, dtype=torch.int64),
        initial_tokens=_INITIAL,
        local_tokens=_LOCAL,
    )
    rows, lens = rows.cpu(), lens.cpu()
    return [
        {int(r) for r in rows[t].tolist() if r >= 0} for t in range(tokens)
    ], lens.tolist()


def test_prefill_selection_ignores_the_rest_of_the_batch() -> None:
    probe = [3000, 1700]
    solo, solo_lens = _prefill(probe, max_logits_bytes=1 << 30)
    # More rows and a small logits cap: the batch tiles differently.
    batched, _ = _prefill([900, 2500, *probe, 4000, 64], max_logits_bytes=64 * 4000)
    assert batched[2:4] == solo
    for selection, length, n in zip(solo, probe, solo_lens):
        assert n == min(length, _TOPK)
        assert selection == _expected(length)


def _decode(seq_lens: list[int]) -> list[set[int]]:
    reqs = len(seq_lens)
    max_pages = -(-max(seq_lens) // _PAGE)
    # Request r owns pages [r * max_pages, (r + 1) * max_pages).
    block_table = (
        torch.arange(reqs * max_pages, dtype=torch.int32, device="cuda")
        .view(reqs, max_pages)
        .contiguous()
    )
    cache = torch.zeros(
        reqs * max_pages * _PAGE, _ROW_BYTES, dtype=torch.uint8, device="cuda"
    )
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    seq_lens_2d = lens.unsqueeze(1).contiguous()
    slots, counts = dsa_decode_topk(
        torch.randn(reqs, _HEADS, _DIM, device="cuda", dtype=torch.bfloat16),
        torch.randn(reqs, _HEADS, device="cuda"),
        lens,
        block_table,
        page_size=_PAGE,
        topk=_TOPK,
        softmax_scale=1.0,
        batch_invariant=True,
        index_k_cache=cache,
        seq_lens_2d=seq_lens_2d,
        plan=dsa_plan(page_size=_PAGE, seq_lens_2d=seq_lens_2d),
        initial_tokens=_INITIAL,
        local_tokens=_LOCAL,
    )
    slots, counts = slots.cpu(), counts.cpu()
    base = [r * max_pages * _PAGE for r in range(reqs)]
    return [
        {int(s) - base[r] for s in slots[r].tolist() if s >= 0} for r in range(reqs)
    ]


def test_decode_selection_ignores_the_rest_of_the_batch() -> None:
    solo = _decode([3000])
    batched = _decode([700, 3000, *([2200] * 30)])
    assert batched[1] == solo[0] == _expected(3000)
