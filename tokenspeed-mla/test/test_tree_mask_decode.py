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

import torch


def build_ancestor_mask(parent: list[int]) -> torch.Tensor:
    """ancestor[qi, j] is True iff new-token j is on qi's ancestry path."""

    q_len = len(parent)
    ancestor = torch.zeros(q_len, q_len, dtype=torch.bool)
    for qi in range(q_len):
        j = qi
        seen = 0
        while j != -1:
            ancestor[qi, j] = True
            j = parent[j]
            seen += 1
            assert seen <= q_len, f"cycle in parent array: {parent}"
    return ancestor


def chain_parent(q_len: int) -> list[int]:
    return [i - 1 for i in range(q_len)]


def build_custom_mask(
    ancestor: torch.Tensor, seq_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten per-request [q_len x K_b] masks and return int32 offsets.

    History columns are fully visible. The last q_len columns carry the
    tree-ancestor mask for the drafted tokens.
    """

    batch, q_len, q_len_2 = ancestor.shape
    assert q_len == q_len_2
    offsets = torch.empty(batch, dtype=torch.int32)
    flat_masks = []
    offset = 0
    for b in range(batch):
        offsets[b] = offset
        K = int(seq_lens[b])
        hist = K - q_len
        assert hist >= 0
        mask = torch.zeros(q_len, K, dtype=torch.bool)
        mask[:, :hist] = True
        mask[:, hist:K] = ancestor[b]
        flat_masks.append(mask.reshape(-1))
        offset += q_len * K
    return torch.cat(flat_masks).contiguous(), offsets


def test_chain_tree_mask_matches_causal_layout():
    for q_len in (1, 2, 4, 8, 16):
        ancestor = build_ancestor_mask(chain_parent(q_len))
        causal = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool))
        assert torch.equal(ancestor, causal)


def test_custom_mask_layout_and_offsets_for_variable_batch():
    ancestor = torch.stack(
        [
            build_ancestor_mask([-1, 0, 0, 2]),
            build_ancestor_mask([-1, 0, 1, 1]),
        ]
    )
    seq_lens = torch.tensor([10, 13], dtype=torch.int32)

    custom_mask, offsets = build_custom_mask(ancestor, seq_lens)

    assert offsets.tolist() == [0, 40]
    first = custom_mask[offsets[0] : offsets[1]].view(4, 10)
    second = custom_mask[offsets[1] :].view(4, 13)

    assert first[3].tolist() == [True] * 6 + [True, False, True, True]
    assert second[2].tolist() == [True] * 9 + [True, True, True, False]


def test_custom_mask_rejects_cycles_in_tree_parent():
    try:
        build_ancestor_mask([1, 0])
    except AssertionError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("expected cycle detection to fail")
