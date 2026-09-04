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

from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.state.qsa import bind_qsa_indexers


class _TestAttentionBackend:
    register_speculative_state_backend = (
        AttentionBackend.register_speculative_state_backend
    )
    commit_speculative_state_after_verify = (
        AttentionBackend.commit_speculative_state_after_verify
    )
    find_speculative_state_backend = AttentionBackend.find_speculative_state_backend

    def __init__(self, *, is_draft: bool) -> None:
        self.is_draft = is_draft


class _TestIndexer:
    def __init__(self) -> None:
        self.commits: list[torch.Tensor] = []

    def commit_verified(self, accepted_lengths: torch.Tensor) -> None:
        self.commits.append(accepted_lengths.clone())


def test_qsa_backend_commits_only_mixed_batch_verify_rows() -> None:
    attn_backend = _TestAttentionBackend(is_draft=False)
    indexers = [_TestIndexer(), _TestIndexer()]

    qsa_backend = bind_qsa_indexers(attn_backend, indexers)
    assert bind_qsa_indexers(attn_backend, indexers) is qsa_backend

    attn_backend.commit_speculative_state_after_verify(
        torch.tensor([9, 1, 3], dtype=torch.int32),
        num_extends=1,
    )

    for indexer in indexers:
        assert len(indexer.commits) == 1
        torch.testing.assert_close(
            indexer.commits[0], torch.tensor([1, 3], dtype=torch.int32)
        )


def test_qsa_backend_draft_never_commits_target_acceptance() -> None:
    attn_backend = _TestAttentionBackend(is_draft=True)
    indexer = _TestIndexer()

    qsa_backend = bind_qsa_indexers(attn_backend, [indexer])
    attn_backend.commit_speculative_state_after_verify(
        torch.tensor([4], dtype=torch.int32),
        num_extends=0,
    )

    assert qsa_backend is None
    assert not hasattr(attn_backend, "_speculative_state_backends")
    assert indexer.commits == []


def test_qsa_backend_rejects_rebinding_to_another_model() -> None:
    attn_backend = _TestAttentionBackend(is_draft=False)
    bind_qsa_indexers(attn_backend, [_TestIndexer()])

    with pytest.raises(RuntimeError, match="cannot be rebound"):
        bind_qsa_indexers(attn_backend, [_TestIndexer()])
