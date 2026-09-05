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

"""QSA speculative side-state lifecycle.

QSA projections and cache algorithms live in ``attention.qsa.indexer`` because
they are model-owned modules. This backend owns the execution-side boundary:
target verification stages candidates during forward, then commits only the
accepted prefix after speculative sampling has produced per-request lengths.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer


class QSABackend:
    """Coordinate post-verification commits for one target model's indexers."""

    def __init__(self) -> None:
        self._indexers: tuple[QSAIndexer, ...] = ()

    def bind_indexers(self, indexers: Iterable[QSAIndexer]) -> None:
        """Bind the stable model-owned indexers used by verify commits.

        Args:
            indexers: QSA indexers belonging to the model attached to this
                attention backend.

        Returns:
            None.
        """

        bound = tuple(indexers)
        if self._indexers is bound:
            return
        if self._indexers:
            same = len(self._indexers) == len(bound) and all(
                previous is current
                for previous, current in zip(self._indexers, bound, strict=True)
            )
            if not same:
                raise RuntimeError("QSA backend cannot be rebound to another model")
        self._indexers = bound

    def commit_after_mtp_verify(
        self,
        accepted_lengths: torch.Tensor,
        *,
        num_extends: int,
    ) -> None:
        """Commit accepted target-verify candidates to every QSA indexer.

        Args:
            accepted_lengths: Per-request accepted lengths returned by the
                speculative sampler. Mixed batches place extend requests first.
            num_extends: Number of leading extend requests to exclude from the
                target-verify commit.

        Returns:
            None.
        """

        if num_extends < 0 or num_extends > accepted_lengths.shape[0]:
            raise ValueError(
                "QSA verify commit received an invalid extend prefix: "
                f"{num_extends} for {accepted_lengths.shape[0]} requests"
            )
        verify_lengths = accepted_lengths[num_extends:]
        if verify_lengths.numel() == 0:
            return
        if not self._indexers:
            raise RuntimeError("QSA verify commit ran before indexers were bound")
        for indexer in self._indexers:
            indexer.commit_verified(verify_lengths)


def bind_qsa_indexers(
    attn_backend: AttentionBackend,
    indexers: Iterable[QSAIndexer],
) -> QSABackend | None:
    """Attach QSA verify state to an attention backend and bind its indexers.

    The attachment is lazy so ordinary attention backend construction remains
    model-agnostic. CUDA-graph capture executes the model forward once, which
    establishes this stable binding before any graph replay can reach the
    post-verification hook.

    Args:
        attn_backend: Outermost attention backend used by the model forward.
        indexers: Model-owned QSA indexers to commit after target verification.

    Returns:
        The backend-local QSA lifecycle coordinator, or ``None`` for a draft
        backend because draft QSA state does not consume target acceptance.
    """

    full_backend = getattr(attn_backend, "full_attn_backend", attn_backend)
    if getattr(full_backend, "is_draft", False):
        return None
    qsa_backend = attn_backend.find_speculative_state_backend(QSABackend)
    if qsa_backend is None:
        qsa_backend = QSABackend()
        attn_backend.register_speculative_state_backend(qsa_backend)
    qsa_backend.bind_indexers(indexers)
    return qsa_backend


__all__ = ["QSABackend", "bind_qsa_indexers"]
