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

"""Qwen4-Exp composition of attention, PLE and QSA cache consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.backends.specific.qsa_indexer import (
        QSAIndexerBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
        Qwen4ExpPLEBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        MambaAttnBackend,
    )


def qwen4_exp_backend(attn_backend: AttentionBackend) -> Qwen4ExpBackend:
    """Resolve the model's composite without depending on an attention leaf."""
    if not isinstance(attn_backend, Qwen4ExpBackend):
        raise RuntimeError("Qwen4-Exp layers require their model's composite backend")
    return attn_backend


class Qwen4ExpBackend(HybridLinearAttnBackend):
    """Broadcast cache lifecycle calls; model layers retain their compute order."""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackend | None,
        full_attn_layers: list[int],
        ple_backend: Qwen4ExpPLEBackend | None,
        indexer_backend: QSAIndexerBackend | None,
    ) -> None:
        super().__init__(full_attn_backend, linear_attn_backend, full_attn_layers)
        self.ple_backend = ple_backend
        self.indexer_backend = indexer_backend

    def child_backends(self) -> tuple[AttentionBackend, ...]:
        return super().child_backends() + tuple(
            backend
            for backend in (self.ple_backend, self.indexer_backend)
            if backend is not None
        )

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        self.full_attn_backend.advance_draft_forward_metadata(seq_lens)
        if self.indexer_backend is not None:
            self.indexer_backend.advance_draft_forward_metadata(seq_lens)

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        self.full_attn_backend.update_draft_forward_metadata(frontier)
        if self.indexer_backend is not None:
            self.indexer_backend.update_draft_forward_metadata(frontier)

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        self.full_attn_backend.fill_block_decode_seq_lens(bs, block_seq_lens)
        if self.indexer_backend is not None:
            self.indexer_backend.fill_block_decode_seq_lens(bs, block_seq_lens)

    def update_mamba_state_after_mtp_verify(
        self, accepted_lengths: torch.Tensor
    ) -> None:
        super().update_mamba_state_after_mtp_verify(accepted_lengths)
        if self.ple_backend is not None:
            self.ple_backend.commit_verified_state(accepted_lengths)

    def commit_speculative_state_after_verify(
        self, accepted_lengths: torch.Tensor, *, num_extends: int
    ) -> None:
        super().commit_speculative_state_after_verify(
            accepted_lengths, num_extends=num_extends
        )
        if self.indexer_backend is not None:
            self.indexer_backend.commit_after_mtp_verify(
                accepted_lengths, num_extends=num_extends
            )


__all__ = ["Qwen4ExpBackend", "qwen4_exp_backend"]
