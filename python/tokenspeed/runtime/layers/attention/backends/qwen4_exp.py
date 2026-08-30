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

"""Qwen4-Exp extensions for the hybrid GDN attention backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.qsa import bind_qsa_indexers
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    qwen4_exp_ple_conv_field,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer
    from tokenspeed.runtime.models.qwen4_exp_ple import Qwen4ExpPLELayer


def qwen4_exp_linear_backend(
    attn_backend: AttentionBackend,
) -> Qwen4ExpMambaAttnBackend:
    """Resolve and validate the Qwen4-Exp linear-attention backend."""

    backend = getattr(attn_backend, "linear_attn_backend", attn_backend)
    if not isinstance(backend, Qwen4ExpMambaAttnBackend):
        raise RuntimeError("Qwen4-Exp PLE requires its model-specific GDN backend")
    return backend


class Qwen4ExpMambaAttnBackend(MambaAttnBackend):
    """GDN backend with Qwen4-Exp PLE verification state."""

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        super().__init__(config, spec)
        self._ple_layers: tuple[Qwen4ExpPLELayer, ...] = ()
        self._ple_verify_scratch: dict[str, torch.Tensor] = {}

    def _preallocate_aux_verify_workspace(
        self, max_bs: int, draft_token_num: int
    ) -> int:
        self._ensure_ple_verify_scratch(max_bs, draft_token_num)
        return sum(tensor.nbytes for tensor in self._ple_verify_scratch.values())

    def _ensure_ple_verify_scratch(self, max_bs: int, draft_token_num: int) -> None:
        """Allocate graph-stable PLE context and convolution rollback rows."""
        arena = getattr(self.kv_pool, "arena", None)
        plan = getattr(arena, "plan", None)
        if plan is None:
            return
        fields = [
            field
            for field in plan.fields
            if field.group_id == QWEN4_EXP_PLE_CACHE_GROUP
        ]
        if not fields:
            return
        rows = max_bs * (draft_token_num + 1)
        if self._ple_verify_scratch and all(
            tensor.shape[0] >= rows for tensor in self._ple_verify_scratch.values()
        ):
            return
        scratch: dict[str, torch.Tensor] = {}
        for field in fields:
            cache_field = arena.field(field.field_id)
            scratch[field.field_id] = cache_field.new_zeros(
                (rows, *cache_field.shape[1:])
            )
        self._ple_verify_scratch = scratch

    def ple_verify_scratch(
        self, context_field_id: str, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return shared context and per-layer PLE convolution verify rows."""
        context = self._ple_verify_scratch.get(context_field_id)
        conv = self._ple_verify_scratch.get(qwen4_exp_ple_conv_field(layer_id))
        if context is None or conv is None:
            return None
        return context, conv

    def bind_ple_layers(self, layers: Iterable[Qwen4ExpPLELayer]) -> None:
        """Bind the stable model-owned PLE layers used by verify commits."""

        bound = tuple(layers)
        if self._ple_layers is bound:
            return
        if self._ple_layers:
            same = len(self._ple_layers) == len(bound) and all(
                previous is current
                for previous, current in zip(self._ple_layers, bound, strict=True)
            )
            if not same:
                raise RuntimeError("PLE backend cannot be rebound to another model")
        self._ple_layers = bound

    def _commit_aux_verified_state(
        self,
        accepted_length: torch.Tensor,
        pages_by_group: dict[str, torch.Tensor],
    ) -> None:
        pages = pages_by_group.get(QWEN4_EXP_PLE_CACHE_GROUP)
        if pages is None:
            return
        for layer in self._ple_layers:
            layer.commit_verified(accepted_length, pages)


def bind_qwen4_exp_side_state(
    attn_backend: AttentionBackend,
    ple_layers: Iterable[Qwen4ExpPLELayer],
    qsa_indexers: Iterable[QSAIndexer],
) -> None:
    """Bind Qwen4-Exp PLE and QSA state to their owning backends."""

    ple_layers = tuple(ple_layers)
    qsa_indexers = tuple(qsa_indexers)
    if ple_layers:
        qwen4_exp_linear_backend(attn_backend).bind_ple_layers(ple_layers)
    if qsa_indexers:
        bind_qsa_indexers(attn_backend, qsa_indexers)


__all__ = [
    "Qwen4ExpMambaAttnBackend",
    "bind_qwen4_exp_side_state",
    "qwen4_exp_linear_backend",
]
