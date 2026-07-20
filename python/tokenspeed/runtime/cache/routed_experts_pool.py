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

"""Rollout Routing Replay (R3): per-KV-slot storage of MoE routing decisions.

R3 (`arXiv:2510.11370 <https://arxiv.org/abs/2510.11370>`_) stabilizes RL
training of Mixture-of-Experts models by making the training forward *replay*
the exact expert selection the rollout/inference engine made, instead of
letting the two routers pick experts independently (numerical differences flip
routing decisions for tokens and inflate the train/inference KL). To replay,
the trainer needs the per-token, per-layer top-k expert ids the rollout used.

**Why store routing in a slot-indexed pool (the key design choice).**
A transient per-forward capturer (as in vllm-ascend's ``RoutedExpertsCapturer``)
only sees routing for tokens actually forwarded in that step. But on a
prefix-cache hit the shared prefix tokens are *dropped from the forward
entirely* — their KV is reused and their MoE layers never run — so a transient
capturer returns nothing for the prefix and you would have to force a re-forward
to recover it. Storing routing in a pool **indexed by the same KV slot as the
KV cache** makes routing follow the KV: a prefix hit that reuses slots also
reuses the routing captured when those slots were first written, with zero
recompute and guaranteed consistency with the reused KV.

This module provides the storage (:class:`RoutedExpertsPool`) and the
per-forward capture/commit controller (:class:`RoutedExpertsCapturer`). Wiring
the capture hook into the MoE forward and the retrieval into the prefix-cache
hit path is documented in ``docs/guides/routing-replay-r3.md`` (it requires
threading ``out_cache_loc`` into the MoE block and handling TP/EP all-gather
alignment, which are model-forward changes).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Sentinel for an unwritten routing entry. -1 matches the fill the existing
# expert-distribution recorder uses for empty topk slots
# (moe/distribution_recorder.py::_DetailSinglePassGatherer.reset).
ROUTING_UNSET = -1


class RoutedExpertsPool:
    """Persistent, KV-slot-indexed store of per-layer top-k expert ids.

    Layout mirrors the KV pools (``layers/attention/kv_cache/mha.py``): a single
    contiguous buffer of shape ``[size + 1, num_moe_layers, top_k]`` indexed on
    dim 0 by the flat KV **slot** index. Row 0 is a reserved padding/dummy slot,
    exactly like the KV pools' reserved slot 0, so a padding slot index of 0
    never aliases a real token's routing.

    The per-token memory cost is ``num_moe_layers * top_k * 4`` bytes (int32),
    typically ~1-2 KB/token — roughly 1-5% of the KV footprint per token.

    Args:
        size: Number of real KV slots (matches the KV pool's ``size``). One
            extra reserved slot (index 0) is allocated internally.
        num_moe_layers: Number of MoE layers whose routing is stored.
        top_k: Experts selected per token per layer (``num_experts_per_tok``).
        device: Device for the buffer (e.g. ``"cuda"``). Defaults to ``"cpu"``.
        dtype: Integer dtype for stored ids. Defaults to ``torch.int32``,
            matching ``TopK.topk_indices_dtype``.
        fill: Sentinel written to unset entries. Defaults to
            :data:`ROUTING_UNSET`.

    Attributes:
        buffer: The backing tensor ``[size + 1, num_moe_layers, top_k]``.
    """

    def __init__(
        self,
        size: int,
        num_moe_layers: int,
        top_k: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.int32,
        fill: int = ROUTING_UNSET,
    ) -> None:
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}.")
        if num_moe_layers <= 0:
            raise ValueError(f"num_moe_layers must be positive, got {num_moe_layers}.")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")

        self.size = size
        self.num_moe_layers = num_moe_layers
        self.top_k = top_k
        self.fill = fill
        # +1 for the reserved padding slot at index 0.
        self.buffer = torch.full(
            (size + 1, num_moe_layers, top_k),
            fill,
            dtype=dtype,
            device=device,
        )

    def store_layer(self, layer_id: int, loc: torch.Tensor, topk_ids: torch.Tensor) -> None:
        """Scatter one layer's routing into the pool at the given KV slots.

        Args:
            layer_id: MoE layer index in ``[0, num_moe_layers)``.
            loc: 1-D int tensor of KV slot indices, shape ``[n]``. These are the
                same ``out_cache_loc`` slots the KV cache was written to.
            topk_ids: Expert ids for those slots, shape ``[n, top_k]``.

        Mirrors ``MHATokenToKVPool.set_kv_buffer``'s scatter-by-``loc``.
        """
        self._check_layer(layer_id)
        if topk_ids.shape[0] != loc.shape[0]:
            raise ValueError(
                f"topk_ids rows ({topk_ids.shape[0]}) must match loc rows "
                f"({loc.shape[0]}); routing rows must be the rank-local tokens "
                "that out_cache_loc indexes (see TP/EP alignment note in docs)."
            )
        if topk_ids.shape[1] != self.top_k:
            raise ValueError(
                f"topk_ids has top_k={topk_ids.shape[1]}, pool expects {self.top_k}."
            )
        self.buffer[loc.long(), layer_id, :] = topk_ids.to(self.buffer.dtype)

    def gather(self, loc: torch.Tensor) -> torch.Tensor:
        """Return all layers' routing for the given KV slots.

        Args:
            loc: 1-D int tensor of KV slot indices, shape ``[n]``.

        Returns:
            Tensor ``[n, num_moe_layers, top_k]``. Slots never written read back
            as ``fill``. This is the retrieval used on a prefix-cache hit to
            recover the shared prefix's routing without recompute.
        """
        return self.buffer[loc.long()]

    def gather_layer(self, layer_id: int, loc: torch.Tensor) -> torch.Tensor:
        """Return one layer's routing for the given KV slots, shape ``[n, top_k]``."""
        self._check_layer(layer_id)
        return self.buffer[loc.long(), layer_id, :]

    def reset(self) -> None:
        """Clear the whole pool back to ``fill`` (e.g. between eval runs)."""
        self.buffer.fill_(self.fill)

    @property
    def num_bytes(self) -> int:
        """Total bytes held by the backing buffer."""
        return self.buffer.element_size() * self.buffer.nelement()

    def _check_layer(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.num_moe_layers:
            raise ValueError(
                f"layer_id={layer_id} out of range [0, {self.num_moe_layers})."
            )


class RoutedExpertsCapturer:
    """Per-forward capture controller that commits routing to a pool by slot.

    Lifecycle per model forward::

        capturer.begin_forward(out_cache_loc)   # slots the KV write targets
        ...                                       # for each MoE layer:
        capturer.capture(layer_id, topk_ids)      # after select_experts
        capturer.commit()                         # scatter all layers into pool

    ``begin_forward(None)`` (or never calling it) makes ``capture`` a no-op, so
    the hook can stay installed and cost nothing on requests that did not ask
    for routing replay.

    Retrieval is just ``capturer.pool.gather(slots)``.

    TP/EP note: under tensor/expert parallelism ``topk_ids`` rows are the
    *global* all-gathered token set, not the rank-local tokens ``out_cache_loc``
    indexes. When the row counts disagree we skip the layer (and count it in
    :attr:`skipped_misaligned`) rather than corrupt the pool — aligning the
    gathered rows back to local is the remaining model-forward work tracked in
    the docs.
    """

    def __init__(self, pool: RoutedExpertsPool) -> None:
        self.pool = pool
        self._loc: torch.Tensor | None = None
        self._captured: dict[int, torch.Tensor] = {}
        self.skipped_misaligned: int = 0

    @property
    def active(self) -> bool:
        """True between ``begin_forward(loc)`` and ``commit()`` for a real ``loc``."""
        return self._loc is not None

    def begin_forward(self, out_cache_loc: torch.Tensor | None) -> None:
        """Start capturing for a forward whose KV write targets ``out_cache_loc``.

        Pass ``None`` to leave capture disabled for this forward.
        """
        self._loc = out_cache_loc
        self._captured.clear()

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Record one MoE layer's ``topk_ids`` for the active forward.

        No-op when capture is not active. Rows that don't line up with the
        forward's slots (TP/EP all-gather) are skipped and counted.
        """
        if self._loc is None:
            return
        if topk_ids.shape[0] != self._loc.shape[0]:
            self.skipped_misaligned += 1
            return
        # Detach + clone: topk_ids is transient inside the MoE forward and may be
        # freed/overwritten before commit.
        self._captured[layer_id] = topk_ids.detach().clone()

    def commit(self) -> None:
        """Scatter all captured layers into the pool, then end the forward."""
        if self._loc is not None:
            for layer_id, topk_ids in self._captured.items():
                self.pool.store_layer(layer_id, self._loc, topk_ids)
        self._loc = None
        self._captured.clear()


# --------------------------------------------------------------------------- #
# Global accessor, mirroring moe/distribution_recorder.py's global recorder so
# the MoE forward hook can reach the capturer without threading it through every
# layer signature.
# --------------------------------------------------------------------------- #

_global_routed_experts_capturer: RoutedExpertsCapturer | None = None


def get_global_routed_experts_capturer() -> RoutedExpertsCapturer | None:
    """Return the process-global capturer, or ``None`` if R3 capture is off."""
    return _global_routed_experts_capturer


def set_global_routed_experts_capturer(
    capturer: RoutedExpertsCapturer | None,
) -> None:
    """Install (or clear with ``None``) the process-global capturer."""
    global _global_routed_experts_capturer
    _global_routed_experts_capturer = capturer
