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

"""GDN/mamba2 state-slab management for the flat KV pool.

Extracted verbatim from ``MHATokenToKVPool`` so the "MHA" (multi-head
attention KV) class no longer mixes recurrent-state bookkeeping into its
attention-KV responsibility. Behavior is byte-identical: same equalization
pre-check, same flat-GDN gate, same slab shapes/dtypes/count, same null-page
(row 0) convention, and the same ``get_state_buffers`` return values and
error cases.

The state slabs live under the SAME page-id space as the KV pages (a single
block-id space): one ``(conv, ssm)`` pair per state LAYER, row-indexed by
page id, row 0 the never-written null page.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.configs import paged_cache_spec
from tokenspeed.runtime.configs.flat_memory_plan import (
    equalized_block_size,
    state_const_bytes,
)
from tokenspeed.runtime.configs.paged_cache_spec import STATE_LAYER_TYPES
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class FlatStateSlabs:
    """Owns the GDN/mamba2 conv+ssm state slabs for the flat KV pool.

    Args:
        layer_types: Per-layer type labels; state layers carry the
            ``STATE_LAYER_TYPES`` labels. Drives the slab pairing.
        conv_state_shape / temporal_state_shape: Per-state-layer mamba2
            state tensor shapes (configs' mamba2_cache_params). The temporal
            shape is K-last ``[Hv, V, K]``; ``None`` on pure-attention models.
        conv_dtype / ssm_dtype: State dtypes; default to ``default_dtype``.
        default_dtype: Pool store dtype used when a state dtype is ``None``.
        page_size: The (already-equalized) page size P.
        size: Total token slots (the flat pool size; must be whole pages
            when slabs are active).
        kv_bytes_per_slot: Bytes one KV history slot occupies
            (``2 * head_num * head_dim * store_dtype.itemsize``); used only
            by the equalization pre-check.

    The equalization pre-check runs in ``__init__`` (same trigger, same
    ``ValueError``). Device tensors are allocated lazily by ``allocate`` so
    the pool can keep them inside its memory-saver "kv_cache" region.
    """

    def __init__(
        self,
        *,
        layer_types: tuple[str, ...],
        conv_state_shape: tuple[int, ...] | None,
        temporal_state_shape: tuple[int, ...] | None,
        conv_dtype: torch.dtype | None,
        ssm_dtype: torch.dtype | None,
        default_dtype: torch.dtype,
        page_size: int,
        size: int,
        kv_bytes_per_slot: int,
    ):
        self._layer_types = tuple(layer_types or ())
        # Per-state-layer mamba2 shapes (configs' mamba2_cache_params);
        # None on pure-attention models.
        self._conv_state_shape = (
            tuple(conv_state_shape) if conv_state_shape is not None else None
        )
        self._temporal_state_shape = (
            tuple(temporal_state_shape) if temporal_state_shape is not None else None
        )
        self._conv_dtype = conv_dtype if conv_dtype is not None else default_dtype
        self._ssm_dtype = ssm_dtype if ssm_dtype is not None else default_dtype
        self.page_size = page_size
        self.size = size

        # layer_id -> state pair index (the n-th state layer binds pair n).
        # Derives purely from layer_types; shared by the KV skip set on the
        # pool and the state-slab block below.
        self._layer_state_pair: dict[int, int] = {
            layer_id: pair
            for pair, layer_id in enumerate(
                layer_id
                for layer_id, label in enumerate(self._layer_types)
                if label in STATE_LAYER_TYPES
            )
        }

        if (
            self._conv_state_shape is not None
            and self._temporal_state_shape is not None
        ):
            # The flat plan packs [conv|ssm] state rows and KV rows into one
            # page-id space, so P must already cover the widest constant row
            # (the equalizer would otherwise inflate it behind the
            # allocator's back).
            equalized = equalized_block_size(
                layer_types=list(self._layer_types),
                kv_bytes_per_slot=kv_bytes_per_slot,
                state_const_bytes=state_const_bytes(
                    self._conv_state_shape,
                    self._conv_dtype,
                    self._temporal_state_shape,
                    self._ssm_dtype,
                ),
                block_size=self.page_size,
                alignment=1,
            )
            if equalized != self.page_size:
                raise ValueError(
                    "page_size must be pre-equalized for state layers; need "
                    f">= {equalized} (got {self.page_size})"
                )

        # Flat GDN predicate: ONE boolean gates both skipping per-layer KV on
        # state layers and allocating the state slabs -- the plan sizing
        # (registry) charges exactly full-layer KV + state rows, so the two
        # decisions must never diverge.
        self._flat_gdn = (
            bool(self._layer_state_pair)
            and self._conv_state_shape is not None
            and self._temporal_state_shape is not None
            and paged_cache_spec.scheduler_ext_flat_kvcache()
        )

        self.num_pages_with_null: int | None = None
        self.state_slabs: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def is_active(self) -> bool:
        """True iff the flat-GDN gate is on (state slabs will be allocated).

        When False the pool must NOT skip per-layer KV on state layers and
        must not allocate slabs (pure attention, radix ext, spec decode, or
        missing state shapes)."""
        return self._flat_gdn

    @property
    def state_layer_ids(self) -> frozenset[int]:
        """Layer ids that carry NO per-layer KV (state layers under flat
        GDN). Empty unless active, so non-flat profiles keep full KV."""
        return frozenset(self._layer_state_pair) if self._flat_gdn else frozenset()

    def is_state_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` is a KV-less state layer under flat GDN."""
        return self._flat_gdn and layer_id in self._layer_state_pair

    def allocate(self, device: str) -> None:
        """Allocate the ``(conv, ssm)`` slab pairs on ``device`` when active.

        Idempotent-shaped: a no-op leaving ``state_slabs == []`` when the
        flat-GDN gate is off. Call inside the pool's memory-saver "kv_cache"
        region so the slabs share the KV discard-on-sleep policy.
        """
        if not self._flat_gdn:
            self.state_slabs = []
            return
        # State slabs (GDN/mamba2 conv+ssm rows): one (conv, ssm) pair per
        # state LAYER (n-th state layer -> pair n), row-indexed by page id
        # over the SAME page-id space as the KV pages; row 0 is the null
        # page, never written -- mirrors the KV buffers' +page_size
        # dummy-page convention.
        assert self.size % self.page_size == 0, "flat pool size must be whole pages"
        self.num_pages_with_null = self.size // self.page_size + 1
        # The config publishes SSM rows directly in the kernels' native K-last
        # [Hv, V, K] layout, so the slab preserves the declared shape.
        self.state_slabs = [
            (
                torch.zeros(
                    (self.num_pages_with_null, *self._conv_state_shape),
                    dtype=self._conv_dtype,
                    device=device,
                ),
                torch.zeros(
                    (self.num_pages_with_null, *self._temporal_state_shape),
                    dtype=self._ssm_dtype,
                    device=device,
                ),
            )
            for _ in range(len(self._layer_state_pair))
        ]
        logger.info(
            "State slabs: %d (conv, ssm) pairs x %d page rows (row 0 = null page)",
            len(self._layer_state_pair),
            self.num_pages_with_null,
        )

    def get_state_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(conv, ssm) state slab pair for a state layer; the n-th state
        layer (within-state-label occurrence order, the slab pairing order)
        binds pair n. Raises ValueError for non-state layers."""
        pair = self._layer_state_pair.get(layer_id)
        if pair is None:
            raise ValueError(
                f"layer {layer_id} is not a state layer "
                f"(layer_types={self._layer_types!r})"
            )
        if not self.state_slabs:
            raise ValueError(
                f"layer {layer_id} is a state layer but no state "
                "slabs were allocated (state shapes missing or "
                "radix ext)"
            )
        return self.state_slabs[pair]
