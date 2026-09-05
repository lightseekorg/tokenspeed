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

"""Per-forward memo of the write-slot mappings DeepSeek V4's layers share."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TypeVar

T = TypeVar("T")


class DeepseekV4ForwardSlotMappings:
    """The token-shaped write-slot mappings one forward's layers share.

    The SWA mapping, the compressor state / compressed-cache mappings per
    compress ratio and the indexer-state mapping depend only on the forward's
    rows and the published metadata, not on the layer, so the first layer
    that needs one computes it and every later layer reuses it (each is a
    chain of small elementwise kernels — per layer they once baked ~7 kernels
    x 61 layers into the captured decode graph).

    This is backend scratch, deliberately outside the metadata slots:
    ``graph_ptr_guard`` snapshots every tensor a slot reaches, and a mapping
    computed under capture must not be pinned there. The backend clears the
    memo in every slot publisher, and every forward is preceded by a publish
    (extend init, decode refresh, the drafter's per-step advance, capture
    seeding), so no mapping outlives the forward it was computed for.
    """

    def __init__(self) -> None:
        self._entries: dict[Hashable, object] = {}

    def clear(self) -> None:
        self._entries.clear()

    def get_or_compute(self, key: Hashable, compute: Callable[[], T]) -> T:
        """Return the mapping memoized under ``key``, computing it on the
        first request of the forward.

        Args:
            key: The mapping's identity, e.g. ``"swa"`` or
                ``("state", compress_ratio)``.
            compute: Builds the mapping when the memo has no entry.

        Returns:
            The memoized value; the same object for every later call within
            the forward.
        """
        try:
            return self._entries[key]  # type: ignore[return-value]
        except KeyError:
            value = compute()
            self._entries[key] = value
            return value
