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

"""Lifecycle protocol for models owning peer-sharded VMM device storage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PeerStorageOwner(Protocol):
    """Model-owned CUDA VMM peer storage participating in weight lifecycle.

    Unlike :class:`~tokenspeed.runtime.layers.moe.peer_weights.PeerWeightOwner`
    (which owns DWDP MoE staging/prefetch controllers), this protocol covers
    read-mostly VMM-backed peer tables such as peer-sharded Engram embeddings
    that live outside PyTorch's ordinary caching allocator.
    """

    def peer_storage_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """Return parameters backed by peer VMM storage."""
        ...

    def peer_storage_released(self) -> bool:
        """Whether any owned peer VMM storage is currently released."""
        ...

    def release_peer_storage(self, *, backup: bool) -> None:
        """Unmap and release physical VMM storage while keeping VAs reserved."""
        ...

    def resume_peer_storage(self, *, backup: bool) -> None:
        """Reallocate and remap VMM storage at the same virtual addresses."""
        ...

    def close_peer_storage(self) -> None:
        """Permanently unmap and free peer VMM storage at shutdown."""
        ...
