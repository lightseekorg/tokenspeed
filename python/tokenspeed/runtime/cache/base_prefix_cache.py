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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, NamedTuple

import torch


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Page indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        device_prefix_length:   Length of the common prefix in tokens (not pages).
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if KVStore is not enabled,
                            this **must** be the same as `last_device_node`.
        host_hit_length :   Number of tokens hit on the host, if applicable.
                            0 if KVStore is not enabled.
                            Note: node.host_value stores token indices.
    """

    device_indices: torch.Tensor | None = None
    last_device_node: Any = None
    device_prefix_length: int = 0
    last_host_node: Any = None
    host_hit_length: int = 0


class BasePrefixCache(ABC):
    """Cache can be indexed by either rid or key."""

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def match_prefix(self, **kwargs) -> MatchResult:
        """Match a request prefix and optionally prepare req-local state.

        Unified contract:
        - When called without `req`, implementations should behave like a pure
          prefix lookup and avoid mutating request-local state.
        - When called with `req`, implementations may prepare request-local
          execution state required by that cache backend.
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def cache_finished_req(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def cache_unfinished_req(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evict(self, num_tokens: int, evict_callback: Callable[..., None]):
        raise NotImplementedError

    @abstractmethod
    def inc_lock_ref(self, node):
        raise NotImplementedError

    @abstractmethod
    def dec_lock_ref(self, node):
        raise NotImplementedError

    @abstractmethod
    def evictable_size(self):
        raise NotImplementedError

    @abstractmethod
    def protected_size(self):
        raise NotImplementedError()

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()
