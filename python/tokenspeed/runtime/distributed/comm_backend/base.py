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

"""Abstract base class for communication backends."""

from abc import ABC, abstractmethod

import torch

from tokenspeed.runtime.distributed.mapping import Group


class CommBackend(ABC):
    """Interface that all communication backends must implement.

    All group parameters are tuples of global ranks, e.g. (0, 1, 2, 3).
    Process groups are looked up from pg_manager, not created here.
    """

    # ---- Collective ops ----

    @abstractmethod
    def all_reduce(
        self, tensor: torch.Tensor, group: Group, op=None
    ) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor: ...

    @abstractmethod
    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None: ...

    @abstractmethod
    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor: ...

    # ---- Token-aware ops (uneven token distribution) ----

    @abstractmethod
    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor: ...

    @abstractmethod
    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor: ...
