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
from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

import torch

from tokenspeed.runtime.distributed.mapping import Group


@dataclass(frozen=True)
class AllReducePlan:
    """Producer outputs paired with their selected reduction.

    ``outputs`` must be fully written before calling :meth:`run`.
    """

    outputs: tuple[torch.Tensor, ...]
    _run: Callable[[], tuple[torch.Tensor, ...]]

    def run(self) -> tuple[torch.Tensor, ...]:
        return self._run()


class CommBackend(ABC):
    """Interface that all communication backends must implement.

    All group parameters are tuples of global ranks, e.g. (0, 1, 2, 3).
    Process groups are looked up from pg_manager, not created here.
    """

    # ---- Collective ops ----

    @overload
    def all_reduce(
        self, tensor: torch.Tensor, group: Group, op=None
    ) -> torch.Tensor: ...

    @overload
    def all_reduce(
        self, tensor: tuple[torch.Tensor, ...], group: Group, op=None
    ) -> tuple[torch.Tensor, ...]: ...

    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Reduce one tensor or a collection of independent tensors."""
        if isinstance(tensor, torch.Tensor):
            raise NotImplementedError
        tensors = tensor
        if len(tensors) == 0:
            raise ValueError("all-reduce requires at least one tensor")
        return tuple(self.all_reduce(value, group, op=op) for value in tensors)

    def prepare_all_reduce_lane(self, group: Group, hidden_dim: int) -> bool:
        """Prepare an implementation-specific one-shot lane when supported."""

        return False

    def plan_all_reduce(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> AllReducePlan:
        """Allocate producer outputs and bind their reduction strategy.

        Args:
            shapes: Shapes of the outputs the producer will write.
            like: Tensor providing dtype and device for ordinary allocations.
            group: Global ranks participating in every reduction.
            op: Reduction operation.

        Returns:
            A plan containing writable outputs and their reduction operation.
        """
        if not shapes:
            raise ValueError("all-reduce plan requires at least one output")
        outputs = tuple(like.new_empty(shape) for shape in shapes)

        def run() -> tuple[torch.Tensor, ...]:
            reduced = self.all_reduce(outputs, group, op=op)
            assert isinstance(reduced, tuple)
            return reduced

        return AllReducePlan(outputs, run)

    def all_reduce_residual_attnres(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float,
        group: Group,
        op=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce attention output, accumulate residual, and finish AttnRes.

        Backends without a fused collective preserve the model's BF16
        boundaries through the existing operations.
        """
        from tokenspeed_kernel.ops.activation.triton import attnres_combine

        reduced = self.all_reduce(partial, group, op=op)
        residual_out = residual + reduced
        hidden = attnres_combine(
            residual_out,
            score_weight,
            output_weight,
            eps,
            scratch,
            torch.empty_like(residual_out),
        )
        return hidden, residual_out

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

    @abstractmethod
    def all_to_all_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        """Even-split all_to_all. output and input must have same numel
        divisible by len(group).
        """
        ...

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
