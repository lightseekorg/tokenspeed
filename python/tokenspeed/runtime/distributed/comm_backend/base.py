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
from typing import TYPE_CHECKING, Protocol

import torch

from tokenspeed.runtime.distributed.mapping import Group

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.comm_backend.projection import (
        ProjectionSpec,
        ProjectionWorkspace,
    )
    from tokenspeed.runtime.execution.workspace import WorkspacePool


class CommWork(Protocol):
    """A collective started by an ``*_async`` op (``torch.distributed.Work``
    is one)."""

    def wait(self) -> object:
        """Order the current stream after the collective; the host does not
        block on a device collective."""


class CommBackend(ABC):
    """Interface that all communication backends must implement.

    All group parameters are tuples of global ranks, e.g. (0, 1, 2, 3).
    Process groups are looked up from pg_manager, not created here.
    """

    def for_named_groups(self, name: str) -> "CommBackend":
        """A backend running its collectives on the process groups named
        ``name`` (``ProcessGroupManager.init_process_group``) instead of the
        shared ones.

        A named group has communicators of its own, so these collectives never
        queue behind the shared ones: a caller overlaps transfers with the
        shared collectives by starting them with an ``*_async`` op here.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not run on named process groups"
        )

    # ---- Collective ops ----

    def get_prepared_all_gather(self, group: Group, dtype: torch.dtype):
        """Return a prepared column all-gather allowed by this backend's policy."""
        return None

    def broadcast_from_first_rank(self, tensor: torch.Tensor, group: Group) -> bool:
        """Broadcast in place when prepared; otherwise let the caller fall back."""
        return False

    def prepare_projection(
        self, spec: "ProjectionSpec", scratch_pool: "WorkspacePool | None" = None
    ) -> "ProjectionWorkspace":
        """Prepare model-owned scratch for this backend before graph capture."""
        from tokenspeed.runtime.distributed.comm_backend.projection import (
            prepare_projection_workspace,
        )

        return prepare_projection_workspace(spec, self, scratch_pool)

    def try_fused_dp_column_linear(
        self,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor | None,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
        workspace: "ProjectionWorkspace",
    ) -> bool:
        """Run fused DP column-linear, or leave out untouched and return False."""
        return False

    def try_fused_dp_row_linear(
        self,
        tensor: torch.Tensor,
        input_scales: torch.Tensor | None,
        weight: torch.Tensor,
        weight_scales: torch.Tensor | None,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
        workspace: "ProjectionWorkspace",
    ) -> bool:
        """Run fused DP row-linear, or leave out untouched and return False."""
        return False

    def projection_all_gather(
        self,
        tensor: torch.Tensor,
        rows: int,
        quantize: bool,
        workspace: "ProjectionWorkspace",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gather owner rows into borrowed activations and optional FP8 scales."""
        from tokenspeed.runtime.distributed.comm_backend.projection import (
            projection_all_gather,
        )

        return projection_all_gather(tensor, rows, quantize, workspace, self)

    def projection_all_to_all(
        self,
        tensor: torch.Tensor,
        rows: int,
        inverse: bool,
        quantize: bool,
        out: torch.Tensor | None,
        workspace: "ProjectionWorkspace",
        *,
        input_scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exchange token/channel axes using prepared layout and fusion resources."""
        from tokenspeed.runtime.distributed.comm_backend.projection import (
            projection_all_to_all,
        )

        return projection_all_to_all(
            tensor,
            rows,
            inverse,
            quantize,
            out,
            workspace,
            self,
            input_scales=input_scales,
        )

    def acquire_projection_output(
        self, rows: int, workspace: "ProjectionWorkspace"
    ) -> torch.Tensor:
        """Borrow a GEMM destination consumed by projection_reduce_scatter."""
        from tokenspeed.runtime.distributed.comm_backend.projection import (
            acquire_projection_output,
        )

        return acquire_projection_output(rows, workspace, self)

    def projection_reduce_scatter(
        self, tensor: torch.Tensor, rows: int, workspace: "ProjectionWorkspace"
    ) -> torch.Tensor:
        """Reduce padded owner segments into an owned local output tensor."""
        from tokenspeed.runtime.distributed.comm_backend.projection import (
            projection_reduce_scatter,
        )

        return projection_reduce_scatter(tensor, rows, workspace, self)

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

    def supports_all_reduce_mhc_norm(
        self, x: torch.Tensor, norm_weight: torch.Tensor, group: Group
    ) -> bool:
        """Whether the prepared group can fuse the anticipated output spec of x."""
        return False

    def all_reduce_mhc_norm(
        self, x, residual, post, comb, pre, weight, eps, group: Group
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute admitted fusion; return residual/norm outputs or raise."""
        raise NotImplementedError(
            "This backend does not support mHC communication fusion"
        )

    def prepare_all_reduce_lane(self, group: Group, hidden_dim: int) -> bool:
        """Prepare an implementation-specific one-shot lane when supported."""

        return False

    def prepare_all_reduce_buffers(
        self,
        group: Group,
        *,
        staged_max_numel: int,
        producer_direct_max_numel: int,
        dtype: torch.dtype,
    ) -> bool:
        """Return false when the backend has no persistent buffers to prepare."""

        return False

    def can_acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> bool:
        """Whether ``acquire_all_reduce_outputs`` returns producer-direct memory.

        The base implementation of ``acquire_all_reduce_outputs`` always
        succeeds, but by allocating ordinary tensors that the collective must
        then stage. Callers that only want the buffers when the reduction can
        consume them in place -- and would otherwise keep a persistent buffer of
        their own -- ask here first.

        Must be answerable without side effects beyond the backend's own lazy
        setup, and must return the same answer on every rank of ``group``: a
        caller that acquires on some ranks and not others would leave the group
        disagreeing on which collective runs.
        """

        return False

    def all_reduce_mhc_post(
        self,
        tensor: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        group: Group,
    ) -> torch.Tensor | None:
        """Sum ``tensor`` over ``group`` and apply the hc=4 mHC post in one step.

        Args:
            tensor: BF16 [T, H] partial, normally from
                ``acquire_all_reduce_outputs``.
            residual: BF16 residual streams [T, 4, H].
            post: FP32 post coefficients [T, 4].
            comb: FP32 combination [T, 4 input, 4 output].
            group: Global ranks participating in the reduction.

        Returns:
            The new BF16 residual streams [T, 4, H], or None when the backend
            has no fused path for these operands (nothing has run then; the
            caller reduces and applies the post itself). The answer must agree
            on every rank of ``group``.
        """
        return None

    def acquire_all_reduce_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
        like: torch.Tensor,
        group: Group,
        op=None,
    ) -> tuple[torch.Tensor, ...]:
        """Acquire writable outputs for a later all-reduce.

        Args:
            shapes: Shapes of the outputs the producer will write.
            like: Tensor providing dtype and device for ordinary allocations.
            group: Global ranks participating in every reduction.
            op: Reduction operation.

        Returns:
            Writable outputs accepted by ``all_reduce`` after production.
        """
        if not shapes:
            raise ValueError("all-reduce requires at least one output")
        return tuple(like.new_empty(shape) for shape in shapes)

    @abstractmethod
    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor: ...

    @abstractmethod
    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None: ...

    def all_gather_single_async(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> CommWork:
        """Start ``all_gather_single`` without ordering the current stream
        after it.

        The collective reads ``input`` after the work queued on the current
        stream so far. ``wait()`` on the returned work orders the stream it is
        called on after the collective; ``output`` holds the gathered rows
        from there on. The backend keeps both tensors' memory from reuse until
        the collective completes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no asynchronous all-gather"
        )

    @abstractmethod
    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor: ...

    @abstractmethod
    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: Group,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
    ) -> None:
        """All-to-all along dim 0 into a pre-allocated output buffer.

        Without split sizes both tensors split evenly into ``len(group)``
        chunks. With them, ``input_split_sizes[i]`` rows go to the group's
        ``i``-th rank and ``output_split_sizes[i]`` rows arrive from it (a
        zero-row chunk is allowed); the sizes must sum to the tensors' rows.
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
