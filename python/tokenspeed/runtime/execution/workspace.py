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

"""One shared scratch block per device for per-operation kernel workspaces.

The contract, which every caller must hold to:

* **Re-fetch every use; never hold a view across ``allocate`` calls.** Growth
  drops the block and allocates a larger one, so the address changes; a held
  view keeps the dead block alive and reads stale bytes forever. Addresses are
  only stable once the pool is frozen.

* **Request everything an operation needs in one call.** Views from a single
  ``allocate`` never overlap; views from separate calls deliberately share the
  same bytes. The reuse is the point -- the block converges to the largest
  single request instead of the sum of every caller.

* **Main stream only.** Cross-call reuse is safe because callers launch their
  kernels in order on one stream, so an earlier op is done with its scratch
  before a later op's kernels run. Work on a side stream runs concurrently and
  must bring its own buffer.

* **The block must reach its peak before the first CUDA-graph capture.** A
  graph records the raw address of any view captured into it, so the executor
  freezes the pool right before capture and a frozen pool refuses to grow
  (naming the caller that asked). Cache sizing grows the block to the known
  peak before profiling free memory, which both satisfies this rule and makes
  the bytes come out of the measured KV budget instead of the utilization
  headroom -- see the attention registry.

Sizing policy (constants, batch shapes, env overrides) belongs to callers; the
pool only carves.
"""

from __future__ import annotations

import inspect
import math
import os

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

_ALIGN = 256


def _round_up(n: int, align: int = _ALIGN) -> int:
    return (n + align - 1) // align * align


def _caller() -> str:
    """First frame outside this module, to name who tripped a frozen pool."""
    frame = inspect.currentframe()
    while frame is not None:
        if os.path.basename(frame.f_code.co_filename) != os.path.basename(__file__):
            return (
                f"{os.path.basename(frame.f_code.co_filename)}:"
                f"{frame.f_lineno}:{frame.f_code.co_name}"
            )
        frame = frame.f_back
    return "unknown"


class WorkspacePool:
    """One device's shared scratch block; the contract is in the module docstring."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._block: torch.Tensor | None = None
        self._frozen = False

    def allocate(
        self, *specs: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        """Return one view per ``(shape, dtype)``, none overlapping the others.

        The views are valid until the next ``allocate`` on this pool, which
        hands out the same bytes again.
        """
        if not specs:
            return []

        sizes = [_round_up(math.prod(shape) * dtype.itemsize) for shape, dtype in specs]
        total = sum(sizes)
        block = self._block
        held = 0 if block is None else block.numel()

        if held < total:
            if self._frozen:
                raise RuntimeError(
                    f"workspace pool is frozen but {_caller()} needs "
                    f"{total / (1 << 20):.2f} MB (block holds "
                    f"{held / (1 << 20):.2f} MB). Growing would move the block "
                    "out from under any CUDA graph that captured a view of it; "
                    "the block must reach its peak before the first capture."
                )
            logger.info(
                "workspace block: %.2f MB -> %.2f MB (%s)",
                held / (1 << 20),
                total / (1 << 20),
                _caller(),
            )
            # Drop the old block before allocating the larger one so the
            # caching allocator can serve the new request from those bytes.
            self._block = None
            del block
            self._block = torch.empty(total, dtype=torch.uint8, device=self.device)
            block = self._block

        views = []
        offset = 0
        for (shape, dtype), size in zip(specs, sizes):
            end = offset + math.prod(shape) * dtype.itemsize
            views.append(block[offset:end].view(dtype).view(shape))
            offset += size
        return views

    def freeze(self) -> None:
        """Pin the block's address; growth now raises. Call before graph capture."""
        self._frozen = True
        logger.info(
            "workspace frozen: %.2f MB",
            (0 if self._block is None else self._block.numel()) / (1 << 20),
        )

    def unfreeze(self) -> None:
        """Allow growth again, for reconfiguration that re-runs cache sizing."""
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen


_pools: dict[torch.device, WorkspacePool] = {}


def workspace_pool(device: torch.device | str) -> WorkspacePool:
    """The pool for ``device``, created on first use."""
    key = torch.device(device)
    pool = _pools.get(key)
    if pool is None:
        pool = WorkspacePool(key)
        _pools[key] = pool
    return pool


def reset_workspace_pools() -> None:
    """Drop every pool. For tests that build more than one engine per process."""
    _pools.clear()
