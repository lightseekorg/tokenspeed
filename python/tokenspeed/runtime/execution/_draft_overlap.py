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

"""Overlap a captured decode graph's draft tail with execution-stream preparation.

The graph owns all model work. Only the prefix event releases the execution
stream; the result fence always covers the graph tail and output copies.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.drafter.base import BaseDrafter

logger = get_colorful_logger(__name__)

# Unset: on wherever the configuration is eligible; "1": required; "0": off.
DRAFT_GRAPH_OVERLAP_ENV = "TOKENSPEED_DRAFT_GRAPH_OVERLAP"


def draft_overlap_ineligibility(
    *,
    nvidia_decode_graphs: bool,
    pp_size: int,
    dp_size: int,
    spec_algo: str | None,
    drafter: BaseDrafter | None,
    grammar_backend: str,
    shared_moe_row_masks: bool,
    cache_layers: Callable[[], tuple[set[int], set[int]]],
    separate_metadata: bool,
) -> str | None:
    """The first condition a configuration misses for draft-tail overlap.

    Args:
        nvidia_decode_graphs: CUDA (not HIP) decode graphs are captured.
        pp_size: Pipeline-parallel size.
        dp_size: Data-parallel size.
        spec_algo: The speculative algorithm, or None.
        drafter: The drafter, or None without speculation.
        grammar_backend: The grammar backend name.
        shared_moe_row_masks: Expert-load recording or MegaMoE's live-request
            mask is installed.
        cache_layers: Returns the target and draft KV cache layer sets; read
            only once the conditions above hold.
        separate_metadata: Target and drafter own separate attention backends.

    Returns:
        The unmet condition, or None when the overlap can be enabled.
    """
    from tokenspeed.runtime.execution.drafter.eagle import Eagle

    if not nvidia_decode_graphs or pp_size != 1 or dp_size != 1:
        return "Draft graph overlap requires NVIDIA decode graphs, DP1/PP1"
    if spec_algo != "MTP" or type(drafter) is not Eagle:
        return "Draft graph overlap requires the Eagle MTP drafter"
    reason = drafter.decode_overlap_ineligibility()
    if reason is not None:
        return reason
    if grammar_backend != "none":
        return "Draft graph overlap does not support grammar"
    if shared_moe_row_masks:
        return "Draft graph overlap does not support shared MoE row masks"
    target_layers, draft_layers = cache_layers()
    if (
        not target_layers
        or not draft_layers
        or not target_layers.isdisjoint(draft_layers)
        or not separate_metadata
    ):
        return "Draft graph overlap requires separate target/draft cache layers and metadata"
    return None


def resolve_draft_overlap(
    setting: str | None, ineligibility: Callable[[], str | None]
) -> bool:
    """Whether decode graphs overlap their draft tail with the next round's preparation.

    Args:
        setting: The ``TOKENSPEED_DRAFT_GRAPH_OVERLAP`` value. Unset (None) turns
            the overlap on wherever ``ineligibility`` finds no unmet condition
            and logs the condition otherwise; "1" requires it; "0" turns it off
            without checking.
        ineligibility: Returns the first unmet condition, or None.

    Raises:
        ValueError: A value other than 0 or 1, or "1" on a configuration that
            misses a condition.
    """
    if setting not in (None, "0", "1"):
        raise ValueError(f"{DRAFT_GRAPH_OVERLAP_ENV} must be 0 or 1")
    if setting == "0":
        return False
    reason = ineligibility()
    if reason is None:
        return True
    if setting == "1":
        raise ValueError(reason)
    logger.info(f"Draft graph overlap off: {reason}")
    return False


@dataclass(frozen=True)
class RoundFence:
    """Keep both completion events and their graph alive until result retirement."""

    done: torch.cuda.Event
    outputs_done: torch.cuda.Event
    graph: torch.cuda.CUDAGraph
    prefix_ready: torch.cuda.Event

    def synchronize(self) -> None:
        self.done.synchronize()


@dataclass
class _PendingRound:
    graph: torch.cuda.CUDAGraph
    execution_stream: torch.cuda.Stream
    fence: RoundFence | None


class DecodeGraphOverlap:
    """Release a complete model graph's prefix without retiring its tail.

    Construct only for a configuration whose tail read/write ownership was
    checked. Capture invokes snapshot() before record_prefix(); replay never
    runs either Python method. The owner calls fence_outputs() once before
    settle(), and settle() before another model graph or any resource reuse.
    """

    def __init__(self) -> None:
        self.round_stream = torch.cuda.Stream()
        self._events: dict[torch.cuda.CUDAGraph, torch.cuda.Event] = {}
        self._capture_event: torch.cuda.Event | None = None
        self._prefix_records = 0
        self._pending: _PendingRound | None = None
        self._batch: tuple[tuple[int, int], ...] = ()

    def prepare_batch(
        self, admissions: tuple[tuple[int, int], ...], *, reset_state: bool
    ) -> None:
        """Join before a changed batch or reset can mutate tail-owned state.

        ``admissions`` pairs each admission identity with its request-pool slot.
        ``reset_state`` also joins for extends and history reseeding.
        """
        if reset_state or admissions != self._batch:
            self.settle()
        self._batch = admissions

    @contextmanager
    def capture(self, graph: torch.cuda.CUDAGraph) -> Generator[None, None, None]:
        if self._capture_event is not None or self._pending is not None:
            raise RuntimeError("overlap capture requires no active capture or round")
        ready = torch.cuda.Event(external=True)
        self._capture_event = ready
        self._prefix_records = 0
        try:
            yield
            if self._prefix_records != 1:
                raise RuntimeError(
                    f"expected one draft prefix, got {self._prefix_records}"
                )
            self._events[graph] = ready
        finally:
            self._capture_event = None

    def snapshot(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy a tail input while capturing, before the prefix can release it."""
        return tensor.clone() if self._capture_event is not None else tensor

    def record_prefix(self) -> None:
        if self._capture_event is not None:
            self._prefix_records += 1
            if self._prefix_records != 1:
                raise RuntimeError("draft prefix recorded more than once")
            self._capture_event.record()

    def replay(self, graph: torch.cuda.CUDAGraph, *, live_rows: int) -> None:
        if self._pending is not None:
            raise RuntimeError("model graph replay before pending draft settled")
        if live_rows == 0:
            graph.replay()
            return
        ready = self._events[graph]
        execution = torch.cuda.current_stream()
        self.round_stream.wait_stream(execution)
        with torch.cuda.stream(self.round_stream):
            graph.replay()
        execution.wait_event(ready)
        self._pending = _PendingRound(graph, execution, None)

    def fence_outputs(
        self, outputs_done: torch.cuda.Event
    ) -> RoundFence | torch.cuda.Event:
        """Return one result fence covering real candidates and output copies."""
        pending = self._pending
        if pending is None:
            return outputs_done
        if pending.fence is not None:
            raise RuntimeError("output completion fenced twice")
        done = torch.cuda.Event()
        with torch.cuda.stream(self.round_stream):
            self.round_stream.wait_event(outputs_done)
            done.record()
        fence = RoundFence(
            done, outputs_done, pending.graph, self._events[pending.graph]
        )
        pending.fence = fence
        return fence

    def settle(self) -> None:
        pending = self._pending
        if pending is None:
            return
        if pending.fence is None:
            raise RuntimeError("draft settle before output completion was fenced")
        # DeviceHandle's non-forward work already orders behind execution_stream.
        # Keep that contract even when settle is called on the default stream.
        pending.execution_stream.wait_event(pending.fence.done)
        self._pending = None

    def release(self) -> None:
        """Drain before destroying captured event handles or their graph pool."""
        self.settle()
        self.round_stream.synchronize()
        self._events.clear()
