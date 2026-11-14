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

"""Benchmark-only GDN placement on co-scheduled CUDA SM resources."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from cuda.bindings import driver as cu


def checked(result):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA placement operation failed: {result}")
    values = result[1:]
    return values[0] if len(values) == 1 else values


@dataclass(frozen=True)
class PlacementSpec:
    name: str
    max_ctas: int
    sm_count: int | None
    coscheduled_sm_count: int | None


PLACEMENTS = {
    "unrestricted24": PlacementSpec("unrestricted24", 24, None, None),
    "unrestricted16": PlacementSpec("unrestricted16", 16, None, None),
    "green16": PlacementSpec("green16", 16, 16, 16),
    "green32": PlacementSpec("green32", 24, 32, 16),
    "unrestricted32": PlacementSpec("unrestricted32", 32, None, None),
    "green32_cta32": PlacementSpec("green32_cta32", 32, 32, 16),
}


class CommitPlacement:
    """Order GDN launches on a primary or spatially bounded CUDA stream.

    Args:
        spec: Explicit CTA limit and optional co-scheduled SM partition.

    Green contexts share the primary context's allocations. CUDA graph nodes
    retain the execution context from capture; capture this scope itself when
    benchmarking graphs, rather than changing their replay stream afterward.
    """

    def __init__(self, *, spec: PlacementSpec):
        if spec.max_ctas <= 0:
            raise ValueError("max_ctas must be positive")
        if (spec.sm_count is None) != (spec.coscheduled_sm_count is None):
            raise ValueError(
                "SM count and co-scheduled count must be supplied together"
            )
        self.spec = spec
        self.device = torch.cuda.current_device()
        self.resource = None
        self.descriptor = None
        self.green_context = None
        self.raw_stream = None
        self.stream: torch.cuda.Stream | None = None
        self.ready = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.active = False
        self.closed = False
        self.actual_sm_count: int | None = None
        self.actual_coscheduled_sm_count: int | None = None
        torch.cuda.init()
        if spec.sm_count is not None:
            available = checked(
                cu.cuDeviceGetDevResource(
                    cu.CUdevice(self.device),
                    cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
                )
            )
            params = cu.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
            params.smCount = spec.sm_count
            params.coscheduledSmCount = spec.coscheduled_sm_count
            params.preferredCoscheduledSmCount = spec.coscheduled_sm_count
            params.flags = 0
            groups, remainder = checked(
                cu.cuDevSmResourceSplit(1, available, 0, [params])
            )
            self.resource = groups[0]
            self.actual_sm_count = self.resource.sm.smCount
            self.actual_coscheduled_sm_count = self.resource.sm.smCoscheduledAlignment
            if (self.actual_sm_count, self.actual_coscheduled_sm_count) != (
                spec.sm_count,
                spec.coscheduled_sm_count,
            ):
                raise RuntimeError("CUDA did not preserve the requested SM placement")
            self.descriptor = checked(cu.cuDevResourceGenerateDesc([self.resource], 1))
            self.green_context = checked(
                cu.cuGreenCtxCreate(
                    self.descriptor,
                    cu.CUdevice(self.device),
                    cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
                )
            )
            self.raw_stream = checked(
                cu.cuGreenCtxStreamCreate(
                    self.green_context, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
                )
            )
            self.stream = torch.cuda.ExternalStream(int(self.raw_stream), self.device)

    @contextmanager
    def ordered(self):
        """Yield the launch stream and join its writes back to the caller."""
        if self.closed or self.active:
            raise RuntimeError("Placement is closed or already in use")
        parent = torch.cuda.current_stream(self.device)
        self.active = True
        try:
            if self.stream is None:
                yield parent
            else:
                self.ready.record(parent)
                self.stream.wait_event(self.ready)
                with torch.cuda.stream(self.stream):
                    yield self.stream
                    self.done.record(self.stream)
                parent.wait_event(self.done)
        finally:
            self.active = False

    def launch(self, payload, parameters, **arguments):
        """Launch the saved GDN algorithm with this experiment's CTA limit."""
        from tokenspeed_kernel.ops.attention.gdn.gluon_nvidia import (
            launch_gdn_replay_commit_persistent,
        )

        with self.ordered():
            return launch_gdn_replay_commit_persistent(
                payload,
                parameters,
                max_ctas=self.spec.max_ctas,
                num_buffers=2,
                num_warps=8,
                max_registers=168,
                **arguments,
            )

    def metadata(self):
        return {
            "name": self.spec.name,
            "max_ctas": self.spec.max_ctas,
            "partition_sms": self.actual_sm_count,
            "coscheduled_sms": self.actual_coscheduled_sm_count,
            "green_context": (
                int(self.green_context) if self.green_context is not None else None
            ),
            "stream": self.stream.cuda_stream if self.stream is not None else None,
        }

    def close(self):
        """Release resources after all graph owners referencing them are gone."""
        if self.closed:
            return
        if self.active:
            raise RuntimeError("Cannot close an active placement")
        if self.raw_stream is not None:
            checked(cu.cuStreamSynchronize(self.raw_stream))
            checked(cu.cuStreamDestroy(self.raw_stream))
            self.raw_stream = None
            self.stream = None
        if self.green_context is not None:
            checked(cu.cuGreenCtxDestroy(self.green_context))
            self.green_context = None
        self.closed = True
