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

"""Decide where a fused decoder-layer kernel may serve a model's decode steps.

A fused decoder layer runs a whole decoder layer, its cross-GPU exchanges
included, in one launch that keeps one workgroup resident on every compute
unit while its workgroups wait on each other. That launch is correct only
when nothing else takes a compute unit while it is dispatched, and only when
every rank of the group runs the same sequence of fused launches. This module
holds the model-independent half of that decision:

* ``deployment_decline_reasons`` lists the server settings under which some
  other kernel can run beside a fused launch;
* ``agree_served_shapes`` turns each rank's row shapes and local refusals into
  one served set that every rank of the group holds, so a decline on one rank
  declines the shape everywhere.

The model chooses the fused layout per forward from the agreed set only,
which depends on the padded batch shape and nothing rank-local, so a captured
graph records the same choice on every rank. See
``docs/design/monokernel.md``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch.distributed as dist

# Variables that restrict which compute units a queue may use.
_CU_MASK_VARIABLES = ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")


@dataclass(frozen=True)
class FusedLayerDeployment:
    """The server settings that decide whether a fused launch can be resident.

    Attributes:
        host_kv_store: Whether the host KV store (Host L2) is on.
        kvstore_io_backend: How it copies pages (``--kvstore-io-backend``).
        expert_load_recorder: ``--expert-distribution-recorder-mode``.
        memory_saver: Whether the memory saver may release device memory.
        forward_side_stream: Whether the model forks a side stream inside its
            forward (a captured fork can run beside a fused launch).
        environ: The process environment.
    """

    host_kv_store: bool
    kvstore_io_backend: str
    expert_load_recorder: str | None
    memory_saver: bool
    forward_side_stream: bool
    environ: Mapping[str, str]


def current_deployment(
    server_args: Mapping[str, object], *, forward_side_stream: bool
) -> FusedLayerDeployment:
    """This process's deployment, from the server arguments and environment.

    Args:
        server_args: The published server arguments
            (``tokenspeed.runtime.utils.env.global_server_args_dict``).
        forward_side_stream: Whether the calling model forks a side stream
            inside its forward.
    """
    args = server_args
    return FusedLayerDeployment(
        host_kv_store=bool(args["enable_kvstore"]),
        kvstore_io_backend=args["kvstore_io_backend"],
        expert_load_recorder=args["expert_distribution_recorder_mode"],
        memory_saver=bool(args["enable_memory_saver"]),
        forward_side_stream=forward_side_stream,
        environ=dict(os.environ),
    )


def deployment_decline_reasons(deployment: FusedLayerDeployment) -> list[str]:
    """Why fused decoder layers cannot run under ``deployment``; empty if they
    can.

    Every reason names a source of kernels that may run concurrently with a
    fused launch (which then waits on a workgroup that never becomes resident
    and fail-stops the group), or state the fused path cannot keep.
    """
    reasons = []
    if deployment.host_kv_store and deployment.kvstore_io_backend == "kernel":
        reasons.append(
            "the host KV store copies pages with kernels on side streams; "
            "serve with --kvstore-io-backend direct or --disable-kvstore"
        )
    if deployment.environ.get("HSA_ENABLE_SDMA") == "0":
        reasons.append(
            "HSA_ENABLE_SDMA=0 runs asynchronous copies as kernels on their streams"
        )
    for name in _CU_MASK_VARIABLES:
        if deployment.environ.get(name):
            reasons.append(f"{name} restricts the compute units a launch can use")
    if deployment.expert_load_recorder is not None:
        reasons.append("fused layers do not record expert loads")
    if deployment.memory_saver:
        reasons.append(
            "the memory saver can release the fused launch state apart from "
            "the communication buffers it indexes"
        )
    if deployment.forward_side_stream:
        reasons.append("the model forks a side stream inside its forward")
    return reasons


# A decode forward's row shape: (requests, tokens per request). Plain decode of
# 8 requests and a verify of one request with 8 tokens both carry 8 rows, but
# they are different shapes to a fused layer.
Shape = tuple[int, int]


@dataclass(frozen=True)
class ServedShapes:
    """The decode row shapes every rank of a group serves with a fused layer.

    Attributes:
        shapes: ``(requests, tokens per request)`` shapes served on every rank.
        declined: For each declined shape, ``"rank <r>: <reason>"`` lines,
            ``r`` being the rank in the agreeing group.
    """

    shapes: frozenset[Shape]
    declined: Mapping[Shape, tuple[str, ...]]

    def report(self, what: str) -> str:
        """The startup report: the served shapes, and why each other one is not."""

        def name(shape: Shape) -> str:
            return f"{shape[0]} x {shape[1]} rows"

        served = ", ".join(name(shape) for shape in sorted(self.shapes)) or "none"
        lines = [f"{what}: serves {served}"]
        for shape in sorted(self.declined):
            reasons = "; ".join(self.declined[shape])
            lines.append(f"  {name(shape)} declined: {reasons}")
        return "\n".join(lines)


def agree_served_shapes(
    candidates: Sequence[Shape],
    local_reasons: Mapping[Shape, Sequence[str]],
    group: dist.ProcessGroup | None,
) -> ServedShapes:
    """Agree on the decode row shapes a group serves with a fused layer.

    Args:
        candidates: ``(requests, tokens per request)`` shapes the model can run
            fused, the same on every rank.
        local_reasons: This rank's reasons to decline a shape (missing or
            empty: this rank can serve it).
        group: The CPU (gloo) group of the ranks that run each fused launch
            together, or None for a single process.

    Returns:
        The shapes no rank declined, and the reasons for the others.
    """
    local = {
        (int(requests), int(width)): tuple(local_reasons.get((requests, width), ()))
        for requests, width in sorted(candidates)
    }
    if group is None:
        gathered = [local]
    else:
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, local, group=group)
    if any(set(other) != set(local) for other in gathered):
        raise RuntimeError(
            f"fused decoder layers: ranks propose different shapes: "
            f"{[sorted(other) for other in gathered]}"
        )
    declined = {}
    for shape in local:
        lines = tuple(
            f"rank {rank}: {reason}"
            for rank, other in enumerate(gathered)
            for reason in other[shape]
        )
        if lines:
            declined[shape] = lines
    return ServedShapes(frozenset(set(local) - set(declined)), declined)
