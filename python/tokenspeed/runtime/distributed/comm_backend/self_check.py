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

"""Startup self-check of the in-switch (NVLS multimem) all-reduce.

``--batch-invariant-collectives`` promises one association order per
reduction. The ordered fold is that by construction; the in-switch reduction
is that by measurement -- the switch's order is fixed for a fixed issuer but
nothing in software pins it -- so a deployment that will use it verifies the
claim on its own groups before it serves anything ("verify, don't trust"):

1. **Run stability** -- the same payload reduces to the same bits across
   ``repetitions`` launches. A difference is a fault and refuses startup.
2. **Batch invariance** -- a row's sum does not depend on how many rows ride
   along (the full payload and its first half agree on the shared rows). A
   difference refuses startup likewise.
3. **One function per kind** -- every group of a kind (every attention TP
   group, say) reduces the identical payload to identical bits and every
   rank takes the same route, so a request's bits cannot depend on the
   replica serving it. The switch's order is a property of the GPU set, and
   two sets of three or more GPUs have been measured to differ, so this is
   not a fault but a topology: the kind's groups are pinned to the ordered
   fold (``AutoBackend.pin_ordered_fold``), on every rank alike, and the
   deployment stays bitwise at the fold's speed for that kind.

The payload (``multimem_probe_payload``) is ill-conditioned by construction,
so two association orders disagree on a large fraction of its elements
rather than on a few per ten million. Runs once per deployment on the groups
the route sends to the switch, in well under a second after the kernels'
first compile.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.triton import multimem_probe_payload

from tokenspeed.runtime.distributed.comm_backend.auto import (
    AutoBackend,
    Collective,
    Route,
)
from tokenspeed.runtime.distributed.comm_backend.base import Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

logger = logging.getLogger(__name__)

# Rows of the probe payload per group; enough for the probe to separate
# orders on thousands of elements per row width, small enough to stay cheap.
_PROBE_ROWS = 32
_PROBE_SEED = 0x5EED


class MultimemSelfCheckError(RuntimeError):
    """The in-switch reduction failed its startup verification."""


def _world_agrees(verdict: bool, world_group: Group, device: torch.device) -> bool:
    """MIN-reduce a per-rank verdict over the world so every rank decides alike."""
    vote = torch.tensor([int(verdict)], dtype=torch.int32, device=device)
    dist.all_reduce(
        vote,
        op=dist.ReduceOp.MIN,
        group=pg_manager.get_process_group("nccl", world_group),
    )
    return bool(vote.item())


def _gather_over_world(tensor: torch.Tensor, world_group: Group) -> list[torch.Tensor]:
    chunks = [torch.empty_like(tensor) for _ in world_group]
    dist.all_gather(
        chunks,
        tensor.contiguous(),
        group=pg_manager.get_process_group("nccl", world_group),
    )
    return chunks


def verify_multimem_all_reduce(
    backend: AutoBackend,
    *,
    groups: Sequence[tuple[str, Group]],
    world_group: Group,
    rank: int,
    hidden_size: int,
    device: torch.device,
    repetitions: int,
) -> list[tuple[str, Route]]:
    """Verify the in-switch all-reduce on every group the route sends to it.

    Args:
        backend: The communication backend whose ``route`` decides and whose
            ``pin_ordered_fold`` records a kind that is not one function.
        groups: ``(kind, group)`` pairs to check, e.g. ``("attention TP",
            mapping.attn.tp_group)``; every rank passes the same kinds in the
            same order (the checks are collective). Groups of one rank and
            kinds whose route is not the switch are skipped.
        world_group: Every rank of the deployment, for the agreement checks.
        rank: This process's global rank.
        hidden_size: Row width the probe reduces (the model's hidden size, the
            width the serving all-reduces have).
        device: This rank's device.
        repetitions: Launches whose results must agree bitwise.

    Returns:
        ``(kind, route)`` for every kind that was headed for the switch:
        ``Route.MULTIMEM`` once verified, ``Route.ORDERED_FOLD`` once pinned
        there. For the startup log.

    Raises:
        MultimemSelfCheckError: A repetition or a shorter payload changed the
            bits. The message names the kind, the group and what differed.
    """
    probe = torch.empty((_PROBE_ROWS, hidden_size), dtype=torch.bfloat16, device=device)
    outcome: list[tuple[str, Route]] = []
    decided: dict[Group, Route] = {}
    for kind, group in groups:
        if group in decided:
            # The same ranks under another name (dense TP is often the
            # attention TP group): one group, one decision.
            outcome.append((kind, decided[group]))
            continue
        on_switch = (
            len(group) > 1
            and backend.route(Collective.ALL_REDUCE, probe, group) is Route.MULTIMEM
        )
        # Every rank must take the same route for a kind, or two replicas
        # would reduce the same request differently: a kind the switch does
        # not reach everywhere stays on the fold everywhere.
        if not _world_agrees(on_switch, world_group, device):
            if on_switch:
                backend.pin_ordered_fold(group)
                logger.warning(
                    f"batch-invariant all-reduce: the {kind} group {group} is "
                    "kept on the ordered fold because multicast does not reach "
                    "every group of that kind (or another is a group of one)"
                )
                decided[group] = Route.ORDERED_FOLD
                outcome.append((kind, Route.ORDERED_FOLD))
            continue
        if not on_switch:
            continue
        payload = multimem_probe_payload(
            group.index(rank), len(group), _PROBE_ROWS, hidden_size, device, _PROBE_SEED
        )
        reference = backend.all_reduce(payload.clone(), group)
        for repetition in range(1, repetitions):
            again = backend.all_reduce(payload.clone(), group)
            if not torch.equal(again, reference):
                differing = int((again != reference).sum())
                raise MultimemSelfCheckError(
                    f"the in-switch all-reduce over the {kind} group {group} is "
                    f"not run-stable: repetition {repetition} of {repetitions} "
                    f"differs from the first in {differing} of {reference.numel()} "
                    "elements. Launch with --force-deterministic-rsag to keep "
                    "every reduction on the ordered fold."
                )
        half = backend.all_reduce(payload[: _PROBE_ROWS // 2].clone(), group)
        if not torch.equal(half, reference[: _PROBE_ROWS // 2]):
            differing = int((half != reference[: _PROBE_ROWS // 2]).sum())
            raise MultimemSelfCheckError(
                f"the in-switch all-reduce over the {kind} group {group} is not "
                f"batch-invariant: reducing {_PROBE_ROWS // 2} rows changes "
                f"{differing} of {half.numel()} elements against reducing "
                f"{_PROBE_ROWS}. Launch with --force-deterministic-rsag to keep "
                "every reduction on the ordered fold."
            )
        # Every group of this kind reduced the identical payload. The gathered
        # results are the same list on every rank, so "all equal mine" is one
        # verdict for the world.
        peers = _gather_over_world(reference, world_group)
        differing = [
            world_group[peer]
            for peer, result in enumerate(peers)
            if not torch.equal(result, reference)
        ]
        if differing:
            backend.pin_ordered_fold(group)
            logger.warning(
                f"batch-invariant all-reduce: the {kind} groups reduce the same "
                f"payload to different bits (rank {rank}'s group {group} against "
                f"ranks {differing}: {int((peers[world_group.index(differing[0])] != reference).sum())} "
                f"of {reference.numel()} probe elements); the switch's order is a "
                "property of the GPU set, so this kind is kept on the ordered fold"
            )
            decided[group] = Route.ORDERED_FOLD
            outcome.append((kind, Route.ORDERED_FOLD))
            continue
        decided[group] = Route.MULTIMEM
        outcome.append((kind, Route.MULTIMEM))
    return outcome


__all__ = ["MultimemSelfCheckError", "verify_multimem_all_reduce"]
