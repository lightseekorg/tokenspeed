# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
#
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

"""Executes a committed expert placement: move slot weights, switch the tables.

Runs on the forward thread (``DeviceHandle.apply_expert_placement``), inside
the execution stream, one chunk of layers per call. Per layer:

1. P2P: receive every incoming slot into the **staging buffer** and send
   every outgoing slot from its live tensor, in one ``batch_isend_irecv``
   over the EP device group, then wait. The plan orders both ends by logical
   expert id, so the pairs match without any further agreement. Live slots
   are not written until every P2P completed, so the sends need no clone.
2. Same-GPU copies live -> staging (an expert moving between two local slots
   may be part of a cycle).
3. Live writes: staging -> live for the received and same-GPU slots, then
   free-riders live -> live in slot order (the source landed first).
4. The layer's routing tables switch in place
   (``ExpertLocationMetadata.switch_layer``; the host rows of the whole chunk
   were prepared once up front), so a forward never sees a layer whose
   tables and slots disagree.

The staging buffer is one layer's worth of slot tensors, reserved at startup
before the KV arena is sized (a transient allocation inside the op could not
be guaranteed under a cache-first memory budget). Slot tensors are the
*processed* parameters (quantized weights and scales included): the rebalance
moves bytes between identical slots and never re-runs weight processing.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.moe.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
    get_global_expert_location_metadata,
)
from tokenspeed.runtime.moe.expert_rebalance import ExpertRebalanceSpecs, SlotMoves

__all__ = ["ExpertLocationUpdater", "build_expert_location_updater"]

logger = logging.getLogger(__name__)


class ExpertLocationUpdater:
    """Moves expert weights between slots and switches the routing tables in place.

    Args:
        placement: The live placement whose tables are rewritten.
        weights_of_layer: Each MoE layer's slot tensors, every one
            ``[num_local, ...]`` in slot order (the processed parameters). A
            layer absent from the mapping (a pipeline stage that does not hold
            it) still has its tables switched.
        process_group: The EP device process group the P2P rides; its issue
            order is one sequence on every rank (the MoE kernels' group).
        peer_ranks: The global rank of every EP rank, in EP-rank order.
        ep_rank: This rank's position among ``peer_ranks``.
        all_to_all_ep: Whether each rank routes only its own tokens (the load
            must then be summed over the EP group).
        num_groups: The model's expert groups, or None.
        num_nodes: Nodes the job spans.
    """

    def __init__(
        self,
        placement: ExpertLocationMetadata,
        weights_of_layer: Mapping[int, Sequence[torch.Tensor]],
        *,
        process_group: dist.ProcessGroup,
        peer_ranks: Sequence[int],
        ep_rank: int,
        all_to_all_ep: bool,
        num_groups: int | None,
        num_nodes: int,
    ) -> None:
        if len(peer_ranks) != placement.ep_size or ep_rank != placement.ep_rank:
            raise ValueError(
                f"EP group of {len(peer_ranks)} ranks (rank {ep_rank}) does not "
                f"match the placement's ep_size={placement.ep_size}, "
                f"ep_rank={placement.ep_rank}"
            )
        self._placement = placement
        self._weights_of_layer = {
            int(layer_id): tuple(tensors)
            for layer_id, tensors in weights_of_layer.items()
        }
        self._process_group = process_group
        self._peer_ranks = tuple(int(rank) for rank in peer_ranks)
        self._ep_rank = ep_rank
        self.specs = ExpertRebalanceSpecs(
            num_layers=placement.num_layers,
            num_logical_experts=placement.num_logical_experts,
            num_physical_experts=placement.num_physical_experts,
            ep_size=placement.ep_size,
            ep_rank=placement.ep_rank,
            ep_rank_nodes=placement.ep_rank_nodes,
            all_to_all_ep=all_to_all_ep,
            num_groups=num_groups,
            num_nodes=num_nodes,
        )
        self._staging = self._reserve_staging()

    def _reserve_staging(self) -> tuple[torch.Tensor, ...]:
        """One layer's worth of slot tensors; every layer must share their layout."""
        if not self._weights_of_layer:
            raise ValueError("the model exposes no routed expert weights to rebalance")
        num_local = self._placement.num_local_physical_experts
        reference: tuple[torch.Tensor, ...] | None = None
        for layer_id, tensors in sorted(self._weights_of_layer.items()):
            if not 0 <= layer_id < self._placement.num_layers:
                raise ValueError(
                    f"layer {layer_id} is outside the placement's "
                    f"{self._placement.num_layers} layers"
                )
            for index, tensor in enumerate(tensors):
                if tensor.ndim < 1 or tensor.shape[0] != num_local:
                    raise ValueError(
                        f"layer {layer_id} tensor {index} has shape "
                        f"{tuple(tensor.shape)}; slot tensors are "
                        f"[{num_local}, ...], one row per local slot"
                    )
                if not tensor[0].is_contiguous():
                    raise ValueError(
                        f"layer {layer_id} tensor {index} slots are not contiguous; "
                        "P2P moves whole slots"
                    )
            if reference is None:
                reference = tensors
                continue
            if len(tensors) != len(reference) or any(
                t.shape != r.shape or t.dtype != r.dtype or t.device != r.device
                for t, r in zip(tensors, reference)
            ):
                raise ValueError(
                    f"layer {layer_id}'s slot tensors differ in layout from the "
                    "first MoE layer's; the staging buffer is shared by every layer"
                )
        assert reference is not None
        return tuple(torch.empty_like(tensor) for tensor in reference)

    @property
    def staging_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._staging)

    def prewarm(self) -> None:
        """Exchange one element with every peer so the first rebalance pays no channel setup."""
        ep_size = len(self._peer_ranks)
        if ep_size == 1:
            return
        device = self._staging[0].device
        send = torch.zeros(ep_size, dtype=torch.int32, device=device)
        recv = torch.empty(ep_size, dtype=torch.int32, device=device)
        ops = []
        for rank in range(ep_size):
            if rank == self._ep_rank:
                continue
            ops.append(self._p2p(dist.isend, send[rank], rank))
            ops.append(self._p2p(dist.irecv, recv[rank], rank))
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    def _p2p(self, op, tensor: torch.Tensor, ep_rank: int) -> dist.P2POp:
        return dist.P2POp(
            op, tensor, peer=self._peer_ranks[ep_rank], group=self._process_group
        )

    def apply(
        self,
        layer_ids: Sequence[int],
        new_rows: torch.Tensor,
        moves_by_layer: Mapping[int, SlotMoves],
    ) -> None:
        """Move the weights of ``layer_ids`` and switch each layer's tables.

        Args:
            layer_ids: The chunk's layers, in order.
            new_rows: ``[len(layer_ids), num_physical]`` host rows of the
                committed placement.
            moves_by_layer: This rank's ``SlotMoves`` per layer.
        """
        # The host side of the switch (inverse rows, valid counts, static map
        # rows) is derived once for the chunk; each layer then switches with
        # a few device copies right after its weights landed.
        prepared = self._placement.prepare_layers(layer_ids, new_rows)
        for index, layer_id in enumerate(prepared.layer_ids):
            moves = moves_by_layer[layer_id]
            tensors = self._weights_of_layer.get(layer_id)
            if tensors is not None and not moves.is_empty:
                self._move_weights(layer_id, tensors, moves)
            # The table switch follows the layer's weights on the same stream.
            self._placement.switch_layer(prepared, index)

    def _move_weights(
        self, layer_id: int, tensors: Sequence[torch.Tensor], moves: SlotMoves
    ) -> None:
        staging = self._staging
        ops: list[dist.P2POp] = []
        for dst_slot, src_rank in moves.recv:
            for buffer in staging:
                ops.append(self._p2p(dist.irecv, buffer[dst_slot], src_rank))
        for src_slot, dst_rank in moves.send:
            for tensor in tensors:
                ops.append(self._p2p(dist.isend, tensor[src_slot], dst_rank))
        if ops:
            # Watchdog line: a hang in the batch below names the plan it ran.
            digest = hashlib.blake2b(
                repr((moves.recv, moves.send)).encode(), digest_size=4
            ).hexdigest()
            logger.info(
                f"Expert rebalance layer {layer_id}: {len(moves.recv)} slots in, "
                f"{len(moves.send)} slots out, {len(moves.local_copy)} same-GPU, "
                f"{len(moves.free_rider)} free-riders (plan {digest})"
            )
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        for dst_slot, src_slot in moves.local_copy:
            for tensor, buffer in zip(tensors, staging):
                buffer[dst_slot].copy_(tensor[src_slot])
        for dst_slot, _ in (*moves.recv, *moves.local_copy):
            for tensor, buffer in zip(tensors, staging):
                tensor[dst_slot].copy_(buffer[dst_slot])
        for dst_slot, src_slot in moves.free_rider:
            for tensor in tensors:
                tensor[dst_slot].copy_(tensor[src_slot])


def _all_to_all_ep(model: torch.nn.Module) -> bool:
    """Whether the model's MoE layers own all-to-all dispatch (uniform over layers)."""
    flags = {
        module.supports_all_to_all_ep
        for module in model.modules()
        if isinstance(module, MoELayer)
    }
    if len(flags) != 1:
        raise ValueError(
            "the model's MoE layers disagree on all-to-all EP"
            if flags
            else "the model has no MoE layers to rebalance"
        )
    return flags.pop()


def build_expert_location_updater(
    model: torch.nn.Module, model_config, server_args
) -> ExpertLocationUpdater:
    """Reserve the staging buffer and warm the P2P channels for ``--enable-eplb``.

    Called by the target ``ModelRunner`` right after ``load_model`` and before
    the cache profile sizes the KV arena. The EP device and gloo groups are
    created here if the MoE mapping did not already (idempotent).
    """
    placement = get_global_expert_location_metadata()
    if placement is None or placement.physical_load is None:
        raise ValueError(
            "--enable-eplb needs the expert placement with load recording "
            "(--expert-distribution-recorder-mode stat)"
        )
    mapping = server_args.mapping
    pg_manager.init_process_group(mapping.moe.ep_group)
    updater = ExpertLocationUpdater(
        placement,
        model.routed_experts_weights_of_layer,
        process_group=pg_manager.get_device_process_group(mapping.moe.ep_group),
        peer_ranks=mapping.moe.ep_group,
        ep_rank=mapping.moe.ep_rank,
        all_to_all_ep=_all_to_all_ep(model),
        num_groups=ModelConfigForExpertLocation.from_model_config(
            model_config
        ).num_groups,
        num_nodes=mapping.nnodes,
    )
    updater.prewarm()
    logger.info(
        f"Expert rebalance: staging buffer of {updater.staging_bytes / 2**20:.1f} MiB "
        f"reserved; P2P channels warmed over {len(mapping.moe.ep_group)} EP ranks"
    )
    return updater
