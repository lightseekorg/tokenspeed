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

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist

from tokenspeed.runtime.moe import eplb_logger


@dataclass(frozen=True)
class TransferPlanEntry:
    src_rank: int
    src_local_slot: int
    dst_rank: int
    dst_local_slot: int
    logical_expert_id: int

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.src_rank,
            self.src_local_slot,
            self.dst_rank,
            self.dst_local_slot,
            self.logical_expert_id,
        )


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _physical_owner(physical_id: int, num_local: int) -> tuple[int, int]:
    rank, local = divmod(int(physical_id), int(num_local))
    return rank, local


def _find_physical_slot(metadata, layer_idx: int, logical_id: int) -> int:
    row = metadata.physical_to_logical_map_cpu[int(layer_idx)].tolist()
    for physical_id, mapped_logical in enumerate(row):
        if int(mapped_logical) == int(logical_id):
            return int(physical_id)
    raise KeyError(f"logical expert {logical_id} is not present in layer {layer_idx}")


class WeightRelocator:
    def __init__(
        self,
        host_cache,
        layer_states: dict[int, object],
        *,
        rank: int,
        ep_size: int,
        strategy: str = "host_first",
        eplb_pg=None,
    ):
        self.host_cache = host_cache
        self.layer_states = layer_states
        self.rank = int(rank)
        self.ep_size = int(ep_size)
        self.strategy = strategy
        self.eplb_pg = eplb_pg
        self._ready_layers: deque[int] = deque()

    @staticmethod
    def build_transfer_plan(
        old_metadata,
        new_metadata,
        *,
        layer_idx: int,
        ep_size: int,
        num_local: int,
    ) -> list[tuple[int, int, int, int, int]]:
        del ep_size
        layer_idx = int(layer_idx)
        old_row = old_metadata.physical_to_logical_map_cpu[layer_idx].tolist()
        new_row = new_metadata.physical_to_logical_map_cpu[layer_idx].tolist()
        out: list[TransferPlanEntry] = []
        for dst_physical, logical_id in enumerate(new_row):
            logical_id = int(logical_id)
            if logical_id < 0:
                continue
            if logical_id == int(old_row[dst_physical]):
                continue
            src_physical = _find_physical_slot(old_metadata, layer_idx, logical_id)
            src_rank, src_local = _physical_owner(src_physical, num_local)
            dst_rank, dst_local = _physical_owner(dst_physical, num_local)
            out.append(
                TransferPlanEntry(
                    src_rank=src_rank,
                    src_local_slot=src_local,
                    dst_rank=dst_rank,
                    dst_local_slot=dst_local,
                    logical_expert_id=logical_id,
                )
            )
        return [entry.as_tuple() for entry in out]

    def estimate_transfer_stats(
        self, old_metadata, new_metadata, layer_ids: list[int]
    ) -> dict[str, int]:
        transfer_entries = 0
        local_dst_entries = 0
        local_src_entries = 0
        expected_h2d_bytes = 0
        for layer_id in layer_ids:
            layer = self.host_cache.layer_handle(int(layer_id))
            params = self.host_cache.expert_dim_params(int(layer_id))
            num_local = int(layer.num_local_experts)
            plan = self.build_transfer_plan(
                old_metadata,
                new_metadata,
                layer_idx=int(layer_id),
                ep_size=self.ep_size,
                num_local=num_local,
            )
            transfer_entries += len(plan)
            for src_rank, _src_local, dst_rank, dst_local, _logical_id in plan:
                if int(dst_rank) == self.rank:
                    local_dst_entries += 1
                    for param in params.values():
                        dst_tensor = param.detach()[int(dst_local)]
                        expected_h2d_bytes += (
                            dst_tensor.numel() * dst_tensor.element_size()
                        )
                if int(src_rank) == self.rank and int(dst_rank) != self.rank:
                    local_src_entries += 1
        return {
            "transfer_entries": transfer_entries,
            "local_dst_entries": local_dst_entries,
            "local_src_entries": local_src_entries,
            "expected_h2d_bytes": expected_h2d_bytes,
        }

    def submit(self, old_metadata, new_metadata, layer_ids: list[int]) -> list[int]:
        for layer_id in layer_ids:
            self._relocate_layer(old_metadata, new_metadata, int(layer_id))
            self._mark_layer_ready(int(layer_id))
        return self.consume_ready()

    def poll_finished_layers(self) -> list[int]:
        return self.consume_ready()

    def consume_ready(self) -> list[int]:
        out = list(self._ready_layers)
        self._ready_layers.clear()
        return out

    def drain(self) -> list[int]:
        return self.consume_ready()

    def _relocate_layer(self, old_metadata, new_metadata, layer_id: int) -> None:
        layer = self.host_cache.layer_handle(layer_id)
        params = self.host_cache.expert_dim_params(layer_id)
        num_local = int(layer.num_local_experts)
        plan = self.build_transfer_plan(
            old_metadata,
            new_metadata,
            layer_idx=layer_id,
            ep_size=self.ep_size,
            num_local=num_local,
        )
        if not plan:
            return

        state = self.layer_states.get(layer_id)
        stream_context = self._stream_context(state)
        with stream_context:
            local_sources = self._snapshot_local_sources(params, plan)
            recv_work = []
            send_work = []
            for src_rank, src_local, dst_rank, dst_local, logical_id in plan:
                if dst_rank == self.rank and src_rank == self.rank:
                    self._copy_local(params, local_sources, src_local, dst_local)
                elif dst_rank == self.rank:
                    if self.strategy == "host_first":
                        self._copy_from_host(params, layer_id, logical_id, dst_local)
                    elif _dist_ready():
                        recv_work.extend(
                            self._recv_remote(
                                params, src_rank, dst_local, layer_id, logical_id
                            )
                        )
                    else:
                        self._copy_from_host(params, layer_id, logical_id, dst_local)
            if self.strategy == "p2p":
                for src_rank, src_local, dst_rank, dst_local, logical_id in plan:
                    if src_rank == self.rank and dst_rank != self.rank:
                        if _dist_ready():
                            send_work.extend(
                                self._send_remote(
                                    local_sources,
                                    src_local,
                                    dst_rank,
                                    layer_id,
                                    logical_id,
                                )
                            )
                        else:
                            eplb_logger.debug_rate(
                                "relocator.remote_send_skipped layer=%s logical=%s",
                                layer_id,
                                logical_id,
                                every=100,
                            )
            for item in recv_work + send_work:
                item.wait()
        if state is not None:
            state.relocate_done_event.record()

    def _stream_context(self, state):
        if state is None:
            return nullcontext()
        stream = getattr(state, "aux_stream", None)
        if isinstance(stream, torch.cuda.Stream):
            return torch.cuda.stream(stream)
        return nullcontext()

    def _snapshot_local_sources(self, params, plan):
        sources: dict[tuple[str, int], torch.Tensor] = {}
        for src_rank, src_local, dst_rank, *_ in plan:
            if src_rank != self.rank:
                continue
            if dst_rank != self.rank and self.strategy != "p2p":
                continue
            for name, param in params.items():
                key = (name, src_local)
                if key not in sources:
                    sources[key] = param.detach()[src_local].clone()
        return sources

    def _copy_local(
        self, params, local_sources, src_local: int, dst_local: int
    ) -> None:
        for name, param in params.items():
            source = local_sources[(name, src_local)]
            param.detach()[dst_local].copy_(source, non_blocking=True)

    def _copy_from_host(
        self, params, layer_id: int, logical_id: int, dst_local: int
    ) -> None:
        if not self.host_cache.has(layer_id, logical_id):
            raise RuntimeError(
                f"EPLB cannot relocate layer={layer_id} logical={logical_id}: "
                "source expert is not in local/shared host cache; rebuild the "
                "host_first shared cache or use --eplb-relocate-strategy p2p"
            )
        host_params = self.host_cache.get_expert_dim_params(layer_id, logical_id)
        for name, param in params.items():
            param.detach()[dst_local].copy_(
                host_params[name].to(param.device, non_blocking=True), non_blocking=True
            )

    def _send_remote(
        self,
        local_sources,
        src_local: int,
        dst_rank: int,
        layer_id: int,
        logical_id: int,
    ) -> list[object]:
        work = []
        for param_name, src_slot in sorted(local_sources):
            if int(src_slot) != int(src_local):
                continue
            tensor = local_sources[(param_name, src_slot)]
            tag = self._tag(layer_id, logical_id, param_name)
            work.append(
                dist.isend(
                    tensor.contiguous(), dst=int(dst_rank), group=self.eplb_pg, tag=tag
                )
            )
        return work

    def _recv_remote(
        self, params, src_rank: int, dst_local: int, layer_id: int, logical_id: int
    ) -> list[object]:
        work = []
        for name, param in sorted(params.items()):
            dst_tensor = param.detach()[dst_local]
            tag = self._tag(layer_id, logical_id, name)
            work.append(
                dist.irecv(dst_tensor, src=int(src_rank), group=self.eplb_pg, tag=tag)
            )
        return work

    @staticmethod
    def _tag(layer_id: int, logical_id: int, name: str) -> int:
        return (
            int(layer_id) * 1_000_003 + int(logical_id) * 97 + hash(name)
        ) & 0x7FFFFFFF

    def _mark_layer_ready(self, layer_id: int) -> None:
        self._ready_layers.append(layer_id)
