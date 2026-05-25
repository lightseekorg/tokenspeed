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

import atexit
import os
import re
from dataclasses import dataclass
from multiprocessing import shared_memory

import torch
import torch.distributed as dist

from tokenspeed.runtime.moe import eplb_logger

_SHM_ALIGN_BYTES = 256
_OWNED_SHM_NAMES: set[str] = set()


@dataclass
class _CacheEntry:
    layer: torch.nn.Module
    params: dict[str, torch.Tensor]


def _iter_moe_layers(model):
    from tokenspeed.runtime.layers.moe.layer import MoELayer

    for module in model.modules():
        if isinstance(module, MoELayer):
            yield module


def _pin_or_plain(tensor: torch.Tensor) -> torch.Tensor:
    try:
        return tensor.pin_memory()
    except RuntimeError as exc:
        eplb_logger.warning("host_cache.pin_fallback reason=%s", str(exc))
        return tensor


def _iter_expert_dim_params(layer) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    num_local = int(layer.num_local_experts)
    for name, param in layer.named_parameters(recurse=False):
        if name == "correction_bias" or "shared_experts" in name:
            continue
        data = param.detach()
        if data.ndim > 0 and data.shape[0] == num_local:
            out[name] = data
    return out


def _layer_ep_size(layer, server_args) -> int:
    if hasattr(layer, "ep_size"):
        return int(layer.ep_size)
    mapping = getattr(server_args, "mapping", None)
    moe_mapping = getattr(mapping, "moe", None)
    return int(getattr(moe_mapping, "ep_size", 1))


def _loaded_logical_range_for_rank(layer, server_args, ep_rank: int) -> tuple[int, int]:
    num_local_physical = int(layer.num_local_experts)
    ep_size = _layer_ep_size(layer, server_args)
    num_physical = int(getattr(layer, "num_experts", num_local_physical * ep_size))
    redundant = int(getattr(layer, "ep_num_redundant_experts", 0) or 0)
    num_logical = num_physical - redundant
    if num_logical <= 0:
        return int(ep_rank) * num_local_physical, num_local_physical
    if num_logical % ep_size != 0:
        raise ValueError(
            "HostWeightCache requires logical experts to be evenly sharded "
            f"across EP ranks: {num_logical=} {ep_size=}"
        )
    num_loaded_per_rank = num_logical // ep_size
    if num_loaded_per_rank > num_local_physical:
        raise ValueError(
            "HostWeightCache saw fewer local physical slots than checkpoint "
            f"experts: {num_loaded_per_rank=} {num_local_physical=}"
        )
    return int(ep_rank) * num_loaded_per_rank, num_loaded_per_rank


def _loaded_logical_range(layer, server_args) -> tuple[int, int]:
    ep_rank = int(getattr(layer, "ep_rank", 0))
    return _loaded_logical_range_for_rank(layer, server_args, ep_rank)


def _dist_barrier(pg) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=pg)


def _align_size(size: int) -> int:
    return (int(size) + _SHM_ALIGN_BYTES - 1) // _SHM_ALIGN_BYTES * _SHM_ALIGN_BYTES


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _sanitize_shm_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]", "_", value)
    return value[:180]


def _shared_base_name(server_args, ep_size: int) -> str:
    explicit = getattr(server_args, "eplb_shared_memory_name", None)
    if explicit:
        return _sanitize_shm_name(str(explicit))
    mapping = getattr(server_args, "mapping", None)
    moe_mapping = getattr(mapping, "moe", None)
    ep_group = getattr(moe_mapping, "ep_group", None)
    if ep_group is not None:
        group_token = "_".join(str(int(rank)) for rank in ep_group)
    else:
        group_token = "default"
    port = getattr(server_args, "port", None)
    if port is None:
        port = getattr(server_args, "base_port", None)
    if port is None:
        port = os.getppid()
    return _sanitize_shm_name(f"tokenspeed_eplb_{port}_g{group_token}_ep{int(ep_size)}")


def _shared_memory_name(base_name: str, layer_id: int, owner_rank: int) -> str:
    return _sanitize_shm_name(f"{base_name}_l{int(layer_id)}_r{int(owner_rank)}")


def _create_shared_memory(name: str, size: int) -> shared_memory.SharedMemory:
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=int(size))
        _OWNED_SHM_NAMES.add(getattr(shm, "_name", name))
        return shm
    except FileExistsError:
        eplb_logger.warning("host_cache.shm_stale_unlink name=%s", name)
        stale = shared_memory.SharedMemory(name=name)
        stale.close()
        stale.unlink()
        shm = shared_memory.SharedMemory(name=name, create=True, size=int(size))
        _OWNED_SHM_NAMES.add(getattr(shm, "_name", name))
        return shm


def _unregister_imported_shm(shm: shared_memory.SharedMemory) -> None:
    # Python versions differ on whether attaching to an existing SharedMemory
    # block is tracked. Calling resource_tracker.unregister unconditionally can
    # print noisy KeyError traces from the tracker process, so cleanup relies on
    # the owning rank unlinking its own blocks.
    return


class HostWeightCache:
    def __init__(self):
        self._entries: dict[tuple[int, int], _CacheEntry] = {}
        self._layers: dict[int, torch.nn.Module] = {}
        self._bytes = 0
        self._owned_bytes = 0
        self._owned_shms: list[shared_memory.SharedMemory] = []
        self._imported_shms: list[shared_memory.SharedMemory] = []
        self._release_registered = False

    @classmethod
    def from_model(
        cls,
        model,
        server_args,
        *,
        shared_pg=None,
        rank: int | None = None,
        ep_size: int | None = None,
        share_across_ep: bool | None = None,
    ):
        cache = cls()
        layers = list(_iter_moe_layers(model))
        if rank is None:
            mapping = getattr(server_args, "mapping", None)
            moe_mapping = getattr(mapping, "moe", None)
            rank = int(getattr(moe_mapping, "ep_rank", 0))
        if ep_size is None:
            mapping = getattr(server_args, "mapping", None)
            moe_mapping = getattr(mapping, "moe", None)
            ep_size = int(getattr(moe_mapping, "ep_size", 1))
        if share_across_ep is None:
            share_across_ep = (
                bool(getattr(server_args, "enable_eplb", False))
                and getattr(server_args, "eplb_relocate_strategy", "host_first")
                == "host_first"
                and int(ep_size) > 1
            )

        if share_across_ep and int(ep_size) > 1:
            if shared_pg is None or not (dist.is_available() and dist.is_initialized()):
                eplb_logger.warning(
                    "host_cache.shared_disabled reason=no_control_pg rank=%s ep_size=%s",
                    rank,
                    ep_size,
                )
                cache._store_local_layers(layers, server_args)
            else:
                cache._store_shared_layers(
                    layers,
                    server_args,
                    rank=int(rank),
                    ep_size=int(ep_size),
                    shared_pg=shared_pg,
                )
        else:
            cache._store_local_layers(layers, server_args)

        if cache._owned_bytes:
            eplb_logger.debug(
                "HostWeightCache built: %d (layer,expert) entries, %.2f GB mapped, %.2f GB owned",
                len(cache._entries),
                cache._bytes / 2**30,
                cache._owned_bytes / 2**30,
            )
        else:
            eplb_logger.debug(
                "HostWeightCache built: %d (layer,expert) entries, %.2f GB",
                len(cache._entries),
                cache._bytes / 2**30,
            )
        return cache

    def _store_local_layers(self, layers, server_args) -> None:
        for layer in layers:
            lid = int(layer.layer_index)
            logical_start, num_loaded = _loaded_logical_range(layer, server_args)
            for local_slot in range(num_loaded):
                self._store(lid, logical_start + local_slot, layer, local_slot)

    def _store_shared_layers(
        self,
        layers,
        server_args,
        *,
        rank: int,
        ep_size: int,
        shared_pg,
    ) -> None:
        base_name = _shared_base_name(server_args, ep_size)
        for layer in layers:
            self._store_shared_owner_layer(layer, server_args, base_name, rank)
        _dist_barrier(shared_pg)
        for owner_rank in range(ep_size):
            if owner_rank == rank:
                continue
            for layer in layers:
                self._import_shared_owner_layer(
                    layer, server_args, base_name, owner_rank
                )
        _dist_barrier(shared_pg)
        if not self._release_registered:
            atexit.register(self.release)
            self._release_registered = True

    def _store_shared_owner_layer(
        self, layer, server_args, base_name: str, owner_rank: int
    ) -> None:
        lid = int(layer.layer_index)
        params = _iter_expert_dim_params(layer)
        logical_start, num_loaded = _loaded_logical_range_for_rank(
            layer, server_args, owner_rank
        )
        total_size = 0
        for _, data in sorted(params.items()):
            expert_size = _tensor_bytes(data[0])
            total_size += _align_size(expert_size) * int(num_loaded)
        if total_size <= 0:
            return

        shm = _create_shared_memory(
            _shared_memory_name(base_name, lid, owner_rank), total_size
        )
        self._owned_shms.append(shm)
        entry_params: dict[int, dict[str, torch.Tensor]] = {
            logical_start + local_slot: {} for local_slot in range(num_loaded)
        }
        offset = 0
        for name, data in sorted(params.items()):
            expert_shape = tuple(data.shape[1:])
            expert_numel = int(data[0].numel())
            expert_size = _tensor_bytes(data[0])
            aligned_size = _align_size(expert_size)
            for local_slot in range(num_loaded):
                host = torch.frombuffer(
                    shm.buf,
                    dtype=data.dtype,
                    count=expert_numel,
                    offset=offset,
                ).view(expert_shape)
                host.copy_(data[local_slot].detach().to("cpu", non_blocking=False))
                logical_id = logical_start + local_slot
                entry_params[logical_id][name] = host
                offset += aligned_size
                self._bytes += expert_size
                self._owned_bytes += expert_size

        for logical_id, host_params in entry_params.items():
            self._entries[(lid, logical_id)] = _CacheEntry(
                layer=layer, params=host_params
            )
        self._layers[lid] = layer

    def _import_shared_owner_layer(
        self, layer, server_args, base_name: str, owner_rank: int
    ) -> None:
        lid = int(layer.layer_index)
        params = _iter_expert_dim_params(layer)
        logical_start, num_loaded = _loaded_logical_range_for_rank(
            layer, server_args, owner_rank
        )
        shm_name = _shared_memory_name(base_name, lid, owner_rank)
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError as exc:
            raise RuntimeError(f"EPLB shared host cache missing {shm_name}") from exc
        _unregister_imported_shm(shm)
        self._imported_shms.append(shm)
        entry_params: dict[int, dict[str, torch.Tensor]] = {
            logical_start + local_slot: {} for local_slot in range(num_loaded)
        }
        offset = 0
        for name, data in sorted(params.items()):
            expert_shape = tuple(data.shape[1:])
            expert_numel = int(data[0].numel())
            expert_size = _tensor_bytes(data[0])
            aligned_size = _align_size(expert_size)
            for local_slot in range(num_loaded):
                host = torch.frombuffer(
                    shm.buf,
                    dtype=data.dtype,
                    count=expert_numel,
                    offset=offset,
                ).view(expert_shape)
                logical_id = logical_start + local_slot
                entry_params[logical_id][name] = host
                offset += aligned_size
                self._bytes += expert_size

        for logical_id, host_params in entry_params.items():
            self._entries[(lid, logical_id)] = _CacheEntry(
                layer=layer, params=host_params
            )
        self._layers[lid] = layer

    def _store(self, layer_id: int, logical_id: int, layer, local_slot: int) -> None:
        params = {}
        for name, data in _iter_expert_dim_params(layer).items():
            host = data[local_slot].detach().to("cpu", non_blocking=False).clone()
            host = _pin_or_plain(host)
            params[name] = host
            nbytes = host.numel() * host.element_size()
            self._bytes += nbytes
            self._owned_bytes += nbytes
        self._entries[(layer_id, logical_id)] = _CacheEntry(layer=layer, params=params)
        self._layers[layer_id] = layer

    def has(self, layer_id: int, logical_id: int) -> bool:
        return (layer_id, logical_id) in self._entries

    def get_expert_dim_params(
        self, layer_id: int, logical_id: int
    ) -> dict[str, torch.Tensor]:
        return self._entries[(layer_id, logical_id)].params

    def layer_handle(self, layer_id: int):
        return self._layers[layer_id]

    def expert_dim_params(self, layer_id: int) -> dict[str, torch.Tensor]:
        return _iter_expert_dim_params(self.layer_handle(layer_id))

    def release(self) -> None:
        self._entries.clear()
        self._layers.clear()
        for shm in self._imported_shms:
            try:
                shm.close()
            except BufferError:
                pass
        self._imported_shms.clear()
        for shm in self._owned_shms:
            _OWNED_SHM_NAMES.discard(getattr(shm, "_name", ""))
            try:
                shm.close()
            except BufferError:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._owned_shms.clear()
