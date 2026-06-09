from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from tokenspeed_kernel.ops.kvcache.cuda import transfer_kv_direct

from tokenspeed.runtime.cache.deepseek_v4_cache_host import (
    DeepseekV4TokenToKVPoolHost,
)
from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter
from tokenspeed.runtime.cache.transfer.types import PAGED_CACHE_KIND
from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    parse_v4_compressor_state_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4TokenToKVPool,
)


@dataclass(frozen=True, slots=True)
class PagedCacheTensorRef:
    group_id: str
    layer_id: int
    device_tensor: torch.Tensor
    host_tensor: torch.Tensor
    page_bytes: int


def _parse_v4_compressed_kv_group_id(group_id: str) -> int | None:
    prefix = "v4.c"
    suffix = "a.compressed_kv"
    if not group_id.startswith(prefix) or not group_id.endswith(suffix):
        return None
    try:
        return int(group_id[len(prefix) : -len(suffix)])
    except ValueError:
        return None


def _ordered_page_pairs(src_pages: Iterable[int], dst_pages: Iterable[int]):
    seen = set()
    pairs = []
    for src_page, dst_page in zip(src_pages, dst_pages):
        pair = (int(src_page), int(dst_page))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    pairs.sort()
    return pairs


class DeepseekV4CachePool:
    """Group-paged DeepSeek V4 L2 transfer pool.

    This pool deliberately does not implement the CacheKind protocol. It is
    driven by typed paged-cache transfers whose group ids carry the page space.
    """

    kind = PAGED_CACHE_KIND

    def __init__(
        self,
        device_pool: DeepseekV4TokenToKVPool,
        host_pool: DeepseekV4TokenToKVPoolHost,
        io_backend: str,
    ) -> None:
        if io_backend not in ("kernel", "direct"):
            raise ValueError(
                f"Unsupported DeepSeek V4 paged-cache io_backend={io_backend}"
            )
        self.device_pool = device_pool
        self.host_pool = host_pool
        self.io_backend = io_backend
        self._counter = LayerDoneCounter(self.num_layers())
        device_pool.register_layer_transfer_counter(self._counter)
        self._transfer_stats: dict[str, dict[str, dict[str, int]]] = {
            "D2H": {},
            "H2D": {},
        }

    @property
    def device(self):
        return self.device_pool.device

    @property
    def host_layout(self) -> str:
        return self.host_pool.layout

    def num_layers(self) -> int:
        return int(self.device_pool.layer_num)

    def supports_layerwise_loadback(self) -> bool:
        return True

    def get_layer_done_counter(self) -> LayerDoneCounter:
        return self._counter

    def get_transfer_stats(self) -> dict[str, dict[str, dict[str, int]]]:
        return {
            direction: {group_id: dict(values) for group_id, values in groups.items()}
            for direction, groups in self._transfer_stats.items()
        }

    def reset_transfer_stats(self) -> None:
        for groups in self._transfer_stats.values():
            groups.clear()

    def local_layer_idx(self, global_layer_id: int) -> int:
        return global_layer_id

    def tensor_refs_for_group(
        self,
        group_id: str,
        layer_idx: int | None = None,
    ) -> list[PagedCacheTensorRef]:
        group_id = str(group_id)
        layer_ids = (
            range(self.num_layers())
            if layer_idx is None
            else range(layer_idx, layer_idx + 1)
        )
        refs: list[PagedCacheTensorRef] = []
        for layer_id in layer_ids:
            ratio = int(self.device_pool.layout.layer_ratio[layer_id])
            if group_id == V4_SWA_KV_GROUP_ID:
                refs.append(
                    self._ref(
                        group_id,
                        layer_id,
                        self.device_pool.swa_kv_buffer[layer_id],
                        self.host_pool.swa_kv_buffer[layer_id],
                    )
                )
                continue

            state_ratio = parse_v4_compressor_state_group_id(group_id)
            if state_ratio is not None:
                if ratio == state_ratio:
                    refs.append(
                        self._ref(
                            group_id,
                            layer_id,
                            self.device_pool.compressor_state_buffer[layer_id],
                            self.host_pool.compressor_state_buffer[layer_id],
                        )
                    )
                continue

            compressed_ratio = _parse_v4_compressed_kv_group_id(group_id)
            if compressed_ratio is not None:
                if ratio == compressed_ratio:
                    refs.append(
                        self._ref(
                            group_id,
                            layer_id,
                            self.device_pool.compressed_kv_buffer[layer_id],
                            self.host_pool.compressed_kv_buffer[layer_id],
                        )
                    )
                    if ratio == 4:
                        refs.append(
                            self._ref(
                                group_id,
                                layer_id,
                                self.device_pool.indexer_kv_buffer[layer_id],
                                self.host_pool.indexer_kv_buffer[layer_id],
                            )
                        )
                continue

            if group_id == V4_INDEXER_COMPRESSOR_STATE_GROUP_ID:
                if ratio == 4:
                    refs.append(
                        self._ref(
                            group_id,
                            layer_id,
                            self.device_pool.indexer_state_buffer[layer_id],
                            self.host_pool.indexer_state_buffer[layer_id],
                        )
                    )
                continue

            raise KeyError(f"unknown DeepSeek V4 paged-cache group {group_id!r}")
        return refs

    @staticmethod
    def _ref(
        group_id: str,
        layer_id: int,
        device_tensor: torch.Tensor | None,
        host_tensor: torch.Tensor | None,
    ) -> PagedCacheTensorRef:
        if device_tensor is None or host_tensor is None:
            raise ValueError(
                f"DeepSeek V4 group {group_id!r} has no tensor for layer {layer_id}"
            )
        return PagedCacheTensorRef(
            group_id=group_id,
            layer_id=layer_id,
            device_tensor=device_tensor,
            host_tensor=host_tensor,
            page_bytes=int(device_tensor[0].nbytes),
        )

    def writeback_paged(self, transfers: list) -> None:
        self._copy_paged(transfers, host_to_device=False, layer_idx=None)

    def loadback_paged(self, transfers: list, layer_idx: int) -> None:
        self._copy_paged(transfers, host_to_device=True, layer_idx=layer_idx)

    def _copy_paged(
        self,
        transfers: list,
        *,
        host_to_device: bool,
        layer_idx: int | None,
    ) -> None:
        for transfer in transfers:
            src_pages = list(getattr(transfer, "src_pages"))
            dst_pages = list(getattr(transfer, "dst_pages"))
            if len(src_pages) != len(dst_pages):
                raise ValueError(
                    "DeepSeek V4 paged transfer page count mismatch for "
                    f"group={transfer.group_id!r}: {len(src_pages)} src vs "
                    f"{len(dst_pages)} dst"
                )
            pairs = _ordered_page_pairs(src_pages, dst_pages)
            if not pairs:
                continue
            ordered_src = torch.tensor(
                [src for src, _ in pairs], dtype=torch.int64, device="cpu"
            )
            ordered_dst = torch.tensor(
                [dst for _, dst in pairs], dtype=torch.int64, device="cpu"
            )
            refs = self.tensor_refs_for_group(transfer.group_id, layer_idx=layer_idx)
            if not refs:
                continue
            direction = "H2D" if host_to_device else "D2H"
            transfer_bytes = len(pairs) * sum(ref.page_bytes for ref in refs)
            if host_to_device:
                src_layers = [ref.host_tensor for ref in refs]
                dst_layers = [ref.device_tensor for ref in refs]
            else:
                src_layers = [ref.device_tensor for ref in refs]
                dst_layers = [ref.host_tensor for ref in refs]
            transfer_kv_direct(
                src_layers=src_layers,
                dst_layers=dst_layers,
                src_indices=ordered_src,
                dst_indices=ordered_dst,
                page_size=1,
            )
            self._record_transfer_stats(
                direction,
                str(transfer.group_id),
                pages=len(pairs),
                transfer_bytes=transfer_bytes,
            )

    def _record_transfer_stats(
        self,
        direction: str,
        group_id: str,
        *,
        pages: int,
        transfer_bytes: int,
    ) -> None:
        stats = self._transfer_stats[direction].setdefault(
            group_id,
            {"calls": 0, "pages": 0, "bytes": 0},
        )
        stats["calls"] += 1
        stats["pages"] += int(pages)
        stats["bytes"] += int(transfer_bytes)
