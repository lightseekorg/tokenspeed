from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.kvcache.cuda import (
    DirectH2DScatterPlan,
    prepare_kv_direct_h2d_scatter_plan,
    transfer_kv_direct,
    transfer_kv_direct_h2d_scatter_prepared,
)

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
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)
_DEBUG = logging.DEBUG
_H2D_SCATTER_MIN_COPY_CALLS = 4096


@dataclass(frozen=True, slots=True)
class PagedCacheTensorRef:
    group_id: str
    layer_id: int
    device_tensor: torch.Tensor
    host_tensor: torch.Tensor
    page_bytes: int


@dataclass(slots=True)
class _PreparedPagedCacheTransfer:
    group_id: str
    src_indices: torch.Tensor
    dst_indices: torch.Tensor
    page_count: int
    span_count: int
    src_indices_device: torch.Tensor | None = None
    dst_indices_device: torch.Tensor | None = None

    def h2d_indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.src_indices_device is None or self.src_indices_device.device != device:
            if device.type == "cuda":
                if not self.src_indices.is_pinned():
                    self.src_indices = self.src_indices.pin_memory()
                if not self.dst_indices.is_pinned():
                    self.dst_indices = self.dst_indices.pin_memory()
            self.src_indices_device = self.src_indices.to(device, non_blocking=True)
            self.dst_indices_device = self.dst_indices.to(device, non_blocking=True)
        assert self.dst_indices_device is not None
        return self.src_indices_device, self.dst_indices_device


@dataclass(frozen=True, slots=True)
class _PagedCopySummary:
    backend: str
    fallback_reason: str
    pages: int
    spans: int
    tensor_refs: int
    effective_copy_calls: int
    total_bytes: int
    buckets: int
    kernel_launches: int


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


def _count_contiguous_spans(pairs: list[tuple[int, int]]) -> int:
    if not pairs:
        return 0
    spans = 1
    prev_src, prev_dst = pairs[0]
    for src_page, dst_page in pairs[1:]:
        if src_page != prev_src + 1 or dst_page != prev_dst + 1:
            spans += 1
        prev_src, prev_dst = src_page, dst_page
    return spans


def _coalesce_page_pairs_by_group(transfers: list) -> dict[str, list[tuple[int, int]]]:
    pairs_by_group: dict[str, list[tuple[int, int]]] = {}
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
        group_id = str(transfer.group_id)
        pairs_by_group.setdefault(group_id, []).extend(pairs)

    return {
        group_id: sorted(set(pairs))
        for group_id, pairs in pairs_by_group.items()
        if pairs
    }


class DeepseekV4CachePool:
    """Group-paged DeepSeek V4 L2 transfer pool.

    This pool deliberately does not implement the CacheKind protocol. It is
    driven by typed paged-cache transfers whose group ids carry the page space.
    """

    kind = PAGED_CACHE_KIND
    loadback_layer_chunk_size = 4

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
        self._h2d_scatter_plans: dict[
            str,
            tuple[int, tuple[tuple[int, int, int, int], ...], DirectH2DScatterPlan],
        ] = {}

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
        prepared = self.prepare_paged_transfers(transfers)
        self.writeback_prepared_paged(prepared)

    def loadback_paged(self, transfers: list, layer_idx: int) -> None:
        prepared = self.prepare_paged_transfers(transfers)
        self.loadback_prepared_paged(prepared, layer_idx)

    def prepare_paged_transfers(
        self, transfers: list
    ) -> list[_PreparedPagedCacheTransfer]:
        """Coalesce scheduler transfer fragments into group-level page copies."""
        prepared: list[_PreparedPagedCacheTransfer] = []
        for group_id, pairs in _coalesce_page_pairs_by_group(transfers).items():
            prepared.append(
                _PreparedPagedCacheTransfer(
                    group_id=group_id,
                    src_indices=torch.tensor(
                        [src for src, _ in pairs],
                        dtype=torch.int64,
                        device="cpu",
                    ),
                    dst_indices=torch.tensor(
                        [dst for _, dst in pairs],
                        dtype=torch.int64,
                        device="cpu",
                    ),
                    page_count=len(pairs),
                    span_count=_count_contiguous_spans(pairs),
                )
            )
        return prepared

    def writeback_prepared_paged(
        self, transfers: list[_PreparedPagedCacheTransfer]
    ) -> None:
        """Submit D2H copies using coalesced page index tensors."""
        for transfer in transfers:
            self._copy_prepared_transfer(
                transfer,
                self.tensor_refs_for_group(transfer.group_id),
                host_to_device=False,
            )

    def loadback_prepared_paged(
        self, transfers: list[_PreparedPagedCacheTransfer], layer_idx: int
    ) -> None:
        """Submit one layer's H2D copy using precomputed page index tensors."""
        self.loadback_prepared_paged_range(transfers, layer_idx, layer_idx + 1)

    def loadback_prepared_paged_range(
        self,
        transfers: list[_PreparedPagedCacheTransfer],
        layer_start: int,
        layer_end: int,
    ) -> None:
        """Submit a small layer range's H2D copies with shared page indices."""
        if layer_start < 0 or layer_end < layer_start or layer_end > self.num_layers():
            raise ValueError(
                "invalid DeepSeek V4 paged loadback layer range: "
                f"[{layer_start}, {layer_end})"
            )
        summaries: list[_PagedCopySummary] = []
        for transfer in transfers:
            refs: list[PagedCacheTensorRef] = []
            for layer_idx in range(layer_start, layer_end):
                refs.extend(
                    self.tensor_refs_for_group(
                        transfer.group_id,
                        layer_idx=layer_idx,
                    )
                )
            summary = self._copy_prepared_transfer(
                transfer,
                refs,
                host_to_device=True,
                layer_start=layer_start,
                layer_end=layer_end,
            )
            if summary is not None:
                summaries.append(summary)
        self._log_h2d_chunk_summary(layer_start, layer_end, summaries)

    def _copy_prepared_transfer(
        self,
        transfer: _PreparedPagedCacheTransfer,
        refs: list[PagedCacheTensorRef],
        *,
        host_to_device: bool,
        layer_start: int | None = None,
        layer_end: int | None = None,
    ) -> _PagedCopySummary | None:
        if transfer.page_count == 0 or not refs:
            return None
        transfer_bytes = transfer.page_count * sum(ref.page_bytes for ref in refs)
        effective_copy_calls = int(transfer.span_count) * len(refs)
        buckets = len({ref.page_bytes for ref in refs})
        if host_to_device:
            src_layers = [ref.host_tensor for ref in refs]
            dst_layers = [ref.device_tensor for ref in refs]
        else:
            src_layers = [ref.device_tensor for ref in refs]
            dst_layers = [ref.host_tensor for ref in refs]

        backend = "direct"
        fallback_reason = ""
        kernel_launches = 0
        if host_to_device:
            assert layer_start is not None and layer_end is not None
            full_refs = self.tensor_refs_for_group(transfer.group_id)
            threshold_copy_calls = max(
                effective_copy_calls,
                int(transfer.span_count) * len(full_refs),
            )
            scatter_result = None
            if threshold_copy_calls >= _H2D_SCATTER_MIN_COPY_CALLS:
                plan, fallback_reason = self._h2d_scatter_plan(
                    transfer.group_id,
                    full_refs,
                )
                if plan is not None:
                    src_indices_device, dst_indices_device = transfer.h2d_indices(
                        torch.device(self.device)
                    )
                    scatter_result = transfer_kv_direct_h2d_scatter_prepared(
                        plan,
                        src_indices_device,
                        dst_indices_device,
                        layer_start,
                        layer_end,
                    )
            else:
                fallback_reason = "below_threshold"
            if scatter_result is not None and scatter_result.used:
                backend = "scatter"
                buckets = int(scatter_result.buckets)
                kernel_launches = int(scatter_result.kernel_launches)
            elif scatter_result is not None:
                fallback_reason = fallback_reason or str(scatter_result.fallback_reason)

        if backend == "direct":
            transfer_kv_direct(
                src_layers=src_layers,
                dst_layers=dst_layers,
                src_indices=transfer.src_indices,
                dst_indices=transfer.dst_indices,
                page_size=1,
            )
        return _PagedCopySummary(
            backend=backend,
            fallback_reason=fallback_reason,
            pages=int(transfer.page_count),
            spans=int(transfer.span_count),
            tensor_refs=len(refs),
            effective_copy_calls=effective_copy_calls,
            total_bytes=transfer_bytes,
            buckets=buckets,
            kernel_launches=kernel_launches,
        )

    @staticmethod
    def _log_h2d_chunk_summary(
        layer_start: int,
        layer_end: int,
        summaries: list[_PagedCopySummary],
    ) -> None:
        if not summaries or not logger.isEnabledFor(_DEBUG):
            return
        backends = {summary.backend for summary in summaries}
        backend = next(iter(backends)) if len(backends) == 1 else "mixed"
        fallback_reasons = sorted(
            {
                summary.fallback_reason
                for summary in summaries
                if summary.fallback_reason
            }
        )
        logger.debug(
            "[cache_op][paged_l2] h2d_chunk layers=[%s,%s) backend=%s "
            "pages=%s spans=%s tensor_refs=%s effective_copy_calls=%s "
            "total_bytes=%s buckets=%s kernel_launches=%s fallback_reason=%s",
            layer_start,
            layer_end,
            backend,
            sum(summary.pages for summary in summaries),
            sum(summary.spans for summary in summaries),
            sum(summary.tensor_refs for summary in summaries),
            sum(summary.effective_copy_calls for summary in summaries),
            sum(summary.total_bytes for summary in summaries),
            sum(summary.buckets for summary in summaries),
            sum(summary.kernel_launches for summary in summaries),
            ",".join(fallback_reasons) if fallback_reasons else "none",
        )

    def _h2d_scatter_plan(
        self,
        group_id: str,
        refs: list[PagedCacheTensorRef],
    ) -> tuple[DirectH2DScatterPlan | None, str]:
        device = torch.device(self.device)
        stream_id = (
            int(torch.cuda.current_stream(device).cuda_stream)
            if device.type == "cuda"
            else 0
        )
        signature = tuple(
            (
                ref.layer_id,
                ref.page_bytes,
                ref.host_tensor.data_ptr(),
                ref.device_tensor.data_ptr(),
            )
            for ref in refs
        )
        cached = self._h2d_scatter_plans.get(group_id)
        if cached is not None and cached[0] == stream_id and cached[1] == signature:
            return cached[2], ""

        plan, reason = prepare_kv_direct_h2d_scatter_plan(
            [ref.host_tensor for ref in refs],
            [ref.device_tensor for ref in refs],
            [ref.layer_id for ref in refs],
        )
        if plan is not None:
            self._h2d_scatter_plans[group_id] = (stream_id, signature, plan)
        return plan, reason
