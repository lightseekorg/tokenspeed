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

import concurrent.futures
import math
import os
import socket
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from itertools import islice

import numpy as np
import numpy.typing as npt
import requests

from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import (
    CachePDPageManifest,
    CacheTransferContract,
    validate_cache_manifest_pair,
    validate_cache_peer_layout,
)
from tokenspeed.runtime.pd.mooncake.conn import MooncakeKVManagerBase
from tokenspeed.runtime.pd.mooncake.entities import (
    KVArgs,
    KVArgsRegisterInfo,
    KVManagerArgs,
    TransferIndexResolution,
    TransferInfo,
    TransferKVChunk,
)
from tokenspeed.runtime.pd.transfer_plan import (
    BufferKind,
    CacheTransferFragment,
    PagedCacheTransferPlanner,
    TransferFragment,
    TransferPlanFragment,
)
from tokenspeed.runtime.pd.utils import (
    DisaggregationMode,
    FastQueue,
    StepCounter,
    group_concurrent_contiguous,
)
from tokenspeed.runtime.utils import (
    get_colorful_logger,
)
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.network import (
    get_free_port,
    get_ip,
    get_local_ip_by_remote,
)

logger = get_colorful_logger(__name__)

# Application-side descriptor batching keeps heterogeneous-TP row fragments
# bounded in Python memory. This is not a Mooncake backend limit.
_TRANSFER_DESCRIPTOR_BATCH_SIZE = 4096


def _validate_cache_fragment_geometry(
    fragment: CacheTransferFragment,
    src_segment,
    dst_segment,
    partition,
) -> None:
    """Validate that a route fragment is a dense slice of its typed fields."""
    key = (fragment.group_id, fragment.field_id)
    if src_segment.element_size <= 0:
        raise ValueError(f"Paged cache fragment field {key!r} is not typed")
    src_end = (
        fragment.src_byte_offset
        + (fragment.rows_per_page - 1) * fragment.src_row_stride_bytes
        + fragment.bytes_per_row
    )
    dst_end = (
        fragment.dst_byte_offset
        + (fragment.rows_per_page - 1) * fragment.dst_row_stride_bytes
        + fragment.bytes_per_row
    )
    if (
        src_end > src_segment.payload_bytes
        or dst_end > dst_segment.payload_bytes
        or any(
            value % src_segment.element_size
            for value in (
                fragment.src_byte_offset,
                fragment.dst_byte_offset,
                fragment.src_row_stride_bytes,
                fragment.dst_row_stride_bytes,
                fragment.bytes_per_row,
            )
        )
    ):
        raise ValueError(f"Paged cache fragment has invalid field bounds for {key!r}")

    if partition is None:
        expected = (
            0,
            0,
            src_segment.payload_bytes,
            dst_segment.payload_bytes,
            src_segment.payload_bytes,
            1,
        )
        actual = (
            fragment.src_byte_offset,
            fragment.dst_byte_offset,
            fragment.src_row_stride_bytes,
            fragment.dst_row_stride_bytes,
            fragment.bytes_per_row,
            fragment.rows_per_page,
        )
        if actual != expected:
            raise ValueError(
                f"replicated Paged cache fragment is not a full field for {key!r}"
            )
        return

    axis = partition.axis
    inner_bytes = math.prod(src_segment.shape[axis + 1 :]) * src_segment.element_size
    expected_rows = math.prod(src_segment.shape[:axis])
    src_dense_row_bytes = src_segment.shape[axis] * inner_bytes
    dst_dense_row_bytes = dst_segment.shape[axis] * inner_bytes
    dense_rows = (
        fragment.rows_per_page == expected_rows
        and fragment.src_row_stride_bytes == src_dense_row_bytes
        and fragment.dst_row_stride_bytes == dst_dense_row_bytes
        and fragment.bytes_per_row % inner_bytes == 0
        and fragment.src_byte_offset % inner_bytes == 0
        and fragment.dst_byte_offset % inner_bytes == 0
        and fragment.src_byte_offset + fragment.bytes_per_row <= src_dense_row_bytes
        and fragment.dst_byte_offset + fragment.bytes_per_row <= dst_dense_row_bytes
    )
    coalesced_full_field = (
        expected_rows > 1
        and fragment.rows_per_page == 1
        and fragment.src_byte_offset == 0
        and fragment.dst_byte_offset == 0
        and fragment.bytes_per_row == src_segment.payload_bytes
        and fragment.bytes_per_row == dst_segment.payload_bytes
        and fragment.src_row_stride_bytes == fragment.bytes_per_row
        and fragment.dst_row_stride_bytes == fragment.bytes_per_row
    )
    if not dense_rows and not coalesced_full_field:
        raise ValueError(f"Paged cache fragment is not a dense field slice for {key!r}")


class MooncakeKVManagerPrefill(MooncakeKVManagerBase):
    def __init__(
        self,
        args: KVManagerArgs,
        kv_args: KVArgs,
    ):
        super().__init__(args, kv_args, DisaggregationMode.PREFILL)

        self.transfer_infos: dict[int, dict[str, TransferInfo]] = {}
        self.decode_kv_args_table: dict[str, KVArgsRegisterInfo] = {}
        self.session_failures = defaultdict(int)
        self.failed_sessions: dict[str, float] = {}
        self.failed_session_ttl = max(
            envs.TOKENSPEED_DISAGGREGATION_FAILED_SESSION_TTL.get(), 0
        )
        self.session_lock = threading.Lock()
        self.kv_layer_ids = list(
            getattr(self.kv_args, "kv_layer_ids", None)
            or range(len(self.kv_args.offsets))
        )
        self.state_layer_ids = list(getattr(self.kv_args, "state_layer_ids", []) or [])
        layer_ids = self.kv_layer_ids + self.state_layer_ids
        self.layer_num = (
            (max(layer_ids) + 1) if layer_ids else len(self.kv_args.offsets)
        )
        self._kv_layer_to_index = {
            layer_id: i
            for i, layer_id in enumerate(self.kv_layer_ids[: len(self.kv_args.offsets)])
        }
        # Under the cache contract ``kv_args.offsets`` describes physical raw
        # slabs (one per arena), not model layers, so the counts above collapse
        # to the slab count. Recover the true attention-layer count from the
        # ``layer.{N}.*`` field ids the contract carries so layerwise step-wait
        # math gates on the right per-layer cache step.
        if self.kv_args.cache_layout is not None:
            layer_ids = {
                layer_id
                for group in self.kv_args.cache_layout.groups
                for segment in group.transfer_segments
                if (layer_id := self._segment_layer_id(segment.field_id)) is not None
            }
            if layer_ids:
                self.layer_num = max(layer_ids) + 1
        self.layerwise_interval = 1
        self.layerwise_debug = envs.TOKENSPEED_PD_LAYERWISE_DEBUG.get()
        self.step_counter = None
        # room -> (bootstrap_token, spec_candidate_ids). Published after the prefill
        # forward; the transfer thread reads it on the wait_for_bootstrap_token path.
        self.prefill_metadata: dict[int, tuple[int, list[int] | None]] = {}
        self.expired_prefill_metadata_rooms: set[int] = set()
        self.bootstrap_token_cond = threading.Condition()
        # Determine the number of threads to use for kv sender
        cpu_count = os.cpu_count()
        transfer_thread_pool_size = (
            envs.TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE.get_set_value_or(
                min(max(4, int(0.75 * cpu_count) // 8), 12)
            )
        )
        transfer_queue_size = envs.TOKENSPEED_DISAGGREGATION_QUEUE_SIZE.get()
        if transfer_thread_pool_size < transfer_queue_size:
            raise ValueError(
                "TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE="
                f"{transfer_thread_pool_size} must be greater than or equal to "
                f"TOKENSPEED_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
        self.start_transfer_thread(transfer_thread_pool_size, transfer_queue_size)
        self.bootstrap_time_out = envs.TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        # Publish this manager only after every field used by its bootstrap and
        # transfer threads has been initialized.
        self.start_prefill_thread()
        self._register_to_bootstrap()

    def register_layerwise_step_counter(
        self, step_counter: StepCounter, interval: int
    ) -> None:
        self.step_counter = step_counter
        self.layerwise_interval = max(int(interval), 1)

    def reserve_layerwise_cache_steps(self) -> int:
        if self.step_counter is None:
            return 0
        cache_step, _ = self.step_counter.current_step()
        self.step_counter.advance_step(
            delta_cache_step=self.layer_num,
            delta_aux_step=0,
        )
        return cache_step

    def set_prefill_metadata(
        self,
        room: int,
        token: int,
        spec_candidate_ids: list[int] | None = None,
    ) -> None:
        with self.bootstrap_token_cond:
            if room in self.expired_prefill_metadata_rooms:
                self.expired_prefill_metadata_rooms.discard(room)
                logger.warning(
                    "Dropping late prefill metadata for expired bootstrap_room=%s",
                    room,
                )
                return
            self.prefill_metadata[room] = (
                token,
                spec_candidate_ids,
            )
            self.bootstrap_token_cond.notify_all()

    def discard_expired_metadata_room(self, room: int) -> None:
        """Best-effort cleanup of per-room metadata and expiry markers."""
        with self.bootstrap_token_cond:
            self.expired_prefill_metadata_rooms.discard(room)
            self.prefill_metadata.pop(room, None)

    def _wait_prefill_metadata(
        self,
        room: int | None,
        fallback_token: int,
        fallback_candidate_ids: list[int] | None,
    ) -> tuple[int, list[int] | None]:
        if room is None or fallback_token != -1:
            return fallback_token, fallback_candidate_ids
        wait_log_interval = max(envs.TOKENSPEED_PD_PREFILL_METADATA_TIMEOUT.get(), 0.01)
        start_time = time.monotonic()
        next_log_time = start_time + wait_log_interval
        with self.bootstrap_token_cond:
            while room not in self.prefill_metadata:
                if self.request_status.get(room) == TransferPoll.Failed:
                    self.expired_prefill_metadata_rooms.add(room)
                    logger.warning(
                        "Prefill metadata unavailable for failed bootstrap_room=%s; using fallback=%s",
                        room,
                        fallback_token,
                    )
                    return (
                        fallback_token,
                        fallback_candidate_ids,
                    )
                now = time.monotonic()
                if now >= next_log_time:
                    logger.debug(
                        "Still waiting for prefill metadata for bootstrap_room=%s after %.2fs",
                        room,
                        now - start_time,
                    )
                    next_log_time = now + wait_log_interval
                self.bootstrap_token_cond.wait(timeout=0.01)
            return self.prefill_metadata[room]

    def _is_session_failed(self, mooncake_session_id: str) -> bool:
        if self.failed_session_ttl <= 0:
            return False
        failed_at = self.failed_sessions.get(mooncake_session_id)
        if failed_at is None:
            return False
        elapsed = time.monotonic() - failed_at
        logger.info(
            "Session %s failed for %.2fs (TTL=%ds).",
            mooncake_session_id,
            elapsed,
            self.failed_session_ttl,
        )
        if elapsed < self.failed_session_ttl:
            return True
        del self.failed_sessions[mooncake_session_id]
        logger.info(
            "Session %s failed TTL expired (%.2fs >= %ds), reset.",
            mooncake_session_id,
            elapsed,
            self.failed_session_ttl,
        )
        return False

    def _mark_session_failed(
        self, mooncake_session_id: str, reason: str = "transfer_failed"
    ) -> None:
        if self.failed_session_ttl <= 0:
            return
        self.failed_sessions[mooncake_session_id] = time.monotonic()
        logger.warning(
            "Session %s marked failed (reason=%s, ttl=%ds).",
            mooncake_session_id,
            reason,
            self.failed_session_ttl,
        )

    def _clear_failed_session(self, mooncake_session_id: str) -> None:
        if mooncake_session_id in self.failed_sessions:
            del self.failed_sessions[mooncake_session_id]
            logger.info(
                "Session %s failed state cleared due to KVArgs registration.",
                mooncake_session_id,
            )
        if mooncake_session_id in self.session_failures:
            del self.session_failures[mooncake_session_id]

    def resolve_transfer_indices(
        self,
        kv_chunk: TransferKVChunk,
        req: TransferInfo,
    ) -> TransferIndexResolution:
        self._validate_cache_transfer(kv_chunk, req)
        src_indices = kv_chunk.prefill_kv_indices
        dst_indices = req.dst_kv_indices[kv_chunk.index_slice]

        valid_len = min(len(src_indices), len(dst_indices))
        # Fast path: empty transfer chunk. Avoid MLA assertions/index ops on empty payload.
        if valid_len == 0:
            empty = np.array([], dtype=np.int64)
            return TransferIndexResolution(src_indices=empty, dst_indices=empty)

        if valid_len < len(src_indices) or valid_len < len(dst_indices):
            logger.warning(
                "Mismatched transfer indices, truncating to %s (src=%s, dst=%s)",
                valid_len,
                len(src_indices),
                len(dst_indices),
            )
        src_indices = src_indices[:valid_len]
        dst_indices = dst_indices[:valid_len]

        return TransferIndexResolution(src_indices, dst_indices)

    def _validate_cache_transfer(
        self,
        kv_chunk: TransferKVChunk,
        req: TransferInfo,
    ) -> None:
        layout = getattr(self.kv_args, "cache_layout", None)
        cache_metadata_present = any(
            value is not None
            for value in (
                kv_chunk.page_manifest,
                req.page_manifest,
                req.peer_cache_layout,
            )
        )
        if layout is None:
            if cache_metadata_present:
                raise ValueError(
                    "legacy Mooncake transfer received Paged cache metadata"
                )
            return

        if (
            kv_chunk.page_manifest is None
            or req.page_manifest is None
            or req.peer_cache_layout is None
        ):
            raise ValueError(
                "Paged cache transfer is missing layout or manifest metadata"
            )
        if req.transfer_fragments and not all(
            isinstance(fragment, CacheTransferFragment)
            for fragment in req.transfer_fragments
        ):
            raise ValueError(
                "Paged cache raw-slab transfer received legacy TP fragments"
            )
        if not kv_chunk.is_last:
            raise ValueError(
                "Paged cache transfer must be submitted as one final chunk"
            )
        if kv_chunk.index_slice.start not in (
            None,
            0,
        ) or kv_chunk.index_slice.step not in (
            None,
            1,
        ):
            raise ValueError(
                "Paged cache transfer page vector must start at offset zero"
            )

        local_tp_rank = self._validate_cache_route(req)
        local_segments = {
            (field.group_id, field.field_id): field for field in layout.plan.fields
        }
        peer_segments = {
            (field.group_id, field.field_id): field
            for field in req.peer_cache_layout.plan.fields
        }
        if req.transfer_fragments:
            src_ranks = {fragment.src_rank for fragment in req.transfer_fragments}
            dst_ranks = {fragment.dst_rank for fragment in req.transfer_fragments}
            if len(src_ranks) != 1 or len(dst_ranks) != 1:
                raise ValueError(
                    "Paged cache TP fragments must describe one source/destination rank"
                )
            if src_ranks != {local_tp_rank}:
                raise ValueError(
                    "Paged cache TP fragment source rank disagrees with receiver"
                )
            for fragment in req.transfer_fragments:
                key = (fragment.group_id, fragment.field_id)
                try:
                    local_segment = local_segments[key]
                    peer_segment = peer_segments[key]
                except KeyError as exc:
                    raise ValueError(
                        f"Paged cache TP fragment names unknown field {key!r}"
                    ) from exc
                partition = layout.transfer_schema.partition_for(fragment.field_id)
                _validate_cache_fragment_geometry(
                    fragment,
                    local_segment,
                    peer_segment,
                    partition,
                )
        elif any(
            (
                local_segment.payload_bytes,
                local_segment.shape,
                local_segment.element_size,
            )
            != (
                peer_segments[key].payload_bytes,
                peer_segments[key].shape,
                peer_segments[key].element_size,
            )
            for key, local_segment in local_segments.items()
        ):
            raise ValueError(
                "Paged cache identity route has different rank-local field geometry"
            )
        validate_cache_manifest_pair(
            kv_chunk.page_manifest,
            req.page_manifest,
            layout,
            req.peer_cache_layout,
        )
        expected_src = tuple(
            page_id
            for group in kv_chunk.page_manifest.groups
            for page_id in group.page_ids
        )
        expected_dst = tuple(
            page_id for group in req.page_manifest.groups for page_id in group.page_ids
        )
        actual_src = tuple(int(page) for page in kv_chunk.prefill_kv_indices)
        actual_dst = tuple(int(page) for page in req.dst_kv_indices)
        if actual_src != expected_src or actual_dst != expected_dst:
            raise ValueError("Paged cache manifest and Mooncake page vector disagree")
        if kv_chunk.index_slice.stop != len(expected_src):
            raise ValueError("Paged cache transfer page slice is incomplete")

    def _validate_cache_route(self, req: TransferInfo) -> int:
        """Replan one Paged route from independently published TP metadata."""
        layout = getattr(self.kv_args, "cache_layout", None)
        if layout is None or req.peer_cache_layout is None:
            raise ValueError("Paged cache route is missing source or peer layout")
        if req.decode_tp_size is None or req.decode_tp_rank is None:
            raise ValueError("Paged cache transfer is missing Decode TP identity")
        prefill_tp_size = self.world_size // self.dp_size
        local_tp_rank = self.attn_tp_rank % prefill_tp_size
        route = PagedCacheTransferPlanner(
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=req.decode_tp_size,
            prefill_layout=layout,
            decode_layout=req.peer_cache_layout,
        ).plan_for_decode_rank(req.decode_tp_rank)
        if local_tp_rank not in route.target_prefill_ranks:
            raise ValueError(
                "Paged cache Decode route targets the wrong Prefill TP rank"
            )
        expected_fragments = route.fragments_by_prefill_rank.get(local_tp_rank, ())
        if tuple(req.transfer_fragments) != expected_fragments:
            raise ValueError(
                "Paged cache transfer fragments disagree with the typed route plan"
            )
        expected_fanout = route.required_dst_info_num_for_prefill_rank(local_tp_rank)
        if req.required_dst_info_num != expected_fanout:
            raise ValueError(
                "Paged cache destination fanout disagrees with the typed route plan"
            )
        return local_tp_rank

    def _validate_cache_room_fanout(self, reqs: tuple[TransferInfo, ...]) -> None:
        """Require one complete, unique Decode-rank set for a Paged source."""
        paged_reqs = tuple(req for req in reqs if req.peer_cache_layout is not None)
        if not paged_reqs:
            return
        if len(paged_reqs) != len(reqs):
            raise ValueError("Paged cache room mixes legacy and typed destinations")
        local_tp_ranks = {self._validate_cache_route(req) for req in paged_reqs}
        decode_tp_sizes = {req.decode_tp_size for req in paged_reqs}
        if len(local_tp_ranks) != 1 or len(decode_tp_sizes) != 1:
            raise ValueError("Paged cache room has inconsistent TP metadata")
        local_tp_rank = next(iter(local_tp_ranks))
        decode_tp_size = next(iter(decode_tp_sizes))
        assert decode_tp_size is not None
        prefill_tp_size = self.world_size // self.dp_size
        planner = PagedCacheTransferPlanner(
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=decode_tp_size,
            prefill_layout=self.kv_args.cache_layout,
            decode_layout=paged_reqs[0].peer_cache_layout,
        )
        expected_decode_ranks = {
            decode_tp_rank
            for decode_tp_rank in range(decode_tp_size)
            if local_tp_rank
            in planner.plan_for_decode_rank(decode_tp_rank).target_prefill_ranks
        }
        actual_decode_ranks = tuple(req.decode_tp_rank for req in paged_reqs)
        if (
            len(actual_decode_ranks) != len(set(actual_decode_ranks))
            or set(actual_decode_ranks) != expected_decode_ranks
        ):
            raise ValueError(
                "Paged cache room destination ranks disagree with the typed route plan"
            )

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        block_iter = iter(transfer_blocks)
        while batch := tuple(islice(block_iter, _TRANSFER_DESCRIPTOR_BATCH_SIZE)):
            src_addrs, dst_addrs, lengths = zip(*batch, strict=True)
            ret = self.engine.batch_transfer_sync(
                mooncake_session_id,
                list(src_addrs),
                list(dst_addrs),
                list(lengths),
            )
            if ret != 0:
                return ret
        return 0

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        executor: concurrent.futures.ThreadPoolExecutor,
        transfer_fragments: tuple[TransferPlanFragment, ...] = (),
        src_page_manifest: CachePDPageManifest | None = None,
        dst_page_manifest: CachePDPageManifest | None = None,
        dst_cache_layout: CacheTransferContract | None = None,
    ):
        if (src_page_manifest is None) != (dst_page_manifest is None):
            raise ValueError(
                "Paged cache transfer requires both source and destination manifests"
            )
        if src_page_manifest is not None:
            if dst_page_manifest is None:
                raise ValueError(
                    "Paged cache transfer requires both source and destination manifests"
                )
            transfer_blocks = self._cache_transfer_blocks(
                dst_ptrs=dst_kv_ptrs,
                src_indices=prefill_kv_indices,
                dst_indices=dst_kv_indices,
                src_manifest=src_page_manifest,
                dst_manifest=dst_page_manifest,
                transfer_fragments=transfer_fragments,
                dst_cache_layout=dst_cache_layout,
            )
            return self._transfer_data(mooncake_session_id, transfer_blocks)
        if dst_cache_layout is not None:
            raise ValueError("legacy Mooncake transfer received a Paged cache layout")

        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        # ``_layer_transfer_blocks`` indexes ``kv_args.offsets`` by its loop var,
        # and ``offsets`` is keyed by KV-LAYER INDEX (one entry per attention/KV
        # layer), not by global layer id. For hybrid models (e.g. Qwen3.5 GDN +
        # attention) only the attention layers carry KV, so len(offsets) is the
        # KV-layer count while ``self.layer_num`` is the (larger) global layer
        # count -- using ``layer_num`` here over-runs ``offsets`` and IndexErrors.
        # Iterate the full KV-index space instead (the layerwise path already
        # maps global->KV index via ``_kv_layer_to_index`` before calling this).
        transfer_blocks = self._layer_transfer_blocks(
            dst_ptrs=dst_kv_ptrs,
            src_blocks=prefill_kv_blocks,
            dst_blocks=dst_kv_blocks,
            begin_layer_id=0,
            end_layer_id=len(self.kv_args.offsets),
            transfer_fragments=transfer_fragments,
        )
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    @staticmethod
    def _segment_layer_id(field_id: str) -> int | None:
        """Parse the model layer id out of a ``layer.{N}.<component>`` field id.

        Returns ``None`` for fields that are not per-layer (no such fields exist
        under current recipes, but the guard keeps layerwise filtering total)."""
        parts = field_id.split(".", 2)
        if len(parts) >= 2 and parts[0] == "layer" and parts[1].isdigit():
            return int(parts[1])
        return None

    def _cache_transfer_blocks(
        self,
        *,
        dst_ptrs: list[int],
        src_indices: npt.NDArray[np.int64],
        dst_indices: npt.NDArray[np.int64],
        src_manifest: CachePDPageManifest,
        dst_manifest: CachePDPageManifest,
        transfer_fragments: tuple[TransferPlanFragment, ...] = (),
        dst_cache_layout: CacheTransferContract | None = None,
    ) -> Iterator[tuple[int, int, int]]:
        layout = self.kv_args.cache_layout
        if layout is None:
            raise ValueError("legacy Mooncake transfer received Paged cache manifests")
        if dst_cache_layout is None:
            raise ValueError(
                "Paged cache transfer requires the complete destination layout"
            )
        cache_fragments = tuple(
            fragment
            for fragment in transfer_fragments
            if isinstance(fragment, CacheTransferFragment)
        )
        if len(cache_fragments) != len(transfer_fragments):
            raise ValueError("Paged cache transfer received legacy TP fragments")

        validate_cache_peer_layout(layout, dst_cache_layout)
        validate_cache_manifest_pair(
            src_manifest,
            dst_manifest,
            layout,
            dst_cache_layout,
        )
        expected_arena_bytes = layout.plan.arena_bytes
        if (
            len(self.kv_args.kv_data_ptrs) != 1
            or tuple(self.kv_args.kv_data_lens) != (expected_arena_bytes,)
            or tuple(self.kv_args.kv_item_lens) != (layout.plan.lcm_block_bytes,)
        ):
            raise ValueError(
                "Paged cache source must publish exactly one arena descriptor"
            )
        if len(dst_ptrs) != 1:
            raise ValueError(
                "Paged cache destination must publish exactly one arena pointer"
            )

        local_segments = {
            (field.group_id, field.field_id): field for field in layout.plan.fields
        }
        peer_segments = {
            (field.group_id, field.field_id): field
            for field in dst_cache_layout.plan.fields
        }
        layout_group_ids = {group.group_id for group in layout.group_specs}
        fragments_by_group: dict[str, list[CacheTransferFragment]] = defaultdict(list)
        fragments_by_field: dict[tuple[str, str], list[CacheTransferFragment]] = (
            defaultdict(list)
        )
        fragment_src_ranks = {fragment.src_rank for fragment in cache_fragments}
        fragment_dst_ranks = {fragment.dst_rank for fragment in cache_fragments}
        if cache_fragments and (
            len(fragment_src_ranks) != 1 or len(fragment_dst_ranks) != 1
        ):
            raise ValueError(
                "Paged cache fragments must describe one source/destination rank"
            )
        local_tp_rank = getattr(self, "attn_tp_rank", None)
        if (
            cache_fragments
            and local_tp_rank is not None
            and fragment_src_ranks != {local_tp_rank}
        ):
            raise ValueError("Paged cache fragment source rank disagrees with receiver")
        for fragment in cache_fragments:
            if fragment.group_id not in layout_group_ids:
                raise ValueError(
                    f"Paged cache fragment names unknown group {fragment.group_id!r}"
                )
            key = (fragment.group_id, fragment.field_id)
            try:
                src_segment = local_segments[key]
                dst_segment = peer_segments[key]
            except KeyError as exc:
                raise ValueError(
                    f"Paged cache fragment names unknown field {key!r}"
                ) from exc
            partition = layout.transfer_schema.partition_for(fragment.field_id)
            _validate_cache_fragment_geometry(
                fragment,
                src_segment,
                dst_segment,
                partition,
            )
            fragments_by_group[fragment.group_id].append(fragment)
            fragments_by_field[key].append(fragment)

        for key, field_fragments in fragments_by_field.items():
            for side in ("src", "dst"):
                intervals = sorted(
                    (
                        getattr(fragment, f"{side}_byte_offset")
                        + row * getattr(fragment, f"{side}_row_stride_bytes"),
                        getattr(fragment, f"{side}_byte_offset")
                        + row * getattr(fragment, f"{side}_row_stride_bytes")
                        + fragment.bytes_per_row,
                    )
                    for fragment in field_fragments
                    for row in range(fragment.rows_per_page)
                )
                if any(
                    right_start < left_end
                    for (_, left_end), (right_start, _) in zip(
                        intervals, intervals[1:], strict=False
                    )
                ):
                    raise ValueError(
                        f"Paged cache fragments overlap in {side} field {key!r}"
                    )

        expected_page_count = sum(len(group.page_ids) for group in src_manifest.groups)
        if (
            len(src_indices) != expected_page_count
            or len(dst_indices) != expected_page_count
        ):
            raise ValueError("Paged cache transfer page vector has trailing entries")
        # Validate every group-scoped vector and typed identity mapping before
        # yielding the first descriptor. The iterator stays bounded, while
        # malformed metadata can never DMA an earlier group before a later
        # group is rejected.
        group_transfers = []
        page_offset = 0
        for group_spec, src_group, dst_group in zip(
            layout.group_specs,
            src_manifest.groups,
            dst_manifest.groups,
            strict=True,
        ):
            page_count = len(src_group.page_ids)
            group_src_indices = src_indices[page_offset : page_offset + page_count]
            group_dst_indices = dst_indices[page_offset : page_offset + page_count]
            if (
                tuple(int(page) for page in group_src_indices) != src_group.page_ids
                or tuple(int(page) for page in group_dst_indices) != dst_group.page_ids
            ):
                raise ValueError(
                    "Paged cache manifest and Mooncake page vector disagree"
                )
            group_transfers.append(
                (
                    group_spec,
                    group_src_indices,
                    group_dst_indices,
                )
            )
            page_offset += page_count

        src_ptr = self.kv_args.kv_data_ptrs[0]
        dst_ptr = dst_ptrs[0]
        for group_spec, group_src_indices, group_dst_indices in group_transfers:
            if cache_fragments:
                for fragment in fragments_by_group.get(group_spec.group_id, ()):
                    key = (fragment.group_id, fragment.field_id)
                    src_segment = local_segments[key]
                    dst_segment = peer_segments[key]
                    for src_page, dst_page in zip(
                        group_src_indices, group_dst_indices, strict=True
                    ):
                        src_page_addr = (
                            src_ptr
                            + layout.plan.field_page_byte_offset(
                                src_segment.field_id, 0
                            )
                            + int(src_page) * src_segment.page_stride_bytes
                            + fragment.src_byte_offset
                        )
                        dst_page_addr = (
                            dst_ptr
                            + dst_cache_layout.plan.field_page_byte_offset(
                                dst_segment.field_id, 0
                            )
                            + int(dst_page) * dst_segment.page_stride_bytes
                            + fragment.dst_byte_offset
                        )
                        for row in range(fragment.rows_per_page):
                            yield (
                                src_page_addr + row * fragment.src_row_stride_bytes,
                                dst_page_addr + row * fragment.dst_row_stride_bytes,
                                fragment.bytes_per_row,
                            )
                continue

            for src_segment in layout.fields_for_group(group_spec.group_id):
                key = (group_spec.group_id, src_segment.field_id)
                dst_segment = peer_segments[key]
                if src_segment.payload_bytes != dst_segment.payload_bytes:
                    raise ValueError(
                        f"Paged cache identity field {key!r} has "
                        "different rank-local payloads"
                    )
                for src_page, dst_page in zip(
                    group_src_indices, group_dst_indices, strict=True
                ):
                    yield (
                        src_ptr
                        + layout.plan.field_page_byte_offset(src_segment.field_id, 0)
                        + int(src_page) * src_segment.page_stride_bytes,
                        dst_ptr
                        + dst_cache_layout.plan.field_page_byte_offset(
                            dst_segment.field_id, 0
                        )
                        + int(dst_page) * dst_segment.page_stride_bytes,
                        src_segment.payload_bytes,
                    )

    def _layer_transfer_blocks(
        self,
        dst_ptrs: list[int],
        src_blocks,
        dst_blocks,
        begin_layer_id: int,
        end_layer_id: int,
        transfer_fragments: tuple[TransferFragment, ...] = (),
    ) -> list[tuple[int, int, int]]:
        transfer_blocks = []
        fragments_by_buffer: dict[int, list[TransferFragment]] = defaultdict(list)
        for fragment in transfer_fragments:
            if fragment.buffer_kind != BufferKind.MAMBA_STATE:
                fragments_by_buffer[fragment.buffer_index].append(fragment)
        has_fragment_plan = bool(transfer_fragments)

        for layer_id in range(begin_layer_id, end_layer_id):
            for ptr_offset in self.kv_args.offsets[layer_id]:
                src_ptr = self.kv_args.kv_data_ptrs[ptr_offset]
                dst_ptr = dst_ptrs[ptr_offset]
                item_len = self.kv_args.kv_item_lens[ptr_offset]
                buffer_fragments = fragments_by_buffer.get(ptr_offset, ())
                if has_fragment_plan:
                    for fragment in buffer_fragments:
                        for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                            page_count = min(len(prefill_index), len(decode_index))
                            if fragment.page_count is not None:
                                page_count = min(page_count, fragment.page_count)
                            for page_offset in range(page_count):
                                src_addr = (
                                    src_ptr
                                    + int(prefill_index[page_offset])
                                    * fragment.src_page_stride_bytes
                                    + fragment.src_byte_offset
                                )
                                dst_addr = (
                                    dst_ptr
                                    + int(decode_index[page_offset])
                                    * fragment.dst_page_stride_bytes
                                    + fragment.dst_byte_offset
                                )
                                transfer_blocks.append(
                                    (src_addr, dst_addr, fragment.bytes_per_page)
                                )
                    continue

                for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                    src_addr = src_ptr + int(prefill_index[0]) * item_len
                    dst_addr = dst_ptr + int(decode_index[0]) * item_len
                    length = item_len * len(prefill_index)
                    transfer_blocks.append((src_addr, dst_addr, length))
        return transfer_blocks

    def _wait_until_cache_step(self, target_step: int) -> None:
        if self.step_counter is None:
            return
        while True:
            ready_step = self.step_counter.query_ready_cache_step()
            if StepCounter.is_step_ready(ready_step, target_step):
                return
            time.sleep(1e-4)

    def send_mamba_cache(
        self,
        mooncake_session_id: str,
        prefill_mamba_indices: npt.NDArray[np.int64] | None,
        dst_state_data_ptrs: list[int],
        dst_mamba_indices: npt.NDArray[np.int64] | None,
        begin_layer_id: int | None = None,
        end_layer_id: int | None = None,
        transfer_fragments: tuple[TransferFragment, ...] = (),
    ) -> int:
        if self.kv_args.state_type != "mamba":
            return 0
        state_ptrs = self.kv_args.state_data_ptrs
        state_item_lens = self.kv_args.state_item_lens
        if (
            not state_ptrs
            or not dst_state_data_ptrs
            or prefill_mamba_indices is None
            or dst_mamba_indices is None
        ):
            return 0
        if len(state_ptrs) != len(dst_state_data_ptrs):
            logger.error(
                "Mamba state tensor count mismatch: prefill=%d decode=%d",
                len(state_ptrs),
                len(dst_state_data_ptrs),
            )
            return -1

        if prefill_mamba_indices.shape != dst_mamba_indices.shape:
            if prefill_mamba_indices.size == 1 and dst_mamba_indices.size > 1:
                prefill_mamba_indices = np.full(
                    dst_mamba_indices.shape,
                    int(prefill_mamba_indices[0]),
                    dtype=np.int64,
                )
            else:
                logger.error(
                    "Mamba state slot count mismatch: prefill=%s decode=%s",
                    prefill_mamba_indices.tolist(),
                    dst_mamba_indices.tolist(),
                )
                return -1

        state_items = list(
            enumerate(zip(state_ptrs, dst_state_data_ptrs, state_item_lens))
        )
        if begin_layer_id is not None or end_layer_id is not None:
            begin = 0 if begin_layer_id is None else begin_layer_id
            end = self.layer_num if end_layer_id is None else end_layer_id
            if len(self.state_layer_ids) != len(state_items):
                logger.error(
                    "Mamba state layer id count mismatch: ids=%d tensors=%d",
                    len(self.state_layer_ids),
                    len(state_items),
                )
                return -1
            state_items = [
                item
                for item, layer_id in zip(state_items, self.state_layer_ids)
                if begin <= layer_id < end
            ]
            if not state_items:
                return 0

        valid = (prefill_mamba_indices >= 0) & (dst_mamba_indices >= 0)
        log_layerwise = getattr(self, "layerwise_debug", False)
        if log_layerwise and begin_layer_id is not None and end_layer_id is not None:
            logger.info(
                "[layerwise_transfer] session=%s layers=[%d,%d) "
                "send mamba tensors=%d bytes=%d",
                mooncake_session_id,
                begin_layer_id,
                end_layer_id,
                len(state_items),
                sum(item_len for _, (_, _, item_len) in state_items) * int(valid.sum()),
            )
        if not valid.any():
            return 0

        src_indices = prefill_mamba_indices[valid]
        dst_indices = dst_mamba_indices[valid]
        src_blocks, dst_blocks = group_concurrent_contiguous(src_indices, dst_indices)
        transfer_blocks = []
        state_fragments_by_buffer: dict[int, list[TransferFragment]] = defaultdict(list)
        for fragment in transfer_fragments:
            if fragment.buffer_kind == BufferKind.MAMBA_STATE:
                state_fragments_by_buffer[fragment.buffer_index].append(fragment)
        has_state_fragment_plan = bool(state_fragments_by_buffer)

        for state_index, (src_ptr, dst_ptr, item_len) in state_items:
            buffer_fragments = state_fragments_by_buffer.get(state_index, ())
            if has_state_fragment_plan:
                for fragment in buffer_fragments:
                    for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                        page_count = min(len(prefill_index), len(decode_index))
                        if fragment.page_count is not None:
                            page_count = min(page_count, fragment.page_count)
                        for page_offset in range(page_count):
                            src_addr = (
                                src_ptr
                                + int(prefill_index[page_offset])
                                * fragment.src_page_stride_bytes
                                + fragment.src_byte_offset
                            )
                            dst_addr = (
                                dst_ptr
                                + int(decode_index[page_offset])
                                * fragment.dst_page_stride_bytes
                                + fragment.dst_byte_offset
                            )
                            transfer_blocks.append(
                                (src_addr, dst_addr, fragment.bytes_per_page)
                            )
                continue

            for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))

        total_bytes = sum(length for _, _, length in transfer_blocks)
        ret = self._transfer_data(mooncake_session_id, transfer_blocks)
        logger.debug(
            "Transferred mamba cache for session=%s slots=%s blocks=%d bytes=%d ret=%s",
            mooncake_session_id,
            src_indices.tolist(),
            len(transfer_blocks),
            total_bytes,
            ret,
        )
        return ret

    def send_kvcache_layerwise(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        begin_cache_step: int,
        interval: int,
        dst_state_data_ptrs: list[int] | None = None,
        prefill_mamba_indices: npt.NDArray[np.int64] | None = None,
        dst_mamba_indices: npt.NDArray[np.int64] | None = None,
        transfer_fragments: tuple[TransferFragment, ...] = (),
    ) -> int:
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        interval = max(int(interval), 1)
        log_layerwise = getattr(self, "layerwise_debug", False)
        for begin_layer_id in range(0, self.layer_num, interval):
            end_layer_id = min(begin_layer_id + interval, self.layer_num)
            target_step = begin_cache_step + end_layer_id - 1
            if log_layerwise:
                logger.info(
                    "[layerwise_transfer] session=%s layers=[%d,%d) wait_cache_step=%d pages=%d",
                    mooncake_session_id,
                    begin_layer_id,
                    end_layer_id,
                    target_step,
                    len(prefill_kv_indices),
                )
            self._wait_until_cache_step(target_step)

            transfer_blocks = []
            if prefill_kv_blocks:
                for global_layer_id in range(begin_layer_id, end_layer_id):
                    kv_layer_index = self._kv_layer_to_index.get(global_layer_id)
                    if kv_layer_index is None:
                        continue
                    transfer_blocks.extend(
                        self._layer_transfer_blocks(
                            dst_ptrs=dst_kv_ptrs,
                            src_blocks=prefill_kv_blocks,
                            dst_blocks=dst_kv_blocks,
                            begin_layer_id=kv_layer_index,
                            end_layer_id=kv_layer_index + 1,
                            transfer_fragments=transfer_fragments,
                        )
                    )
            if transfer_blocks:
                if log_layerwise:
                    total_bytes = sum(length for _, _, length in transfer_blocks)
                    logger.info(
                        "[layerwise_transfer] session=%s layers=[%d,%d) send kv blocks=%d bytes=%d",
                        mooncake_session_id,
                        begin_layer_id,
                        end_layer_id,
                        len(transfer_blocks),
                        total_bytes,
                    )
                ret = self._transfer_data(mooncake_session_id, transfer_blocks)
                if ret != 0:
                    return ret

            ret = self.send_mamba_cache(
                mooncake_session_id,
                prefill_mamba_indices,
                dst_state_data_ptrs or [],
                dst_mamba_indices,
                begin_layer_id=begin_layer_id,
                end_layer_id=end_layer_id,
                transfer_fragments=transfer_fragments,
            )
            if ret != 0:
                return ret
            if log_layerwise:
                logger.info(
                    "[layerwise_transfer] session=%s layers=[%d,%d) done",
                    mooncake_session_id,
                    begin_layer_id,
                    end_layer_id,
                )
        return 0

    def sync_status_to_decode_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        status: int,
        prefill_rank: int,
        bootstrap_token: int = -1,
        spec_candidate_ids: list[int] | None = None,
    ):
        if ":" in remote:
            remote = remote.split(":")[0]
        spec_candidate_payload = (
            np.asarray(spec_candidate_ids, dtype=np.int32).tobytes()
            if spec_candidate_ids is not None
            else b""
        )
        socket, lock = self._connect("tcp://" + remote + ":" + str(dst_port))
        with lock:
            socket.send_multipart(
                [
                    str(room).encode("ascii"),
                    str(status).encode("ascii"),
                    str(prefill_rank).encode("ascii"),
                    str(bootstrap_token).encode("ascii"),
                    spec_candidate_payload,
                ]
            )

    def abort_room(self, room: int, reason: str) -> None:
        """Notify the decode that a room failed before any KV transfer.

        EPD: when the prefill aborts a request on embedding-receive timeout it never
        sends KV, so the decode's dual-dispatched KV receiver would wait
        indefinitely -- its heartbeat only trips if the prefill /health dies, and the
        receiver waiting_timeout only covers the WaitingForInput state (a receiver
        whose prefill never registered a sender is stuck earlier). Push a Failed
        status to every decode endpoint that already pre-allocated for this room
        (mirrors the in-transfer failure path), so the decode raises a FailedEvent and
        the client gets an error instead of hanging. A room whose decode has not
        pre-allocated yet is only marked Failed locally (no endpoint to notify).
        """
        self.record_failure(room, reason)
        self.update_status(room, TransferPoll.Failed)
        for req in list(self.transfer_infos.get(room, {}).values()):
            if not req.is_dummy:
                try:
                    self.sync_status_to_decode_endpoint(
                        req.endpoint,
                        req.dst_port,
                        req.room,
                        TransferPoll.Failed,
                        self.attn_tp_rank,
                    )
                except Exception:
                    logger.exception(
                        "Failed to notify Decode about room-level transfer "
                        "failure (room=%s endpoint=%s:%s)",
                        room,
                        req.endpoint,
                        req.dst_port,
                    )
        self.transfer_infos.pop(room, None)

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            kv_chunk = None
            try:
                kv_chunk = queue.get()
                logger.debug(
                    "[TRANSFER_WORKER] Got transfer request for room %s, is_last=%s, kv_indices_len=%s",
                    kv_chunk.room,
                    kv_chunk.is_last,
                    len(kv_chunk.prefill_kv_indices),
                )
                reqs_to_be_processed = tuple(
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else ()
                )
                try:
                    self._validate_cache_room_fanout(reqs_to_be_processed)
                except ValueError as exc:
                    self.abort_room(
                        kv_chunk.room,
                        f"invalid destination fanout metadata: {exc}",
                    )
                    continue
                required_dst_info_nums = {
                    int(req.required_dst_info_num) for req in reqs_to_be_processed
                }
                expected_dst_info_num = next(iter(required_dst_info_nums), 0)
                has_dummy = any(req.is_dummy for req in reqs_to_be_processed)
                if (
                    len(required_dst_info_nums) != 1
                    or expected_dst_info_num <= 0
                    or len(reqs_to_be_processed) != expected_dst_info_num
                    or has_dummy
                    and not all(req.is_dummy for req in reqs_to_be_processed)
                ):
                    self.abort_room(
                        kv_chunk.room,
                        "incomplete or inconsistent destination fanout metadata",
                    )
                    continue
                polls = []
                dst_ranks_infos = []
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            session_failed = self._is_session_failed(
                                req.mooncake_session_id
                            )
                        if session_failed:
                            logger.info(
                                "Blocked transfer due to failed session "
                                "(room=%s, session=%s).",
                                kv_chunk.room,
                                req.mooncake_session_id,
                            )
                            self.abort_room(
                                kv_chunk.room,
                                "Decode instance could be dead, remote Mooncake "
                                f"session {req.mooncake_session_id} is not alive",
                            )
                            break
                        try:
                            resolved = self.resolve_transfer_indices(kv_chunk, req)
                        except ValueError as exc:
                            logger.exception(
                                "Rejecting malformed transfer metadata for room=%s",
                                kv_chunk.room,
                            )
                            self.abort_room(
                                kv_chunk.room,
                                f"invalid transfer metadata: {exc}",
                            )
                            break

                        logger.debug(
                            "[TRANSFER_WORKER] Calling send_kvcache for room %s, session %s",
                            kv_chunk.room,
                            req.mooncake_session_id,
                        )
                        tm_start = time.monotonic()
                        prefill_metadata = None
                        dst_kv_ptrs = self.decode_kv_args_table[
                            req.mooncake_session_id
                        ].dst_kv_ptrs
                        dst_state_data_ptrs = self.decode_kv_args_table[
                            req.mooncake_session_id
                        ].dst_state_data_ptrs
                        if kv_chunk.begin_cache_step is None:
                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                resolved.src_indices,
                                dst_kv_ptrs,
                                resolved.dst_indices,
                                executor,
                                req.transfer_fragments,
                                src_page_manifest=kv_chunk.page_manifest,
                                dst_page_manifest=req.page_manifest,
                                dst_cache_layout=req.peer_cache_layout,
                            )
                        else:
                            ret = self.send_kvcache_layerwise(
                                req.mooncake_session_id,
                                resolved.src_indices,
                                dst_kv_ptrs,
                                resolved.dst_indices,
                                kv_chunk.begin_cache_step,
                                kv_chunk.layerwise_interval,
                                dst_state_data_ptrs,
                                kv_chunk.prefill_mamba_indices,
                                req.dst_mamba_indices,
                                req.transfer_fragments,
                            )
                        if ret == 0 and kv_chunk.is_last:
                            if kv_chunk.wait_for_bootstrap_token:
                                # The first decode/target-verify step consumes prefill's
                                # sampled bootstrap token and spec candidates; publish
                                # Success only after that metadata is ready.
                                prefill_metadata = self._wait_prefill_metadata(
                                    kv_chunk.room,
                                    kv_chunk.bootstrap_token,
                                    kv_chunk.spec_candidate_ids,
                                )
                            if kv_chunk.begin_cache_step is None:
                                ret = self.send_mamba_cache(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_mamba_indices,
                                    dst_state_data_ptrs,
                                    req.dst_mamba_indices,
                                    transfer_fragments=req.transfer_fragments,
                                )
                        logger.debug(
                            "[TRANSFER_WORKER] send_kvcache returned %s for room %s",
                            ret,
                            kv_chunk.room,
                        )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self._mark_session_failed(
                                        req.mooncake_session_id, reason="send_kvcache"
                                    )
                                    logger.error(
                                        "Session %s failed.", req.mooncake_session_id
                                    )
                            self.abort_room(
                                kv_chunk.room,
                                f"Failed to send KV chunk of {kv_chunk.room} "
                                f"to {req.endpoint}:{req.dst_port}",
                            )
                            break

                        if kv_chunk.is_last:
                            polls.append(True)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == expected_dst_info_num:
                                status = (
                                    TransferPoll.Success
                                    if all(polls)
                                    else TransferPoll.Failed
                                )
                                self.update_status(req.room, status)
                                # bootstrap_token is carried directly in the chunk (set by
                                # DisaggPrefillExecutor._decode after prefill forward).
                                if kv_chunk.wait_for_bootstrap_token:
                                    if prefill_metadata is None:
                                        prefill_metadata = self._wait_prefill_metadata(
                                            kv_chunk.room,
                                            kv_chunk.bootstrap_token,
                                            kv_chunk.spec_candidate_ids,
                                        )
                                    bootstrap_token, spec_candidate_ids = (
                                        prefill_metadata
                                    )
                                else:
                                    bootstrap_token, spec_candidate_ids = (
                                        kv_chunk.bootstrap_token,
                                        kv_chunk.spec_candidate_ids,
                                    )
                                if self.check_status(req.room) == TransferPoll.Failed:
                                    status = TransferPoll.Failed
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint,
                                        dst_port,
                                        room,
                                        status,
                                        self.attn_tp_rank,
                                        bootstrap_token=bootstrap_token,
                                        spec_candidate_ids=spec_candidate_ids,
                                    )
                        elapsed_seconds = time.monotonic() - tm_start
                        if self.kv_transfer_metrics:
                            self.kv_transfer_metrics.observe_kv_transfer_latency(
                                elapsed_seconds
                            )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, TransferPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == TransferPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except ValueError as exc:
                if kv_chunk is None:
                    raise RuntimeError(
                        "Transfer thread failed before receiving a request"
                    ) from exc
                logger.exception(
                    "Rejecting invalid transfer for room=%s", kv_chunk.room
                )
                self.abort_room(
                    kv_chunk.room,
                    f"invalid transfer metadata or route: {exc}",
                )
                continue
            except Exception as exc:
                raise RuntimeError(
                    f"Transfer thread failed because of {exc}. Prefill instance "
                    f"with bootstrap_port={self.bootstrap_port} is dead."
                ) from exc

    def start_prefill_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # TransferPoll.Bootstrapping -> TransferPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                try:
                    room = waiting_req_bytes[0].decode("ascii")
                    mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                except (IndexError, UnicodeError):
                    logger.exception(
                        "Rejecting malformed Mooncake bootstrap message header"
                    )
                    continue
                logger.info(
                    "[Prefill bootstrap_thread] recv multipart: room=%s session_id=%s",
                    room,
                    mooncake_session_id,
                )
                if room == "None":
                    try:
                        register_info = KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    except (IndexError, UnicodeError, ValueError):
                        logger.exception(
                            "Rejecting malformed KV args registration for session=%s",
                            mooncake_session_id,
                        )
                        continue
                    self.decode_kv_args_table[mooncake_session_id] = register_info
                    with self.session_lock:
                        self._clear_failed_session(mooncake_session_id)
                    logger.info(
                        "[Prefill bootstrap_thread] registered kv_args from decode session=%s",
                        mooncake_session_id,
                    )
                    continue
                else:
                    parsed_room = None
                    try:
                        parsed_room = int(room)
                        required_dst_info_num = int(
                            waiting_req_bytes[6].decode("ascii")
                        )
                        transfer_info = TransferInfo.from_zmq(waiting_req_bytes)
                        if mooncake_session_id not in self.decode_kv_args_table:
                            raise ValueError(
                                "pre-allocation references an unregistered "
                                "Decode session"
                            )
                        if transfer_info.peer_cache_layout is not None:
                            self._validate_cache_route(transfer_info)
                        candidate_infos = dict(self.transfer_infos.get(parsed_room, {}))
                        candidate_infos[mooncake_session_id] = transfer_info
                        if len(candidate_infos) > required_dst_info_num:
                            raise ValueError(
                                "pre-allocation exceeds destination fanout"
                            )
                        if len(candidate_infos) == required_dst_info_num:
                            self._validate_cache_room_fanout(
                                tuple(candidate_infos.values())
                            )
                    except (IndexError, UnicodeError, ValueError) as exc:
                        logger.exception(
                            "Rejecting malformed pre-allocation metadata for room=%s",
                            room,
                        )
                        if parsed_room is not None:
                            self.abort_room(
                                parsed_room,
                                f"invalid pre-allocation metadata: {exc}",
                            )
                            try:
                                self.sync_status_to_decode_endpoint(
                                    waiting_req_bytes[1].decode("ascii"),
                                    int(waiting_req_bytes[2].decode("ascii")),
                                    parsed_room,
                                    TransferPoll.Failed,
                                    self.attn_tp_rank,
                                )
                            except Exception:
                                logger.exception(
                                    "Could not notify malformed Decode peer for "
                                    "room=%s",
                                    parsed_room,
                                )
                        continue
                    room = parsed_room
                    self.transfer_infos[room] = candidate_infos
                    logger.info(
                        "[Prefill bootstrap_thread] pre-alloc received: room=%d session=%s got=%d/%d, status -> %s",
                        room,
                        mooncake_session_id,
                        len(self.transfer_infos[room]),
                        required_dst_info_num,
                        (
                            "Bootstrapped"
                            if len(self.transfer_infos[room]) == required_dst_info_num
                            else "waiting more"
                        ),
                    )
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, TransferPoll.Bootstrapped)

        threading.Thread(target=bootstrap_thread).start()

    def start_transfer_thread(
        self, transfer_thread_pool_size: int, transfer_queue_size: int
    ):
        self.transfer_queues: list[FastQueue] = [
            FastQueue() for _ in range(transfer_queue_size)
        ]
        self.executors = [
            concurrent.futures.ThreadPoolExecutor(
                transfer_thread_pool_size // transfer_queue_size
            )
            for _ in range(transfer_queue_size)
        ]
        for queue, executor in zip(self.transfer_queues, self.executors):
            threading.Thread(
                target=self.transfer_worker, args=(queue, executor), daemon=True
            ).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        aux_index: int | None = None,
        bootstrap_token: int = -1,
        begin_cache_step: int | None = None,
        layerwise_interval: int = 1,
        wait_for_bootstrap_token: bool = False,
        mamba_indices: npt.NDArray[np.int64] | None = None,
        spec_candidate_ids: list[int] | None = None,
        page_manifest: CachePDPageManifest | None = None,
    ):
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            raise RuntimeError("Transfer requests can only be added in prefill mode.")
        if is_last and aux_index is None:
            raise ValueError("aux_index must be set for the last transfer chunk.")
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == TransferPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        #  sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
                bootstrap_token=bootstrap_token,
                begin_cache_step=begin_cache_step,
                layerwise_interval=layerwise_interval,
                wait_for_bootstrap_token=wait_for_bootstrap_token,
                prefill_mamba_indices=mamba_indices,
                spec_candidate_ids=spec_candidate_ids,
                page_manifest=page_manifest,
            )
        )

    def receive_decode_prefix_info(self, bootstrap_room: int) -> int:
        """Receive decode prefix info from decode side"""
        # In mooncake implementation, decode_prefix_len is handled via ZMQ messages
        # Check the stored transfer info for this room
        if bootstrap_room in self.transfer_infos:
            for transfer_info in self.transfer_infos[bootstrap_room].values():
                if (
                    hasattr(transfer_info, "decode_prefix_len")
                    and transfer_info.decode_prefix_len > 0
                ):
                    logger.debug(
                        "Found decode_prefix_len=%s for room %s",
                        transfer_info.decode_prefix_len,
                        bootstrap_room,
                    )
                    return transfer_info.decode_prefix_len
        logger.debug("No decode_prefix_len found for room %s, using 0", bootstrap_room)
        return 0

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        if self.dist_init_addr:
            ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0])
        else:
            ip_address = get_ip()

        bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Prefill",
            "world_size": self.world_size,
            "dp_size": self.dp_size,
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "engine_rank": self.kv_args.engine_rank,
            "kv_item_lens": self.kv_args.kv_item_lens,
            "kv_unit_lens": getattr(self.kv_args, "kv_unit_lens", []),
            "state_item_lens": self.kv_args.state_item_lens,
            "state_unit_lens": getattr(self.kv_args, "state_unit_lens", []),
        }
        if self.kv_args.cache_layout is not None:
            payload["cache_layout"] = self.kv_args.cache_layout.to_wire_bytes().decode(
                "ascii"
            )

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    "Prefill instance failed to connect to bootstrap server: %s, %s",
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.error(
                "Prefill instance failed to register with bootstrap server: %s", exc
            )


from tokenspeed.runtime.pd.mooncake.sender import MooncakeKVSender

__all__ = ["MooncakeKVManagerPrefill", "MooncakeKVSender"]
