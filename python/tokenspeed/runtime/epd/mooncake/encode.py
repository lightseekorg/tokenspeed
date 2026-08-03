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

"""Encode-side (data source) manager for the Mooncake embedding transfer.

Split out of :mod:`tokenspeed.runtime.epd.mooncake.embedding_transfer`; the
per-request sender it drives lives in
:mod:`tokenspeed.runtime.epd.mooncake.sender`.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time

import requests
import zmq

from tokenspeed.runtime.epd.entities import (
    REGISTER_ROOM_SENTINEL,
    EmbeddingArgs,
    EmbeddingChunk,
    EmbeddingManagerArgs,
    EmbeddingTransferInfo,
)
from tokenspeed.runtime.epd.mooncake.conn import (
    MooncakeEmbeddingManagerBase,
)
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.utils import DisaggregationMode, FastQueue
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.network import get_free_port, get_local_ip_by_remote

logger = get_colorful_logger(__name__)


def _b(value: object) -> bytes:
    return str(value).encode("ascii")


def validate_fanout_frames(
    infos: list[EmbeddingTransferInfo], chunk: "EmbeddingChunk"
) -> str | None:
    """Pre-send contract check over a room's full fanout set; None when valid.

    Per frame: hidden/dtype must match the chunk; ``span`` (every frame carries
    the image's full row count) must equal the chunk's token count (the
    unchecked RDMA write silently truncates on divergence); the shard must stay
    inside the chunk's rows; and deepstack presence must agree on both sides.
    The set must then be either ALL identity (each frame covers the full span --
    full-copy broadcast) or a proper shard set whose non-empty shards, sorted by
    ``row_start``, tile a contiguous range disjointly (an encode rank under
    encode_tp>1 serves a contiguous BLOCK of the global shards). A gap/overlap
    means the two sides' shard math diverged, so the room must fail loud.
    """
    for info in infos:
        if not (info.hidden == chunk.hidden and info.dtype == chunk.dtype):
            return "embedding shape/dtype contract violated"
        if info.span != chunk.n_tokens:
            return (
                f"image token-count contract violated: receiver expects "
                f"{info.span} rows, encode has {chunk.n_tokens} (G2)"
            )
        if (
            info.row_start < 0
            or info.n_tokens < 0
            or info.row_start + info.n_tokens > chunk.n_tokens
        ):
            return (
                f"embedding shard out of range: rows [{info.row_start}, "
                f"{info.row_start + info.n_tokens}) of {chunk.n_tokens}"
            )
        if info.has_deepstack and not chunk.deepstack_width:
            return (
                "receiver expects deepstack but the chunk carries none "
                "(encode-side cache hit without its deepstack half?)"
            )
        if chunk.deepstack_width and not info.has_deepstack:
            return "chunk carries deepstack but the receiver did not allocate for it"
    if all(i.row_start == 0 and i.n_tokens == chunk.n_tokens for i in infos):
        return None  # identity (full-copy) mode
    shards = sorted((i.row_start, i.n_tokens) for i in infos if i.n_tokens > 0)
    for (start, count), (next_start, _next_count) in zip(shards, shards[1:]):
        if start + count != next_start:
            return (
                "embedding shard frames do not tile contiguously: "
                f"[{start}, {start + count}) then [{next_start}, ...)"
            )
    return None


def shard_payload(
    chunk: "EmbeddingChunk", info: EmbeddingTransferInfo
) -> tuple[int, int, int, int]:
    """Source pointer/length math for one receiver's row shard of a chunk.

    Returns ``(src_ptr, nbytes, src_deepstack_ptr, deepstack_nbytes)`` for the
    rows ``[info.row_start, info.row_start + info.n_tokens)``. Identity frames
    reproduce the whole-chunk payload exactly. Row strides are derived from the
    chunk's own byte counts so dtype size never needs decoding here.
    """
    row_bytes = chunk.nbytes // chunk.n_tokens if chunk.n_tokens else 0
    src = chunk.src_embedding_ptr + info.row_start * row_bytes
    nbytes = info.n_tokens * row_bytes
    deep_src = deep_nbytes = 0
    if chunk.deepstack_width and chunk.deepstack_nbytes:
        deep_row_bytes = (
            chunk.deepstack_nbytes // chunk.n_tokens if chunk.n_tokens else 0
        )
        deep_src = chunk.src_deepstack_ptr + info.row_start * deep_row_bytes
        deep_nbytes = info.n_tokens * deep_row_bytes
    return src, nbytes, deep_src, deep_nbytes


class MooncakeEmbeddingManagerEncode(MooncakeEmbeddingManagerBase):
    """Encode-side (data source) manager: registers itself to the bootstrap
    server so the prefill receiver can find it, listens for the receiver's
    registration + per-request pre-alloc frames, and a worker thread issues the
    one-sided Mooncake write of the embedding.
    """

    def __init__(self, args: EmbeddingManagerArgs, embedding_args: EmbeddingArgs):
        super().__init__(args, embedding_args, DisaggregationMode.ENCODE)
        # room -> {receiver_session -> EmbeddingTransferInfo}
        self.transfer_infos: dict[int, dict[str, EmbeddingTransferInfo]] = {}
        # Bootstrap registration-wait timeout (default 120s).
        self.bootstrap_time_out = envs.TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        # room -> [(deadline, chunk), ...] chunks popped before the receiver
        # registered; flushed by the bootstrap thread once its info arrives.
        self._pending: dict[int, list[tuple]] = {}
        self._pending_lock = threading.Lock()
        # zmq sockets are not thread-safe and status pushes come from any
        # fanout-pool thread: keep one PUSH socket per (thread, endpoint).
        self._status_tls = threading.local()
        self._start_bootstrap_thread()
        self._register_to_bootstrap()
        # K queues sharded by ROOM: a single consumer per room preserves the
        # park/flush/straggler-drop ordering the parking machinery relies on.
        # Each queue's worker uses a pool to issue one room's per-receiver writes
        # concurrently (each prefill rank is a distinct Mooncake session, so a
        # single batch call cannot span them).
        cpu_count = os.cpu_count() or 8
        pool_size = envs.TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE.get_set_value_or(
            min(max(4, int(0.75 * cpu_count) // 8), 12)
        )
        queue_count = envs.TOKENSPEED_DISAGGREGATION_QUEUE_SIZE.get()
        assert pool_size >= queue_count, (
            f"TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE={pool_size} must be >= "
            f"TOKENSPEED_DISAGGREGATION_QUEUE_SIZE={queue_count}"
        )
        self._queues: list[FastQueue] = [FastQueue() for _ in range(queue_count)]
        self._executors = [
            concurrent.futures.ThreadPoolExecutor(max(1, pool_size // queue_count))
            for _ in range(queue_count)
        ]
        for queue, executor in zip(self._queues, self._executors):
            threading.Thread(
                target=self._transfer_worker, args=(queue, executor), daemon=True
            ).start()
        threading.Thread(target=self._park_reaper, daemon=True).start()

    def _start_bootstrap_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def loop():
            while True:
                msg = self.server_socket.recv_multipart()
                # A malformed frame must not kill this daemon thread: a dead
                # listener drops every later registration, parking all chunks
                # until the reaper fails them. Log and continue.
                try:
                    if msg[0].decode("ascii") == REGISTER_ROOM_SENTINEL:
                        # registration frame: consumed, no per-session state needed
                        pass
                    else:
                        info = EmbeddingTransferInfo.from_zmq(msg)
                        self.transfer_infos.setdefault(info.room, {})[
                            info.mooncake_session_id
                        ] = info
                        if (
                            len(self.transfer_infos[info.room])
                            >= info.required_dst_info_num
                        ):
                            self.update_status(info.room, TransferPoll.Bootstrapped)
                        # Re-drive any chunk parked before this room registered.
                        self._flush_pending(info.room)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "dropping malformed embedding bootstrap frame "
                        "(%d parts): %s",
                        len(msg),
                        exc,
                    )

        threading.Thread(target=loop, daemon=True).start()

    def _register_to_bootstrap(self):
        ip = get_local_ip_by_remote()
        bootstrap_host = self.args.bootstrap_host or ip
        payload = {
            "role": "Prefill",  # bootstrap-server role string for "discoverable data source"
            "world_size": self.world_size,
            "dp_size": self.dp_size,
            "rank_ip": ip,
            "rank_port": self.rank_port,
            "engine_rank": self.embedding_args.engine_rank,
        }
        url = f"http://{bootstrap_host}:{self.bootstrap_port}/route"
        # The bootstrap HTTP server starts concurrently with this call, so the
        # first PUTs can race it and hit connection-refused. A dropped
        # registration leaves /route's parallel-info null and the encode silently
        # serves no embeddings, so retry until the server accepts it.
        last_err = None
        for _ in range(60):
            try:
                resp = requests.put(url, json=payload, timeout=5)
                if resp.ok:
                    return
                last_err = f"status {resp.status_code}"
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(0.5)
        logger.error(
            "encode failed to register to bootstrap server after retries: %s", last_err
        )

    def add_transfer_request(self, room: int, chunk: EmbeddingChunk) -> None:
        if (
            room not in self.request_status
            or self.check_status(room) == TransferPoll.Failed
        ):
            return
        self._queue_for(room).put(chunk)

    def _queue_for(self, room: int) -> FastQueue:
        # Room-affinity sharding: one consumer per room keeps the park /
        # _flush_pending re-enqueue / Success-straggler-drop sequence
        # single-threaded per room.
        return self._queues[room % len(self._queues)]

    def _transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            chunk: EmbeddingChunk = queue.get()
            # Already concluded: a re-enqueued straggler copy from _flush_pending
            # (which re-puts a parked chunk on each receiver registration). The
            # room was already sent + popped, so drop it.
            if self.request_status.get(chunk.room) == TransferPoll.Success:
                continue
            # Snapshot via list(): the bootstrap listener inserts into this inner
            # dict without _pending_lock, so next(iter(...)) could raise "dict changed
            # size during iteration". list() is GIL-atomic.
            info_vals = list(self.transfer_infos.get(chunk.room, {}).values())
            # required_dst_info_num (= fanout) is identical across the N frames for a
            # room, so read it off any registered frame; None until one registers.
            need = info_vals[0].required_dst_info_num if info_vals else None
            if (need is None or len(info_vals) < need) and self.request_status.get(
                chunk.room
            ) != TransferPoll.Failed:
                # Fewer than N receivers registered (1->N broadcast not yet complete):
                # park (don't drop) until the full set arrives. _flush_pending re-drives
                # this chunk on every new registration; the reaper fails it on timeout.
                parked = True
                with self._pending_lock:
                    info_vals = list(self.transfer_infos.get(chunk.room, {}).values())
                    need = info_vals[0].required_dst_info_num if info_vals else None
                    if info_vals and need is not None and len(info_vals) >= need:
                        # Full set landed inside the park window: send below instead.
                        parked = False
                    else:
                        self._pending.setdefault(chunk.room, []).append(
                            (time.time() + self.bootstrap_time_out, chunk)
                        )
                if parked:
                    continue
            # All N receivers registered: serve every one concurrently (full copy
            # in identity mode, its row shard otherwise) -- each is a distinct
            # Mooncake session, so concurrency must come from threads.
            infos = list(self.transfer_infos.get(chunk.room, {}).values())
            err = validate_fanout_frames(infos, chunk)
            if err is not None:
                self._fail_room(chunk.room, err, infos)
                continue
            # Wait the ring copy's completion HERE on the daemon (not the
            # encode-loop thread): the one-sided RDMA reads in _send touch GPU
            # memory off any CUDA stream, so they must not precede the device copy
            # that filled the slot (ViT->send corruption hazard). One wait per
            # chunk covers all N receivers (same slot); event.synchronize()
            # releases the GIL while waiting.
            if chunk.copy_event is not None:
                chunk.copy_event.synchronize()
            futures = [executor.submit(self._send, info, chunk) for info in infos]
            failed = False
            for info, future in zip(infos, futures):
                ret = future.result()  # _send returns <0 on error, never raises
                if ret != 0:
                    self.record_failure(chunk.room, f"mooncake transfer ret={ret}")
                    failed = True
                else:
                    # Per-receiver completion sync (each of the N prefill ranks gets
                    # exactly one, matching its required_response_num == 1).
                    self._sync_status(info, TransferPoll.Success)
            if failed:
                # Push Failed to EVERY receiver (idempotent on ones already
                # Success'd: the rank-synced admission MIN aborts the request
                # everywhere anyway).
                self._fail_room(chunk.room, None, infos)
                continue
            # All N receivers served: mark the room Success ONCE and pop only now,
            # so a partial fanout can never conclude/pop early.
            self.update_status(chunk.room, TransferPoll.Success)
            self.transfer_infos.pop(chunk.room, None)

    def _fail_room(
        self,
        room: int,
        reason: str | None,
        infos: list[EmbeddingTransferInfo],
    ) -> None:
        if reason is not None:
            self.record_failure(room, reason)
        self.update_status(room, TransferPoll.Failed)
        for info in infos:
            self._sync_status(info, TransferPoll.Failed)

    def is_parked(self, room: int) -> bool:
        """Whether ``room`` still has a chunk parked awaiting receiver
        registration. Public probe for the executor's ring-slot lease: a parked
        chunk holds its slot's pointer for re-send, so the slot is not reusable
        until the room unparks."""
        with self._pending_lock:
            return room in self._pending

    def fail_room(self, room: int, reason: str | None) -> None:
        """Conclude ``room`` Failed and push Failed to all of its registered
        receivers. Public seam for the encode executor, which must not reach into
        ``transfer_infos`` or the status FSM directly."""
        infos = list(self.transfer_infos.get(room, {}).values())
        self._fail_room(room, reason, infos)

    def _flush_pending(self, room: int) -> None:
        # Re-enqueue this room's parked chunks (in order) now that it registered.
        # Same room -> same queue, so the re-driven copy is consumed by the same
        # single worker that parked it.
        with self._pending_lock:
            parked = self._pending.pop(room, [])
        for _deadline, chunk in parked:
            self._queue_for(room).put(chunk)

    def _park_reaper(self):
        # Fail rooms whose parked chunk outlived bootstrap_time_out (the
        # receiver never registered), so an aborted request never hangs.
        while True:
            time.sleep(1.0)
            now = time.time()
            with self._pending_lock:
                expired = [
                    room
                    for room, parked in self._pending.items()
                    if parked and now >= parked[0][0]
                ]
                for room in expired:
                    self._pending.pop(room, None)
            for room in expired:
                self.record_failure(
                    room, f"receiver never registered within {self.bootstrap_time_out}s"
                )
                self.update_status(room, TransferPoll.Failed)

    def _send(self, info: EmbeddingTransferInfo, chunk: EmbeddingChunk) -> int:
        src, nbytes, deep_src, deep_nbytes = shard_payload(chunk, info)
        if nbytes == 0:
            # Zero-row shard (image span < shard count): nothing to write, but
            # the frame already served as this receiver's registration and it
            # still gets its Success push so its poll() completes.
            return 0
        bufs = [src]
        dsts = [info.dst_embedding_ptr]
        lens = [nbytes]
        if deep_nbytes and info.dst_deepstack_ptr:
            bufs.append(deep_src)
            dsts.append(info.dst_deepstack_ptr)
            lens.append(deep_nbytes)
        # Always go through the BATCH transfer API, even for a single buffer: its
        # binding releases the GIL for the one-sided RDMA write
        # (gil_scoped_release), while singular transfer_sync_write does not
        # reliably, pinning the GIL for the whole write and freezing this daemon's
        # loop + other daemons. A 1-element batch is byte-for-byte the same
        # transfer.
        return self.engine.batch_transfer_sync(
            info.mooncake_session_id, bufs, dsts, lens
        )

    def _sync_status(self, info: EmbeddingTransferInfo, status: int) -> None:
        # One socket per (thread, endpoint): pushes come from fanout-pool threads
        # and zmq sockets are not thread-safe.
        socks = getattr(self._status_tls, "socks", None)
        if socks is None:
            socks = self._status_tls.socks = {}
        endpoint = f"tcp://{info.endpoint}:{info.dst_port}"
        sock = socks.get(endpoint)
        if sock is None:
            sock = zmq.Context.instance().socket(zmq.PUSH)
            sock.connect(endpoint)
            socks[endpoint] = sock
        sock.send_multipart(
            [_b(info.room), _b(int(status)), _b(self.embedding_args.engine_rank)]
        )
