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

"""EPD prefill-side receive glue: fill each multimodal item's ``encoded``
tensor from the Mooncake transfer instead of running the vision tower.

Inverse of ``encode_executor.assign_encoded_embeddings``: the encode worker
row-splits the tower output per item and column-splits the deepstack half; here
the prefill allocates a receive buffer sized to each item's post-merge token
count (and deepstack width), registers it with that item's encode worker via
:class:`MooncakeEmbeddingReceiver`, waits for the push, and assigns the result
onto ``item.encoded`` / ``item.encoded_deepstack`` -- the ``skip-ViT`` form the
prefill ``VisionEmbedder`` consumes (it never re-encodes an item whose
``encoded`` is already set). A wrong token count or deepstack width mis-sizes the
RDMA target (silent corruption), so buffer-sizing + assignment is isolated and
CPU-unit-testable with a fake receiver.

Poll-driven state machine (:class:`EmbeddingReceiveJob`): ``start()`` does only
the non-blocking Phase-1 allocation + receiver construction (it must NEVER block
the prefill event loop); ``poll()`` advances every receiver a little each cycle
and publishes ``item.encoded`` only once EVERY handshaked item has landed. The
scheduler holds the job and does not admit the request to a prefill forward until
``poll()`` returns ``DONE``.

``receive_encoded_embeddings`` is a thin BLOCKING wrapper (start + spin poll) for
synchronous callers and the CPU tests.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

from tokenspeed.runtime.epd.mooncake.receiver import (
    MooncakeEmbeddingReceiver,
)
from tokenspeed.runtime.multimodal.embedder import _item_token_count
from tokenspeed.runtime.multimodal.inputs import MultimodalDataItem
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.utils.env import envs

# (manager, bootstrap_addr, bootstrap_room) -> receiver. Defaults to the real
# MooncakeEmbeddingReceiver; overridden in tests with a fake.
ReceiverFactory = Callable[[Any, str, int], Any]

# poll() return values (the per-job lifecycle status, distinct from the
# per-receiver Poll status it aggregates).
PENDING = "pending"
DONE = "done"
FAILED = "failed"

# Deregistering a receive MR the instant its Success notif arrives races the
# NIC's DMA placement tail (Success is the transport ACK, not proof the last
# bytes cleared HCA->PCIe), tripping `local access violation work queue error`.
# So deregistration is DEFERRED: entries hold the tensor ref (the allocator
# cannot reuse a still-registered address -- the no-double-register invariant)
# and are swept after a grace period, on the scheduler loop (no thread, no lock).
_DEREG_DELAY_S = 0.5
_RECV_POOL_QUARANTINE_S = 10.0
# (due_monotonic, engine, [(tensor, ptr), ...]) in due order (delay is constant).
_pending_dereg: deque = deque()


def _record_current_stream_event(tensor: torch.Tensor) -> torch.cuda.Event | None:
    if not tensor.is_cuda:
        return None
    event = torch.cuda.Event()
    torch.cuda.current_stream(tensor.device).record_event(event)
    return event


def _lazy_deregister(engine: Any, tensors: list[tuple[torch.Tensor, int]]) -> None:
    _pending_dereg.append((time.monotonic() + _DEREG_DELAY_S, engine, tensors))
    _sweep_deregister()


def _sweep_deregister() -> None:
    now = time.monotonic()
    while _pending_dereg and _pending_dereg[0][0] <= now:
        _, engine, tensors = _pending_dereg.popleft()
        for _tensor, ptr in tensors:
            try:
                engine.deregister(ptr)
            except Exception:  # noqa: BLE001 -- best-effort; worst case the MR leaks
                pass


class _RecvBufferPool:
    """Pre-registered, lifetime-stable receive slots for E->P embedding lands.

    With per-request buffers, deregistering then re-registering a recycled
    allocator address mints a NEW rkey for the same range while the encode side's
    Mooncake segment cache still resolves it to the OLD rkey -- every reuse risks
    a `local access violation work queue error` that kills the QP. The pool
    registers ONE region for the engine's lifetime, so the sender's cached
    mapping can never go stale; requests lease slots, the publish path clones the
    landed rows out, and CUDA-backed slots return only after that clone completes.

    Failure path: a FAILED job may still have an in-flight remote write targeting
    its slot, which under a single lifetime MR would land SILENTLY in the next
    tenant's data, so failed slots sit in quarantine until the transfer layer's
    timeouts have expired.

    Single-threaded by design: all callers run on the scheduler loop (no locks).
    """

    def __init__(self, engine: Any, device: Any, slot_bytes: int, n_slots: int):
        self.engine = engine
        self.slot_bytes = slot_bytes
        self.buf = torch.empty(n_slots * slot_bytes, dtype=torch.uint8, device=device)
        engine.register(self.buf.data_ptr(), self.buf.numel())
        self._free = list(range(n_slots))
        self._quarantine: deque = deque()  # (release_due_monotonic, slot)
        self._pending_release: deque = deque()  # (cuda_event, slot)

    def _sweep_pending_release(self) -> None:
        kept = deque()
        while self._pending_release:
            event, slot = self._pending_release.popleft()
            try:
                ready = event.query()
            except Exception:  # noqa: BLE001 -- avoid permanently losing a slot
                logger.warning(
                    "EPD recv pool: CUDA event query failed; releasing slot anyway",
                    exc_info=True,
                )
                ready = True
            if ready:
                self._free.append(slot)
            else:
                kept.append((event, slot))
        self._pending_release = kept

    def _sweep_quarantine(self) -> None:
        now = time.monotonic()
        while self._quarantine and self._quarantine[0][0] <= now:
            self._free.append(self._quarantine.popleft()[1])

    def sweep(self) -> None:
        self._sweep_pending_release()
        self._sweep_quarantine()

    def lease(self, nbytes: int) -> int | None:
        self.sweep()
        if nbytes > self.slot_bytes or not self._free:
            return None
        return self._free.pop()

    def view(self, slot: int, nbytes: int, dtype: torch.dtype, shape) -> torch.Tensor:
        off = slot * self.slot_bytes
        return self.buf[off : off + nbytes].view(dtype).reshape(shape)

    def release(self, slot: int) -> None:
        self._free.append(slot)

    def release_after_copy(self, slot: int, copied_tensor: torch.Tensor) -> None:
        event = _record_current_stream_event(copied_tensor)
        if event is None:
            self.release(slot)
        else:
            self._pending_release.append((event, slot))

    def quarantine(self, slot: int, delay_s: float) -> None:
        self._quarantine.append((time.monotonic() + delay_s, slot))


# (id(engine), str(device)) -> _RecvBufferPool | False (False = disabled).
_POOLS: dict = {}


def _get_pool(engine: Any, device: Any) -> _RecvBufferPool | None:
    key = (id(engine), str(device))
    pool = _POOLS.get(key)
    if pool is None:
        n_slots = envs.TOKENSPEED_EPD_RECV_POOL_SLOTS.get()
        slot_mb = envs.TOKENSPEED_EPD_RECV_POOL_SLOT_MB.get()
        if n_slots <= 0 or slot_mb <= 0:
            pool = False
        else:
            pool = _RecvBufferPool(engine, device, slot_mb << 20, n_slots)
            logger.info(
                "EPD recv pool up: %d slots x %d MB (lifetime MR)", n_slots, slot_mb
            )
        _POOLS[key] = pool
    return pool or None


def shard_rows(span: int, shard_rank: int, shard_size: int) -> tuple[int, int]:
    """Balanced contiguous row shard of ``span`` rows across ``shard_size``
    ranks: returns this rank's ``(row_start, row_count)``. The first
    ``span % shard_size`` ranks get one extra row; shards tile ``[0, span)``
    disjointly and in rank order, which BOTH sides of the transfer (the
    receiver's pre_alloc and the post-receive reassembly) must derive from this
    one function so their geometry can never diverge. ``row_count`` may be 0
    when ``span < shard_size`` (tiny images)."""
    base, rem = divmod(span, shard_size)
    start = shard_rank * base + min(shard_rank, rem)
    count = base + (1 if shard_rank < rem else 0)
    return start, count


class _ItemReceive:
    """Per-item receive bookkeeping for one :class:`EmbeddingReceiveJob`.

    Holds the single ``[n_tokens, hidden]`` receive buffer (+ optional deepstack
    column buffer) and one receiver per owned image, each targeting a contiguous
    row sub-range of that buffer. The buffers are filled in place by the encode
    side's RDMA writes; this object publishes them onto the item only once every
    one of its receivers reaches ``Success`` (handled by the owning job).
    """

    __slots__ = (
        "item",
        "recv_main",
        "recv_deepstack",
        "receivers",
        "pre_alloced",
        "main_slice_ptrs",
        "deepstack_slice_ptrs",
        "spans",
        "row_starts",
        "row_counts",
        "sharded",
        "n_tokens",
        "hidden",
        "pool",
        "pool_slot",
    )

    def __init__(
        self,
        item: MultimodalDataItem,
        recv_main: torch.Tensor,
        recv_deepstack: torch.Tensor | None,
        receivers: list[Any],
        main_slice_ptrs: list[int],
        deepstack_slice_ptrs: list[int],
        spans: list[int],
        row_starts: list[int],
        row_counts: list[int],
        sharded: list[bool],
        n_tokens: int,
        hidden: int,
        pool: _RecvBufferPool | None = None,
        pool_slot: int | None = None,
    ):
        self.item = item
        self.recv_main = recv_main
        self.recv_deepstack = recv_deepstack
        self.receivers = receivers
        self.pool = pool
        self.pool_slot = pool_slot
        # Lazy-pre_alloc latch, one per receiver (index-aligned): each receiver's
        # pre_alloc must be issued exactly once, AFTER it reports Bootstrapped, and
        # never again (a second pre_alloc would double-register the transfer).
        self.pre_alloced = [False] * len(receivers)
        self.main_slice_ptrs = main_slice_ptrs
        self.deepstack_slice_ptrs = deepstack_slice_ptrs
        self.spans = spans
        # Per-image shard geometry (index-aligned with receivers/spans): this
        # rank's row sub-range WITHIN the image, and whether the image was
        # sharded at all (identity images need no reassembly broadcast).
        self.row_starts = row_starts
        self.row_counts = row_counts
        self.sharded = sharded
        self.n_tokens = n_tokens
        self.hidden = hidden


class EmbeddingReceiveJob:
    """Poll-driven, non-blocking EPD embedding receive for ONE request.

    Usage (driven by the scheduler/event loop):

        job = start_embedding_receive(items, manager, ...)
        # ... each cycle, until DONE/FAILED:
        status = job.poll()        # PENDING | DONE | FAILED

    ``start()`` (the constructor) runs Phase-1 only: it sizes + allocates each
    handshaked item's receive buffer, registers it with the Mooncake engine, and
    constructs one :class:`MooncakeEmbeddingReceiver` on the item's room (the
    receiver's own ``__init__`` performs the bootstrap handshake). It does NOT
    wait for any transfer and does NOT call ``pre_alloc`` -- that is deferred to
    ``poll()``, which issues each ``pre_alloc`` lazily once its receiver reports
    Bootstrapped and then waits (across cycles, never blocking) for Success.

    HARD CONSTRAINT: one room/receiver PER ITEM. A single request's
    items may be served by DIFFERENT encode workers, so collapsing to one
    receiver per request is not possible. ``poll()`` returns DONE only when every
    item's receiver is Success.

    Buffer lifetime: by default the receive target is a leased slot from the
    lifetime-registered :class:`_RecvBufferPool`; on DONE the landed rows are
    cloned onto ``item.encoded`` and the slot is reused after that copy completes
    (no MR churn, see the pool docstring). Deepstack models, oversized items, pool
    exhaustion, or ``TOKENSPEED_EPD_RECV_POOL_SLOTS=0`` fall back to a per-request
    buffer registered on start and lazily deregistered after publish. Pooled
    slots stay leased until the CUDA copy into ``item.encoded`` has completed,
    so a following request cannot overwrite rows still being cloned. The GPU cost
    per request is roughly ``n_tokens * hidden * dtype.itemsize`` (plus ``* (1 +
    num_deepstack)`` with deepstack); the caller should cap the number of
    in-flight jobs accordingly.

    Idempotent re-`start`/poll: items whose ``item.encoded`` is already set
    (chunked prefill re-runs the receive per Path-4 forward on the same item) are
    SKIPPED at start time -- no receiver is constructed for them.
    """

    def __init__(
        self,
        items: Sequence[MultimodalDataItem],
        manager: Any,
        *,
        hidden: int,
        num_deepstack: int,
        dtype: torch.dtype,
        device: torch.device | str,
        receiver_factory: ReceiverFactory | None = None,
        shard_rank: int = 0,
        shard_size: int = 1,
    ):
        self.manager = manager
        self.hidden = hidden
        self.num_deepstack = num_deepstack
        # Row sharding across the attn-TP group: with shard_size > 1 this rank
        # registers only its shard_rows() sub-range of each image; after the
        # rank-agreed DONE the caller MUST run reassemble() (rank-lockstep) to
        # rebuild the full rows before any forward consumes item.encoded.
        # shard_size <= 1 is a plain full copy (no reassemble).
        self._shard_rank = shard_rank
        self._shard_size = shard_size
        # torch.dtype is used for buffer allocation (no dtype lost in a string
        # round-trip); the string is only the encode-side wire contract in pre_alloc.
        self.dtype = dtype
        self.dtype_str = str(dtype)
        self._factory: ReceiverFactory = receiver_factory or MooncakeEmbeddingReceiver
        # Terminal latch: once DONE/FAILED, poll() is a cheap no-op returning it
        # (the receivers/buffers have been torn down).
        self._status: str = PENDING

        # Phase 1 (non-blocking): for each item carrying an encode handshake
        # (``item.encode_handshake``), allocate ONE [n_tokens, hidden] buffer
        # (+ deepstack columns), register it with the Mooncake engine, and build a
        # single receiver on the item's room. The handshake lives ON the item: the
        # gateway mints one room per item and the encode worker row-splits the
        # concatenated-subgrid embedding per item -- so one item == one room == one
        # embedding.
        self._items: list[_ItemReceive] = []
        for item in items:
            handshake = getattr(item, "encode_handshake", None)
            if handshake is None:
                # No EPD-routed embedding on this item; leave it for the tower.
                continue

            # Chunked prefill re-runs the receive per forward on the SAME item;
            # ``encoded`` is set only after Success, so ``encoded is not None``
            # means fully received -- skip it (re-bootstrapping a Success room
            # would never re-report Bootstrapped and would stick). Mirrors
            # ``VisionEmbedder``, which never re-encodes an encoded item.
            if item.encoded is not None:
                continue

            self._items.append(self._start_item(item, handshake, device))

        # Nothing to receive (text-only / all-already-encoded / no EPD item):
        # the job is immediately DONE so the scheduler admits the request at once.
        if not self._items:
            self._status = DONE

    def _start_item(
        self,
        item: MultimodalDataItem,
        handshake: Mapping[str, Any],
        device: torch.device | str,
    ) -> _ItemReceive:
        """Allocate + register one item's receive buffer and construct its single
        receiver on the item's room, recording the (possibly sharded) destination
        row range so ``poll()`` can lazily issue ``pre_alloc`` once the receiver
        bootstraps. Does NOT block on any transfer (no ``_wait``, no ``pre_alloc``
        here).

        One room per item: the encode worker concatenates the item's subgrid
        tokens into a single ``[n_tokens, hidden]`` embedding and row-splits it per
        prefill TP rank, so the receive geometry spans the item's FULL token count
        (sum of its offset spans), not per offset.
        """
        elt = torch.empty(0, dtype=self.dtype).element_size()

        # The item's image spans all its concatenated subgrid tokens.
        span = _item_token_count(item)
        addr = f"{handshake['bootstrap_host']}:{handshake['bootstrap_port']}"
        # The receiver __init__ performs the bootstrap handshake (HTTP /route fetch
        # + endpoint registration); on return its poll() is already Bootstrapped or
        # Failed. We do NOT pre_alloc here -- poll() does it once Bootstrapped.
        receiver = self._factory(self.manager, addr, int(handshake["bootstrap_room"]))

        # Row sharding across the attn-TP group: this rank registers only its
        # shard_rows() sub-range, or the full image when sharding is off.
        is_sharded = self._shard_size > 1
        if is_sharded:
            row_start, row_count = shard_rows(span, self._shard_rank, self._shard_size)
        else:
            row_start, row_count = 0, span
        # Length-1 per-image lists so poll()/_packed_to_full()/reassemble() iterate
        # unchanged (one item == one image under per-item rooms).
        receivers: list[Any] = [receiver]
        spans: list[int] = [span]
        row_starts: list[int] = [row_start]
        row_counts: list[int] = [row_count]
        sharded: list[bool] = [is_sharded]

        # The RECEIVE buffer holds only THIS rank's shard rows, packed contiguously
        # (image i at rows [packed_cursor, packed_cursor + row_count)); the FULL
        # embedding is rebuilt later in item.encoded (publish scatter + reassemble),
        # which needs no MR. Registering just the shard rows -- not the full image
        # -- keeps a big multi-image item from overflowing a recv pool slot onto
        # the GIL-held per-request register fallback. With nothing sharded,
        # packed_tokens == full_tokens.
        full_tokens = _item_token_count(item)
        packed_tokens = sum(row_counts)
        nbytes = packed_tokens * self.hidden * elt
        pool = None
        pool_slot = None
        recv_main = None
        if self.num_deepstack == 0:
            # Preferred path: lease a lifetime-registered pool slot (see
            # _RecvBufferPool for why per-request register/deregister kills QPs).
            # Deepstack models keep the legacy path (a second buffer per item,
            # out of the pool's scope).
            pool = _get_pool(self.manager.engine, device)
            if pool is not None:
                pool_slot = pool.lease(nbytes)
                if pool_slot is not None:
                    recv_main = pool.view(
                        pool_slot, nbytes, self.dtype, (packed_tokens, self.hidden)
                    )
                else:
                    logger.info(
                        "EPD recv pool: no slot for %d B (free=%d); falling back "
                        "to per-request registration",
                        nbytes,
                        len(pool._free),
                    )
                    pool = None
        if recv_main is None:
            recv_main = torch.empty(
                (packed_tokens, self.hidden), dtype=self.dtype, device=device
            )
            # Legacy fallback (pool disabled/exhausted, oversized item, or
            # deepstack present): register per request; the deregister is DEFERRED
            # on the publish/fail paths (lazy queue) to soften -- not eliminate --
            # the stale-rkey window. A 0-row packed buffer has nothing to register.
            if packed_tokens > 0:
                self.manager.engine.register(
                    recv_main.data_ptr(),
                    recv_main.numel() * recv_main.element_size(),
                )
        recv_deepstack: torch.Tensor | None = None
        if self.num_deepstack > 0:
            recv_deepstack = torch.empty(
                (packed_tokens, self.hidden * self.num_deepstack),
                dtype=self.dtype,
                device=device,
            )
            if packed_tokens > 0:
                self.manager.engine.register(
                    recv_deepstack.data_ptr(),
                    recv_deepstack.numel() * recv_deepstack.element_size(),
                )

        # Destination pointers into the PACKED buffer: the encode side writes its
        # n_tokens(=row_count) source rows CONTIGUOUSLY at this pointer (it does
        # not re-apply row_start -- that only selects which SOURCE rows to read).
        # A 0-row shard still records its (empty) pointer; its pre_alloc is the
        # registration heartbeat the encode fanout gate counts.
        main_slice_ptrs: list[int] = []
        deepstack_slice_ptrs: list[int] = []
        packed_cursor = 0
        for row_count in row_counts:
            main_slice_ptrs.append(
                recv_main[packed_cursor : packed_cursor + row_count].data_ptr()
            )
            if recv_deepstack is not None:
                deepstack_slice_ptrs.append(
                    recv_deepstack[packed_cursor : packed_cursor + row_count].data_ptr()
                )
            else:
                deepstack_slice_ptrs.append(0)
            packed_cursor += row_count

        return _ItemReceive(
            item=item,
            recv_main=recv_main,
            recv_deepstack=recv_deepstack,
            receivers=receivers,
            main_slice_ptrs=main_slice_ptrs,
            deepstack_slice_ptrs=deepstack_slice_ptrs,
            spans=spans,
            row_starts=row_starts,
            row_counts=row_counts,
            sharded=sharded,
            n_tokens=full_tokens,
            hidden=self.hidden,
            pool=pool,
            pool_slot=pool_slot,
        )

    def poll(self) -> str:
        """Advance the receive state machine one cheap step.

        For every still-pending receiver: if it just reached Bootstrapped, issue
        its (single) ``pre_alloc`` so the encode side learns where to write; if it
        reached Failed -> the whole job is FAILED; otherwise leave it. When EVERY
        receiver of EVERY item is Success, deregister the buffers, publish
        ``item.encoded`` / ``item.encoded_deepstack``, and return DONE.

        Returns ``PENDING`` | ``DONE`` | ``FAILED``. Idempotent after a terminal
        result (the buffers/receivers are gone, so it just returns the latch).
        """
        _sweep_deregister()
        if self._status is not PENDING:
            return self._status

        all_success = True
        for it in self._items:
            for idx, receiver in enumerate(it.receivers):
                status = receiver.poll()
                if status == TransferPoll.Failed:
                    self._fail()
                    return FAILED
                if not it.pre_alloced[idx] and status in (
                    TransferPoll.Bootstrapped,
                    TransferPoll.Transferring,
                    TransferPoll.Success,
                ):
                    # Bootstrapped (or already further along on a fast/in-process
                    # transport): issue the one-shot pre_alloc and latch it -- a
                    # second would double-register the transfer on the encode side.
                    # In shard mode n_tokens carries this rank's row COUNT and the
                    # dst pointers are already offset to the shard's first row; a
                    # row_count of 0 is still sent (it doubles as the encode-side
                    # fanout-gate heartbeat).
                    receiver.pre_alloc(
                        dst_embedding_ptr=it.main_slice_ptrs[idx],
                        n_tokens=it.row_counts[idx],
                        hidden=it.hidden,
                        dtype=self.dtype_str,
                        dst_deepstack_ptr=it.deepstack_slice_ptrs[idx],
                        has_deepstack=self.num_deepstack > 0,
                        row_start=it.row_starts[idx],
                        span=it.spans[idx],
                    )
                    it.pre_alloced[idx] = True
                    # Re-poll once after pre_alloc: an in-process/fake transport
                    # may flip straight to Success on pre_alloc.
                    status = receiver.poll()
                    if status == TransferPoll.Failed:
                        self._fail()
                        return FAILED
                if status != TransferPoll.Success:
                    all_success = False

        if not all_success:
            return PENDING

        # Every image of every item has landed; publish and reclaim. Pooled
        # path: clone the landed rows OUT of the slot (item.encoded must outlive
        # the lease) and release the slot only after that copy completes. Legacy
        # path: hand the buffer itself to item.encoded and queue the registration
        # for DEFERRED drop (the lazy entry holds the tensor ref, so the allocator
        # cannot recycle a still-registered address).
        for it in self._items:
            any_sharded = any(it.sharded)
            if it.pool is not None:
                # Pooled path is deepstack-free. Copy the rows OUT of the slot
                # before releasing it. When sharded the buffer is packed (this
                # rank's rows only), so scatter into a full-layout tensor at each
                # image's absolute offset (reassemble fills the rest); when not
                # sharded the buffer is already full -> clone it.
                it.item.encoded = (
                    self._packed_to_full(it, it.recv_main, self.hidden)
                    if any_sharded
                    else it.recv_main.clone()
                )
                it.item.encoded_deepstack = None
                it.pool.release_after_copy(it.pool_slot, it.item.encoded)
            else:
                # Legacy path. When sharded, the packed buffers are smaller than
                # the image and cannot alias item.encoded, so build separate full
                # tensors; when not sharded, hand the (full) buffers over directly.
                if any_sharded:
                    it.item.encoded = self._packed_to_full(
                        it, it.recv_main, self.hidden
                    )
                    it.item.encoded_deepstack = (
                        self._packed_to_full(
                            it, it.recv_deepstack, self.hidden * self.num_deepstack
                        )
                        if it.recv_deepstack is not None
                        else None
                    )
                else:
                    it.item.encoded = it.recv_main
                    it.item.encoded_deepstack = it.recv_deepstack
                pairs = [(it.recv_main, it.recv_main.data_ptr())]
                if it.recv_deepstack is not None:
                    pairs.append((it.recv_deepstack, it.recv_deepstack.data_ptr()))
                _lazy_deregister(self.manager.engine, pairs)
        # Receive concluded: release each room's bookkeeping from the prefill manager.
        self._clear_receivers()
        self._status = DONE
        return DONE

    def _packed_to_full(
        self, it: "_ItemReceive", packed: torch.Tensor, width: int
    ) -> torch.Tensor:
        """Scatter a PACKED shard buffer (this rank's rows, image-contiguous) into
        a full-layout ``[n_tokens, width]`` tensor, placing each image's shard at
        its absolute row offset and leaving the non-owned rows for ``reassemble``
        to fill. The packed buffer holds ``sum(row_counts)`` rows; the full tensor
        holds ``sum(spans)`` rows (== n_tokens). Identity images (row_count==span,
        row_start==0) copy whole, so a buffer with no real sharding round-trips
        unchanged."""
        full = torch.empty(
            (it.n_tokens, width), dtype=packed.dtype, device=packed.device
        )
        packed_cursor = 0
        full_cursor = 0
        for span, row_start, row_count in zip(it.spans, it.row_starts, it.row_counts):
            if row_count > 0:
                full[full_cursor + row_start : full_cursor + row_start + row_count] = (
                    packed[packed_cursor : packed_cursor + row_count]
                )
            packed_cursor += row_count
            full_cursor += span
        return full

    def reassemble(self, nccl_group: Any, group_ranks: Sequence[int]) -> None:
        """Rebuild full embeddings from row shards via per-image broadcasts.

        Must be called RANK-LOCKSTEP on every attn-TP rank, only after the
        rank-agreed DONE (the drain's MIN all-reduce) and BEFORE any forward
        consumes ``item.encoded``: until then each rank's buffer holds only its
        own shard rows, the rest is uninitialized memory. All ranks iterate the
        identical items/images in identical order issuing identical collectives,
        which also requires the caller to run on the NON-overlap event loop (a
        second thread launching forward collectives concurrently would break the
        cross-rank launch-order guarantee NCCL needs across communicators).

        ``group_ranks`` is the attn-TP group as GLOBAL ranks
        (``mapping.attn.tp_group``): ``dist.broadcast`` takes a global src rank,
        and group rank p == global rank p only in the no-DP single-engine case.
        Per-image sub-range broadcasts (2 x shard_size x n_images launches per
        request).

        No-op for identity images and when sharding is off; safe after _fail
        (items were dropped). Idempotence is NOT required: called exactly once
        per admitted request by the drain.

        Broadcasts target ``item.encoded`` (the PUBLISHED tensor), never
        ``recv_main``: on the pooled path the publish step cloned the rows out
        and queued the slot for reuse after the clone completes, so ``recv_main``
        may later belong to the next tenant; on the legacy path ``item.encoded``
        IS ``recv_main``, so the two are equivalent there.
        """
        if self._shard_size <= 1 or self._status is not DONE:
            return
        for it in self._items:
            main = it.item.encoded
            deep = it.item.encoded_deepstack
            cursor = 0
            for idx, span in enumerate(it.spans):
                if not it.sharded[idx]:
                    cursor += span
                    continue
                for p in range(self._shard_size):
                    start, count = shard_rows(span, p, self._shard_size)
                    if count == 0:
                        continue
                    src = group_ranks[p]
                    dist.broadcast(
                        main[cursor + start : cursor + start + count],
                        src=src,
                        group=nccl_group,
                    )
                    if deep is not None:
                        dist.broadcast(
                            deep[cursor + start : cursor + start + count],
                            src=src,
                            group=nccl_group,
                        )
                cursor += span

    def _clear_receivers(self) -> None:
        """Release each receiver's per-room manager bookkeeping (terminal paths only;
        no-op for fake receivers without clear())."""
        for it in self._items:
            for receiver in it.receivers:
                clear = getattr(receiver, "clear", None)
                if clear is not None:
                    clear()

    def _fail(self) -> None:
        """Tear down on failure and drop our references. A SIBLING image's
        write may still be in flight into the item buffer, so reclamation is
        deferred on both paths: pooled slots go to quarantine (under the
        lifetime MR a late write would land SILENTLY in the next tenant's
        data), legacy buffers go to the lazy-deregistration queue.
        ``item.encoded`` is left unset -- the request is being failed, not
        served.
        """
        for it in self._items:
            if it.pool is not None:
                it.pool.quarantine(it.pool_slot, _RECV_POOL_QUARANTINE_S)
            else:
                pairs = [(it.recv_main, it.recv_main.data_ptr())]
                if it.recv_deepstack is not None:
                    pairs.append((it.recv_deepstack, it.recv_deepstack.data_ptr()))
                _lazy_deregister(self.manager.engine, pairs)
        self._clear_receivers()
        self._items = []
        self._status = FAILED

    @property
    def status(self) -> str:
        return self._status

    def release(self) -> None:
        """Best-effort teardown for an externally-driven abort.

        Used when a rank-agreed FAILED is reached but THIS rank had not yet seen a
        local Failed poll (so its buffers are still registered): free/deregister
        them so a reused allocator address is never left double-registered.
        Idempotent and a no-op once terminal (poll() already tore a DONE/FAILED job
        down). After release() the job is FAILED.
        """
        if self._status is not PENDING:
            return
        self._fail()


def start_embedding_receive(
    items: Sequence[MultimodalDataItem],
    manager: Any,
    *,
    hidden: int,
    num_deepstack: int,
    dtype: torch.dtype,
    device: torch.device | str,
    receiver_factory: ReceiverFactory | None = None,
    shard_rank: int = 0,
    shard_size: int = 1,
) -> EmbeddingReceiveJob:
    """Begin (non-blocking) the per-item EPD embedding receive for one request.

    See :class:`EmbeddingReceiveJob`. The handshake for each EPD-routed item
    rides on ``item.encode_handshake`` (a dict ``{bootstrap_room, bootstrap_host,
    bootstrap_port}``); items without one are left to the vision tower. Returns a
    job whose ``poll()`` the caller drives every cycle until DONE/FAILED; the
    request must not be admitted to a prefill forward before then. With
    ``shard_size > 1`` the caller must also run ``job.reassemble()`` rank-
    lockstep after the rank-agreed DONE, before the request's first forward.
    """
    return EmbeddingReceiveJob(
        items,
        manager,
        hidden=hidden,
        num_deepstack=num_deepstack,
        dtype=dtype,
        device=device,
        receiver_factory=receiver_factory,
        shard_rank=shard_rank,
        shard_size=shard_size,
    )


def receive_encoded_embeddings(
    items: Sequence[MultimodalDataItem],
    manager: Any,
    *,
    hidden: int,
    num_deepstack: int,
    dtype: torch.dtype,
    device: torch.device | str,
    timeout: float = 60.0,
    receiver_factory: ReceiverFactory | None = None,
) -> None:
    """BLOCKING wrapper: fill ``item.encoded`` from the per-item transfers.

    For synchronous callers and the CPU unit tests: the poll-driven
    :class:`EmbeddingReceiveJob` spun to completion in place (start, then
    busy-poll until DONE, raising on FAILED or timeout). Event-loop code should
    drive ``poll()`` once per cycle instead.

    The handshake for each EPD-routed item rides on ``item.encode_handshake``
    (``{bootstrap_room, bootstrap_host, bootstrap_port}``). ``hidden`` is the main
    embedding width, ``num_deepstack`` the deepstack level count (0 if absent),
    and ``dtype`` must match the encode worker's tower output dtype (asserted on
    the encode side before the unchecked RDMA write).
    """
    job = start_embedding_receive(
        items,
        manager,
        hidden=hidden,
        num_deepstack=num_deepstack,
        dtype=dtype,
        device=device,
        receiver_factory=receiver_factory,
    )
    deadline = time.monotonic() + timeout
    while True:
        status = job.poll()
        if status == DONE:
            return
        if status == FAILED:
            raise RuntimeError("encode->prefill embedding transfer failed")
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"embedding receive timed out after {timeout:.1f}s "
                f"(job still {status})"
            )
        time.sleep(0.0005)


def build_prefill_embedding_manager(server_args, global_rank, is_multimodal_active):
    """Stand up the EPD encode->prefill embedding sink, if applicable.

    Only a multimodal *prefill* node receives embeddings from encode workers. Built
    as one TP group (the embedding transport allows prefill_tp = any multiple of
    encode_tp) with no fixed base buffer (receive buffers are allocated per image
    at receive time). Construction spawns a daemon status thread, so build it
    exactly once per rank. Returns None for decode/encode/text-only nodes.
    """
    if server_args.disaggregation_mode != "prefill" or not is_multimodal_active:
        return None

    from tokenspeed.runtime.epd.entities import EmbeddingArgs, EmbeddingManagerArgs
    from tokenspeed.runtime.epd.mooncake.prefill import (
        MooncakeEmbeddingManagerPrefill,
    )

    emb_mgr_args = EmbeddingManagerArgs(
        bootstrap_port=server_args.disaggregation_bootstrap_port,
        tp_size=server_args.mapping.attn.tp_size,
    )
    emb_args = EmbeddingArgs(
        engine_rank=global_rank,
        gpu_id=global_rank,
        ib_device=server_args.disaggregation_ib_device,
        embedding_data_ptr=0,
        embedding_data_len=0,
    )
    return MooncakeEmbeddingManagerPrefill(emb_mgr_args, emb_args)


class EpdPrefillAdmission:
    """EPD prefill-side embedding receive + rank-synced admission.

    Owns the encode->prefill embedding sink (MooncakeEmbeddingManagerPrefill), the
    set of requests whose per-image embeddings are still arriving (``_pending``),
    and the optional NCCL row-shard reassembly. Exists ONLY on a multimodal prefill
    node; the EventLoop holds one (or None) and drives it each non-overlap cycle.

    DECIDE/ACT split: ``drain()`` polls the staged receives, applies the rank-
    lockstep MIN all-reduce + reassembly, and RETURNS ``(admitted, failed)``
    decisions. The EventLoop performs the acts those imply (P->D sender
    register/abort, scheduler submit, output-processor finish) -- they touch
    EventLoop collaborators, so they stay there.
    """

    def __init__(
        self,
        *,
        manager,
        device,
        hidden,
        num_deepstack,
        dtype,
        attn_tp_rank,
        attn_tp_size,
        attn_tp_cpu_group,
        attn_tp_group,
        pg_manager,
    ):
        self._manager = manager
        self._device = device
        self._hidden = hidden
        self._num_deepstack = num_deepstack
        self._dtype = dtype
        self._attn_tp_rank = attn_tp_rank
        self._attn_tp_size = attn_tp_size
        self._attn_tp_cpu_group = attn_tp_cpu_group
        # Requests whose per-image encode->prefill embeddings are still arriving:
        # registered but NOT yet submitted to the scheduler; drain() polls them each
        # cycle and admits (rank-synced) only once ready. Rank-identical in
        # length+order across attn-TP ranks (recv_reqs broadcasts the new-request
        # set), which drain()'s MIN all-reduce relies on.
        self._pending: list = []
        # Deadline for an EPD request's per-image embeddings to all arrive; past it
        # the request is aborted (not waited on forever). Reuse the PD KV-receive
        # wait knob (default 300s): the prefill waiting on the encode->prefill
        # embedding transfer is the direct analog of the decode waiting on the
        # prefill->decode KV transfer, so one operator knob covers both.
        self._embed_timeout: float = float(
            envs.TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT.get()
        )

        # EPD embedding row-sharding: each attn-TP rank receives only 1/N of every
        # image's rows over the wire; the full embedding is rebuilt by an NCCL
        # all-gather in drain() (job.reassemble), which runs on the prefill's
        # NON-overlap loop so the drain and the forward launch from one thread in
        # the same order on every rank -- the cross-rank launch-order consistency
        # NCCL requires across communicators.
        self._shard_embeddings = False
        self._nccl_group = None
        self._group_ranks = tuple(attn_tp_group)
        shard_flag = False
        if attn_tp_size > 1:
            shard_flag = bool(envs.TOKENSPEED_EPD_EMBEDDING_SHARD.get())
            # The flag is a per-process env read but gates a GROUP collective:
            # torn across ranks (e.g. set on one node of a multi-node prefill),
            # flag-on ranks would join the warmup broadcast below while flag-off
            # ranks never do -- a silent boot hang. Agree first, loud.
            flag_t = torch.tensor(
                [int(shard_flag), -int(shard_flag)], dtype=torch.int32
            )
            dist.all_reduce(flag_t, op=dist.ReduceOp.MIN, group=attn_tp_cpu_group)
            if flag_t[0].item() != -flag_t[1].item():
                raise RuntimeError(
                    "TOKENSPEED_EPD_EMBEDDING_SHARD differs across attn-TP ranks; "
                    "set it identically on every node of the prefill engine"
                )
        if shard_flag:
            self._nccl_group = pg_manager.get_process_group("nccl", attn_tp_group)
            self._shard_embeddings = True
            # Warm the communicator at startup: NCCL initializes lazily on the
            # first collective, and nothing else issues torch.distributed NCCL ops
            # on this group in the scheduler process -- without this the FIRST
            # admitted EPD request pays communicator init (hundreds of ms, all
            # ranks) inside the drain, and a misconfigured group would surface on a
            # customer request instead of at boot.
            warmup = torch.zeros(1, device=device)
            dist.broadcast(warmup, src=self._group_ranks[0], group=self._nccl_group)
            torch.cuda.current_stream().synchronize()
            logger.info(
                "EPD embedding row-sharding enabled (attn_tp=%d, NCCL group warm)",
                attn_tp_size,
            )

    def stage(self, request_id, mm_items) -> None:
        """Begin the non-blocking per-image embedding receive and stage it by
        request_id until its embeddings land (polled in drain()). The request payload
        (spec/state/bootstrap) stays with the caller, keyed by request_id; this
        controller tracks only the receive job (mirrors kv_transfer)."""
        job = start_embedding_receive(
            items=mm_items,
            manager=self._manager,
            hidden=self._hidden,
            num_deepstack=self._num_deepstack,
            dtype=self._dtype,
            device=self._device,
            shard_rank=self._attn_tp_rank,
            shard_size=self._attn_tp_size if self._shard_embeddings else 1,
        )
        self._pending.append((request_id, job, time.time()))

    def has_pending(self) -> bool:
        return bool(self._pending)

    def drain(self):
        """Poll staged EPD embedding receives; return ``(admitted_ids, failed_ids)``.

        poll/timeout/MIN-all-reduce/reassemble/release + ``_pending`` bookkeeping
        happen here, rank-lockstep, keyed by request_id. The caller maps the ids back
        to its staged request payloads and performs the acts (kv_transfer
        register/abort, scheduler submit, output-processor finish).

        - admitted_ids: DONE on every rank, reassembled.
        - failed_ids:   FAILED/timed-out, job released.
        """
        if not self._pending:
            return [], []  # rank-identical emptiness -> all ranks skip the collective

        code_of = {FAILED: 0, PENDING: 1, DONE: 2}
        codes = [code_of[job.poll()] for (_rid, job, _ts) in self._pending]

        # Timeout: a still-PENDING request whose per-image embeddings have not all
        # arrived within the deadline is marked FAILED (-> rank-agreed abort below).
        # Without this the prefill waits FOREVER if an embedding is ever lost (a
        # degraded/dead encode worker, a network drop). Folded into the SAME MIN
        # all-reduce: a timed-out job -> code 0 -> all ranks abort it together;
        # union-of-timeout is rank-safe even if ranks cross the deadline a cycle
        # apart (any rank's 0 propagates via MIN).
        _now = time.time()
        for _i in range(len(self._pending)):
            if codes[_i] == 1 and (_now - self._pending[_i][2]) > self._embed_timeout:
                codes[_i] = 0
                logger.warning(
                    "EPD embedding receive timed out after %.0fs for rid=%s; aborting",
                    self._embed_timeout,
                    self._pending[_i][0],
                )

        if self._attn_tp_size > 1:
            t = torch.tensor(codes, dtype=torch.uint8, device="cpu")
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self._attn_tp_cpu_group)
            codes = t.tolist()

        admitted = []
        failed = []
        leftover = []
        for (request_id, job, start_ts), code in zip(self._pending, codes):
            if code == 2:  # DONE on every rank
                # Sharded receive: rebuild the full rows from the per-rank shards
                # FIRST -- item.encoded is shard-only until this runs. Rank-lockstep-
                # safe: codes are identical post-MIN and _pending is rank-identical
                # in length/order, so every rank issues identical collectives in
                # identical order this cycle.
                if self._shard_embeddings:
                    job.reassemble(self._nccl_group, self._group_ranks)
                admitted.append(request_id)
            elif code == 0:  # FAILED/timed-out on some rank -> abort everywhere
                job.release()
                failed.append(request_id)
            else:  # still pending on some rank
                leftover.append((request_id, job, start_ts))
        self._pending = leftover
        return admitted, failed


def make_epd_prefill_admission(
    server_args,
    global_rank,
    *,
    model_config,
    model_executor,
    mapping,
    attn_tp_rank,
    attn_tp_size,
    attn_tp_cpu_group,
    pg_manager,
):
    """Build the EPD-prefill admission controller, or None for non-EPD nodes.

    Returns None unless this is a multimodal *prefill* node (the only node that
    receives encode->prefill embeddings)."""
    manager = build_prefill_embedding_manager(
        server_args, global_rank, model_config.is_multimodal_active
    )
    if manager is None:
        return None
    # Extract the narrow model facts the controller needs (vision dtype, hidden
    # width, deepstack width, device) here -- the controller holds these, not the
    # whole model_executor (mirrors how create_kv_transfer takes a kv_args struct).
    model = model_executor.model_runner.model
    return EpdPrefillAdmission(
        manager=manager,
        device=model_executor.device,
        hidden=model.config.hidden_size,
        num_deepstack=getattr(model, "num_deepstack_embeddings", 0),
        dtype=(getattr(model, "visual", None) or model.vision_tower).dtype,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_cpu_group=attn_tp_cpu_group,
        attn_tp_group=mapping.attn.tp_group,
        pg_manager=pg_manager,
    )
