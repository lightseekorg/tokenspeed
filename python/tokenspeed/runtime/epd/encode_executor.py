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

"""EPD encode-worker execution: run the vision tower, scatter its output back
onto each item, and hand the contiguous embeddings to the Mooncake sender.
"""

from __future__ import annotations

import logging

import torch

from tokenspeed.runtime.epd.mooncake.sender import (
    MooncakeEmbeddingSender,
)
from tokenspeed.runtime.multimodal.embedder import _item_token_count
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)


def assign_encoded_embeddings(
    items: list[MultimodalDataItem],
    output: torch.Tensor,
    model,
) -> None:
    """Scatter a packed vision-tower output onto each item, in place.

    ``output`` is the tower's ``[sum_tokens, width]`` result for ``items`` in
    order (``width = hidden`` for plain models, or ``hidden * (1 + n_deepstack)``
    for deepstack models like Qwen3.5). Each item's row span is its post-merge
    token count (``_item_token_count``); the rows are split accordingly and,
    for deepstack models, column-split via ``model.separate_deepstack_embeds``
    into the main ``[N, hidden]`` and deepstack ``[N, hidden * n_deepstack]``
    halves. Results are made contiguous because a TP-gathered tower output may
    not be, and the transfer ships raw row-major bytes.

    Sets ``item.encoded`` (and ``item.encoded_deepstack`` when the model emits
    deepstack, else ``None``), which is exactly the ``skip-ViT`` form the
    prefill-side VisionEmbedder consumes.
    """
    output = output.reshape(-1, output.shape[-1])
    per_item_tokens = [_item_token_count(item) for item in items]
    total = sum(per_item_tokens)
    if output.shape[0] != total:
        raise ValueError(
            f"vision-tower output has {output.shape[0]} rows but items sum to "
            f"{total} post-merge tokens; check the token-count / grid contract"
        )

    has_deepstack = getattr(model, "num_deepstack_embeddings", 0) > 0
    per_item_embeds = torch.split(output, per_item_tokens, dim=0)
    for item, emb in zip(items, per_item_embeds):
        if has_deepstack:
            main, deep = model.separate_deepstack_embeds(emb)
            item.encoded = main.contiguous()
            item.encoded_deepstack = deep.contiguous()
        else:
            item.encoded = emb.contiguous()
            item.encoded_deepstack = None


class DisaggEncodeExecutor:
    """Drives one encode worker: run the vision tower on a batch of items, then
    ship each item's embedding to its prefill peer over Mooncake.

    Python orchestration: this is invoked by the encode loop, not the
    C++ scheduler. ``execute`` groups items by modality, runs the tower once per
    modality via the model's ``get_image_feature`` / ``get_video_feature``,
    scatters the output onto ``item.encoded`` (see
    :func:`assign_encoded_embeddings`), and queues a transfer per item through
    the per-request :class:`MooncakeEmbeddingSender`. Each request must first be
    ``register``-ed with its prefill peer's bootstrap (host, port, room).
    """

    def __init__(
        self,
        manager,
        multimodal_model,
        device,
        *,
        ring_slots: int = 64,
        ring_bytes: int = 256 * 1024 * 1024,
    ):
        self.manager = manager
        self.model = multimodal_model
        self.device = device
        self.senders = {}
        # RDMA requires every transferred buffer to be a registered memory region,
        # and mooncake rejects OVERLAPPING registrations -- registering each
        # per-request ``item.encoded`` fails because the torch caching allocator
        # packs freed-but-still-registered tensors so a grown region straddles
        # others. Collapse every send through a fixed ring of pre-registered bounce
        # buffers: each slot is registered once at a fixed size (never grows, never
        # overlaps), and ``item.encoded`` is copied into a slot before its async
        # send. ``ring_slots`` / ``ring_bytes`` are injectable for tests and
        # env-tunable; total reservation is slots * slot_bytes PER ring (main, plus
        # deepstack if present), so depth and per-slot bytes must be sized to the
        # model and peak concurrency.
        self._ring_slots = int(
            envs.TOKENSPEED_EPD_ENCODE_RING_SLOTS.get_set_value_or(ring_slots)
        )
        slot_mb = envs.TOKENSPEED_EPD_ENCODE_RING_SLOT_MB.get()
        # Env override is in whole MiB; unset -> keep the exact ``ring_bytes`` arg.
        self._ring_bytes = slot_mb * 1024 * 1024 if slot_mb else ring_bytes
        self._main_ring = None  # lazily allocated on first send (device live by then)
        self._deep_ring = None
        self._ring_idx = 0
        # Per-slot lease: the room whose send last staged into the slot. A slot is
        # reusable only once that room's transfer is TERMINAL and not parked (a
        # parked chunk holds the slot's pointer until bootstrap_time_out and is
        # re-sent on late receiver registration; see _lease_slot), so a full ring
        # DEFERS the send rather than overwriting an in-flight slot.
        self._slot_rooms: list = [None] * self._ring_slots
        # Sends whose ViT output is ready but could not lease a free ring slot
        # (all slots still hold in-flight transfers). Retried non-blocking by
        # drain_deferred() each loop tick (a busy-wait here would GIL-starve the
        # daemon transfer-workers that free the slots and deadlock the loop).
        self._deferred_sends: list = []

    def register(self, request_id, bootstrap_host, bootstrap_port, bootstrap_room):
        self.senders[request_id] = MooncakeEmbeddingSender(
            self.manager, f"{bootstrap_host}:{bootstrap_port}", bootstrap_room
        )

    def _feature_fn(self, modality):
        # IMAGE dispatches through the model's ``image_encoder`` seam, NOT
        # ``get_image_feature`` directly: that seam is what the encoder CUDA-graph
        # wrapper overrides (see _maybe_install_encoder_cudagraph). When the graph
        # is disabled the model leaves ``image_encoder = get_image_feature`` (eager);
        # VIDEO has no captured graph -> always eager.
        if modality == Modality.IMAGE:
            return self.model.image_encoder
        if modality == Modality.VIDEO:
            return self.model.get_video_feature
        raise ValueError(f"unsupported modality for encode: {modality}")

    def execute(self, request_items: list[tuple[str, MultimodalDataItem]]) -> None:
        by_modality = {}
        for _, item in request_items:
            by_modality.setdefault(item.modality, []).append(item)
        with torch.inference_mode():
            for modality, items in by_modality.items():
                output = self._feature_fn(modality)(items)
                assign_encoded_embeddings(items, output, self.model)
        # Stage every embedding into its ring slot, then issue the async
        # Mooncake sends. See _stage_and_send for the copy/RDMA
        # overwrite-safety invariant (one CUDA event gates each transfer).
        self._stage_and_send(request_items)

    def _ensure_rings(self) -> None:
        """Lazily allocate + register the bounce-buffer ring (see ``__init__``)."""
        if self._main_ring is not None:
            return
        self._main_ring = [
            torch.empty(self._ring_bytes, dtype=torch.uint8, device=self.device)
            for _ in range(self._ring_slots)
        ]
        for buf in self._main_ring:
            self.manager.engine.register(buf.data_ptr(), self._ring_bytes)

    def _copy_into(self, ring, slot: int, src) -> tuple[int, int]:
        """Copy ``src``'s bytes into pre-registered ring ``slot``; return its
        (device pointer, byte length). Fails loud if an embedding exceeds a slot."""
        nbytes = src.numel() * src.element_size()
        if nbytes > self._ring_bytes:
            raise RuntimeError(
                f"EPD encode embedding {nbytes} B exceeds ring slot "
                f"{self._ring_bytes} B; raise TOKENSPEED_EPD_ENCODE_RING_SLOT_MB "
                "or the ring_bytes constructor argument"
            )
        buf = ring[slot]
        buf[:nbytes].view(src.dtype).copy_(src.reshape(-1))
        return buf.data_ptr(), nbytes

    def _lease_slot(self) -> "int | None":
        """Return a reusable ring-slot index, or ``None`` if every slot still
        holds an in-flight transfer. NON-BLOCKING: the caller DEFERS rather than
        spinning (a busy-wait would GIL-starve the daemon transfer-workers that
        mark rooms terminal and deadlock the single-threaded loop).

        A slot is reusable once the room it last staged is TERMINAL (Success,
        Failed, or None=already reaped) AND no parked chunk still holds its
        pointer -- the overwrite-safety invariant."""
        mgr = self.manager
        n = self._ring_slots
        for _ in range(n):
            slot = self._ring_idx % n
            self._ring_idx += 1
            room = self._slot_rooms[slot]
            if room is None:
                return slot
            status = mgr.room_status(room)
            if status is None or status in (
                TransferPoll.Success,
                TransferPoll.Failed,
            ):
                if not mgr.is_parked(room):
                    return slot
        return None

    def _stage_and_send(self, items: list[tuple[str, MultimodalDataItem]]) -> None:
        """Lease a ring slot per item and ship it; items that cannot lease a free
        slot (ring full) are DEFERRED for a later non-blocking retry rather than
        blocking the loop. Stages every leased item then issues ONE stream sync
        before the sends, so the one-sided RDMA reads never race the device-to-
        device copies (the ViT->send corruption hazard)."""
        self._ensure_rings()
        staged = []
        for rid, item in items:
            if rid not in self.senders:
                # Sender reaped (its room concluded/failed) -- drop this stale
                # deferred send instead of crashing on senders[rid].
                continue
            slot = self._lease_slot()
            if slot is None:
                self._deferred_sends.append((rid, item))
                continue
            try:
                send_args = self._stage_item(
                    item, self.senders[rid].bootstrap_room, slot
                )
            except Exception as e:
                # A staging error (most plausibly _copy_into rejecting an embedding
                # larger than a ring slot) must fail only THIS item's room, never
                # raise out of the single-threaded encode loop into the engine's
                # SIGUSR1 handler (which kills the whole worker and every other
                # in-flight image). Covers the unguarded send_item() /
                # drain_deferred() callers too; the leased slot returns to the ring
                # once the room is Failed (see _lease_slot).
                self._fail_staged_room(rid, e)
                continue
            staged.append((rid, send_args))
        if not staged:
            return
        # Record ONE CUDA event after all the ring copies above (they ran on the
        # current stream inside _stage_item) and hand it to each transfer rather
        # than host-syncing on this single encode-loop thread. The daemon
        # transfer-worker waits the event before its one-sided RDMA read
        # (embedding_transfer._transfer_worker), so the read never races the copy;
        # _lease_slot keeps the slot until its room is terminal (Success only after
        # the RDMA completes).
        copy_event = None
        if torch.cuda.is_available():
            copy_event = torch.cuda.Event()
            copy_event.record()
        for rid, send_args in staged:
            self.senders[rid].send(copy_event=copy_event, **send_args)

    def drain_deferred(self) -> None:
        """Retry deferred sends (ViT done, waiting for a free ring slot). Non-
        blocking: items that still cannot lease a slot stay deferred. Driven once
        per encode-loop tick; the loop yields the GIL between ticks so the daemon
        transfer-workers can free slots for the next drain."""
        if not self._deferred_sends:
            return
        pending = self._deferred_sends
        self._deferred_sends = []
        self._stage_and_send(pending)

    def has_deferred(self) -> bool:
        return bool(self._deferred_sends)

    def _conclude_room_failed(self, room: int, exc: Exception) -> None:
        """Push Failed to ``room``'s prefill receivers so they abort via the
        rank-synced admission path, instead of the error escaping the encode loop
        and SIGUSR1-ing the worker. The single seam every failure path goes
        through; delegates to the manager's public ``fail_room`` rather than
        reaching into its ``transfer_infos`` / status FSM."""
        self.manager.fail_room(room, str(exc))

    def _fail_staged_room(self, rid: str, exc: Exception) -> None:
        """Per-item staging failure (the unguarded ``send_item`` /
        ``drain_deferred`` callers): conclude ``rid``'s room Failed."""
        sender = self.senders.get(rid)
        if sender is None:
            return
        self._conclude_room_failed(sender.bootstrap_room, exc)
        logger.error(
            "encode staging failed for room %s: %s", sender.bootstrap_room, exc
        )

    def fail_rooms(self, request_ids, exc: Exception) -> int:
        """Conclude every room owned by ``request_ids`` Failed; return the count.
        The owning seam for a batch-level failure that fired before any send was
        issued (ViT / assign_encoded_embeddings): the worker hands its batch's
        request_ids and stays out of the sender/manager internals. Rooms are
        de-duped (a multi-image request shares one room)."""
        rooms = set()
        for rid in request_ids:
            sender = self.senders.get(rid)
            if sender is not None:
                rooms.add(sender.bootstrap_room)
        for room in rooms:
            self._conclude_room_failed(room, exc)
        return len(rooms)

    def reap_concluded_senders(self, pending_request_ids) -> None:
        """Drop per-request senders whose room reached a terminal transfer status
        (the ``senders`` dict otherwise grows forever). Senders whose request_id
        is still awaiting the tower (``pending_request_ids``) are kept -- their
        send has not been queued. Only the sender is dropped; the manager's
        terminal ``request_status`` tombstone stays (the transfer worker's
        straggler-drop and the ring-slot lease both key on it)."""
        for rid in list(self.senders):
            if rid in pending_request_ids:
                continue
            room = self.senders[rid].bootstrap_room
            if self.manager.room_status(room) in (
                TransferPoll.Success,
                TransferPoll.Failed,
            ):
                self.senders.pop(rid, None)

    def _stage_item(self, item: MultimodalDataItem, room, slot: int) -> dict:
        """Copy one item's embedding (and deepstack half, if any) into the leased
        ring ``slot`` and return the scalar ``send`` kwargs. The copy runs on the
        current stream; the CALLER must synchronize before handing these pointers
        to the transfer engine, so the one-sided RDMA read never races the device-
        to-device copy (same hazard class as the ViT->send race)."""
        enc = item.encoded
        self._slot_rooms[slot] = room
        send_ptr, nbytes = self._copy_into(self._main_ring, slot, enc)
        ds_ptr = ds_width = ds_nbytes = 0
        deep = item.encoded_deepstack
        if deep is not None and deep.numel() > 0:
            if self._deep_ring is None:
                self._deep_ring = [
                    torch.empty(self._ring_bytes, dtype=torch.uint8, device=self.device)
                    for _ in range(self._ring_slots)
                ]
                for buf in self._deep_ring:
                    self.manager.engine.register(buf.data_ptr(), self._ring_bytes)
            ds_width = deep.shape[1]
            ds_ptr, ds_nbytes = self._copy_into(self._deep_ring, slot, deep)
        return dict(
            src_embedding_ptr=send_ptr,
            n_tokens=enc.shape[0],
            hidden=enc.shape[1],
            dtype=str(enc.dtype),
            nbytes=nbytes,
            src_deepstack_ptr=ds_ptr,
            deepstack_width=ds_width,
            deepstack_nbytes=ds_nbytes,
        )

    def send_item(self, request_id, item: MultimodalDataItem) -> None:
        """Ship an already-encoded item (``item.encoded`` set) to its prefill peer.
        Used by the encode loop for cache hits, which skip the tower but still
        transfer. Routes through the same lease-or-defer path as ``execute`` so a
        full ring defers (non-blocking) instead of stalling the loop."""
        self._stage_and_send([(request_id, item)])
