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

"""EPD, a multimodal prefill node's first stage: the device side of receiving
each item's embedding from an encode node.

The host's embedding receiver (``ts-pd``'s ``epd::receiver``) contacts the
encode ranks, leases each item a slot of the receive pools, tells the encode
rank where to write, and agrees with the replica when a request's embeddings
all landed. Python owns the pools' memory and two data-plane steps:

* publish, on the forward thread before any forward of the request: copy each
  landed item's rows out of its slot onto the item (``item.encoded`` and
  ``item.encoded_deepstack``), which the prefill ``VisionEmbedder`` takes in
  place of running the vision tower. With row sharding each attention-TP rank
  received only its share of an image's rows (``shard_rows``); the ranks
  rebuild the rest with broadcasts, issued from the forward thread in the
  same order on every rank, like the model's collectives.
* release, on the control thread: the slots whose copies finished, which the
  receiver leases again.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from tokenspeed.runtime.epd.recv_pool import (
    is_epd_prefill_node,
    recv_pool_geometry,
    split_slot,
)
from tokenspeed.runtime.multimodal.inputs import MultimodalDataItem
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)

# One landed item, as the host hands it over: the item's index among the
# request's multimodal items, the slot holding its rows (None for an empty
# shard), the first of the image's rows this rank received, how many, the
# image's rows, and whether the rows are this rank's shard.
LandedItem = tuple[int, int | None, int, int, int, bool]


def shard_rows(span: int, shard_rank: int, shard_size: int) -> tuple[int, int]:
    """A balanced contiguous row shard of ``span`` rows over ``shard_size``
    ranks: this rank's ``(row_start, row_count)``. The first
    ``span % shard_size`` ranks get one extra row; the shards tile
    ``[0, span)`` in rank order. The host's receiver asks the encode rank for
    the same shard (``ts-pd``'s ``epd::shard_rows``). ``row_count`` may be 0
    when ``span < shard_size``."""
    base, rem = divmod(span, shard_size)
    start = shard_rank * base + min(shard_rank, rem)
    count = base + (1 if shard_rank < rem else 0)
    return start, count


@dataclass(frozen=True)
class EpdReceiveSettings:
    """What the host starts its embedding receiver from (``ts-host``'s
    ``receiver_config``).

    Attributes:
        slots: Receive slots; 0 when the pool is disabled, which fails every
            item.
        slot_bytes: Bytes of one slot of the rows' pool.
        pool_base: The rows' pool's address.
        deepstack_slot_bytes: Bytes of one slot of the deepstack rows' pool;
            0 for a model without deepstack.
        deepstack_base: The deepstack rows' pool's address; 0 without one.
        tp_size: The attention-TP size.
        tp_rank: This rank's attention-TP rank.
        hidden: Elements per embedding row.
        num_deepstack: Deepstack levels per row, 0 for a model without
            deepstack.
        dtype: The rows' dtype as the encode node checks it
            (``str(torch.dtype)``).
        element_size: Bytes per element.
        shard: Each rank receives only its share of an image's rows.
        timeout_s: How long a request may wait for its embeddings
            (``TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT``).
    """

    slots: int
    slot_bytes: int
    pool_base: int
    deepstack_slot_bytes: int
    deepstack_base: int
    tp_size: int
    tp_rank: int
    hidden: int
    num_deepstack: int
    dtype: str
    element_size: int
    shard: bool
    timeout_s: float


class ReceivePools:
    """The receive pools, one device buffer: an item's rows land in a slot of
    the rows' pool and, for a model with deepstack, its deepstack rows in the
    same slot of the deepstack pool, which follows it.

    The buffer lives as long as the process, and the host registers it with
    its transfer engine once: a buffer registered per request would recycle
    addresses under new keys while the encode node still holds the old ones.
    """

    def __init__(
        self, *, slots: int, slot_bytes: int, num_deepstack: int, device: Any
    ) -> None:
        self.slots = slots
        self.slot_bytes, self.deepstack_slot_bytes = split_slot(
            slot_bytes, num_deepstack
        )
        self.buffer = torch.empty(
            slots * (self.slot_bytes + self.deepstack_slot_bytes),
            dtype=torch.uint8,
            device=device,
        )

    @property
    def base(self) -> int:
        return self.buffer.data_ptr()

    @property
    def deepstack_base(self) -> int:
        return (
            self.base + self.slots * self.slot_bytes if self.deepstack_slot_bytes else 0
        )

    def rows(
        self, slot: int, count: int, width: int, dtype: torch.dtype, *, deepstack: bool
    ) -> torch.Tensor:
        """The ``count`` rows of ``width`` elements at the start of ``slot``
        of the rows' or the deepstack pool."""
        if deepstack:
            start = self.slots * self.slot_bytes + slot * self.deepstack_slot_bytes
        else:
            start = slot * self.slot_bytes
        nbytes = count * width * dtype.itemsize
        return self.buffer[start : start + nbytes].view(dtype).view(count, width)


class ShardGroup:
    """The attention-TP ranks that rebuild an image from their row shards."""

    def __init__(self, *, nccl_group: Any, group_ranks: Sequence[int]) -> None:
        self.nccl_group = nccl_group
        # Global ranks, which ``dist.broadcast`` names its source by.
        self.group_ranks = tuple(group_ranks)

    def rebuild(self, embedding: torch.Tensor) -> None:
        """Fill every rank's shard of ``embedding``'s rows from that rank."""
        size = len(self.group_ranks)
        for rank, src in enumerate(self.group_ranks):
            start, count = shard_rows(embedding.shape[0], rank, size)
            if count:
                dist.broadcast(
                    embedding[start : start + count], src=src, group=self.nccl_group
                )


class EmbeddingPublisher:
    """The data plane's side of the receive: publishing landed rows onto
    their items, and reporting the slots whose rows were copied out."""

    def __init__(
        self,
        pools: ReceivePools | None,
        *,
        hidden: int,
        num_deepstack: int,
        dtype: torch.dtype,
        device: Any,
        shards: ShardGroup | None,
    ) -> None:
        self._pools = pools
        self._hidden = hidden
        self._num_deepstack = num_deepstack
        self._dtype = dtype
        self._device = torch.device(device)
        self._shards = shards
        # (the event after a publish's copies, their slots), in publish
        # order: appended by the forward thread, read by the control thread.
        self._copies: list[tuple[torch.cuda.Event | None, list[int]]] = []
        self._lock = threading.Lock()

    def publish(
        self,
        landed: Sequence[tuple[Sequence[MultimodalDataItem], Sequence[LandedItem]]],
    ) -> None:
        """Put each landed item's rows onto it: ``landed`` holds one request's
        multimodal items and its landed items each. Every rank of the
        replica publishes the same requests in the same order, so with row
        sharding the ranks' rebuild broadcasts match."""
        slots = []
        rebuilt = []
        for items, landed_items in landed:
            for index, slot, row_start, row_count, rows, sharded in landed_items:
                item = items[index]
                item.encoded = self._take(
                    slot, row_start, row_count, rows, self._hidden, deepstack=False
                )
                item.encoded_deepstack = (
                    self._take(
                        slot,
                        row_start,
                        row_count,
                        rows,
                        self._hidden * self._num_deepstack,
                        deepstack=True,
                    )
                    if self._num_deepstack
                    else None
                )
                if slot is not None:
                    slots.append(slot)
                if sharded:
                    rebuilt.append(item)
        if slots:
            event = None
            if self._device.type == "cuda":
                event = torch.cuda.Event()
                event.record()
            with self._lock:
                self._copies.append((event, slots))
        for item in rebuilt:
            self._shards.rebuild(item.encoded)
            if item.encoded_deepstack is not None:
                self._shards.rebuild(item.encoded_deepstack)

    def _take(
        self,
        slot: int | None,
        row_start: int,
        row_count: int,
        rows: int,
        width: int,
        *,
        deepstack: bool,
    ) -> torch.Tensor:
        """A copy of an item's ``rows`` rows with the landed ones in place;
        a shard's other rows are left for the rebuild."""
        embedding = torch.empty((rows, width), dtype=self._dtype, device=self._device)
        if slot is not None:
            embedding[row_start : row_start + row_count] = self._pools.rows(
                slot, row_count, width, self._dtype, deepstack=deepstack
            )
        return embedding

    def released(self) -> list[int]:
        """The slots whose rows ``publish`` finished copying out, once each."""
        done = []
        with self._lock:
            pending = []
            for event, slots in self._copies:
                if event is None or event.query():
                    done.extend(slots)
                else:
                    pending.append((event, slots))
            self._copies = pending
        return done


@dataclass(frozen=True)
class EpdReceive:
    """An EPD prefill node's first stage's receive: the host's receiver
    settings and the data plane's publisher."""

    settings: EpdReceiveSettings
    publisher: EmbeddingPublisher


def _agreed_shard_flag(attn_tp_size: int, attn_tp_cpu_group: Any) -> bool:
    """``TOKENSPEED_EPD_EMBEDDING_SHARD``, checked equal on every attention-TP
    rank: torn across ranks (set on one node of a multi-node prefill), the
    ranks would disagree on which rows they ask for and on the rebuild
    collectives."""
    if attn_tp_size <= 1:
        return False
    flag = bool(envs.TOKENSPEED_EPD_EMBEDDING_SHARD.get())
    agreed = torch.tensor([int(flag), -int(flag)], dtype=torch.int32)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=attn_tp_cpu_group)
    if agreed[0].item() != -agreed[1].item():
        raise RuntimeError(
            "TOKENSPEED_EPD_EMBEDDING_SHARD differs across attn-TP ranks; "
            "set it identically on every node of the prefill engine"
        )
    return flag


def build_epd_receive(
    server_args,
    *,
    model_config,
    encoder_model_facts: Callable[[], Any],
    mapping,
    attn_tp_rank: int,
    attn_tp_size: int,
    attn_tp_cpu_group: Any,
    pg_manager,
) -> EpdReceive | None:
    """The receive of a multimodal prefill node's first stage, None elsewhere
    (a later pipeline stage receives nothing and follows the first's
    decisions). Allocates the receive pools, which the cache budget left out
    (``recv_pool_bytes``)."""
    if not is_epd_prefill_node(server_args, model_config.is_multimodal_active):
        return None
    # Resolved only past the gate: reading the vision tower's dtype raises on
    # a text-only model, and every text-only PD node calls this.
    facts = encoder_model_facts()
    shards = None
    if _agreed_shard_flag(attn_tp_size, attn_tp_cpu_group):
        shards = ShardGroup(
            nccl_group=pg_manager.get_process_group("nccl", mapping.attn.tp_group),
            group_ranks=mapping.attn.tp_group,
        )
        # NCCL sets a communicator up on its first collective: warm it now, so
        # the first request does not pay for it and a broken group fails the
        # boot instead of a request.
        warmup = torch.zeros(1, device=facts.device)
        dist.broadcast(warmup, src=shards.group_ranks[0], group=shards.nccl_group)
        torch.cuda.current_stream().synchronize()
        logger.info(f"EPD embedding row sharding over {attn_tp_size} attn-TP ranks")
    slots, slot_mb = recv_pool_geometry()
    pools = None
    if slots:
        pools = ReceivePools(
            slots=slots,
            slot_bytes=slot_mb << 20,
            num_deepstack=facts.num_deepstack,
            device=facts.device,
        )
        logger.info(
            f"EPD receive pool: {slots} slots x {slot_mb} MB "
            f"({pools.slot_bytes} B rows, {pools.deepstack_slot_bytes} B deepstack)"
        )
    settings = EpdReceiveSettings(
        slots=slots,
        slot_bytes=pools.slot_bytes if pools is not None else 0,
        pool_base=pools.base if pools is not None else 0,
        deepstack_slot_bytes=pools.deepstack_slot_bytes if pools is not None else 0,
        deepstack_base=pools.deepstack_base if pools is not None else 0,
        tp_size=attn_tp_size,
        tp_rank=attn_tp_rank,
        hidden=facts.hidden,
        num_deepstack=facts.num_deepstack,
        dtype=str(facts.dtype),
        element_size=facts.dtype.itemsize,
        shard=shards is not None,
        timeout_s=float(envs.TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT.get()),
    )
    publisher = EmbeddingPublisher(
        pools,
        hidden=facts.hidden,
        num_deepstack=facts.num_deepstack,
        dtype=facts.dtype,
        device=facts.device,
        shards=shards,
    )
    return EpdReceive(settings=settings, publisher=publisher)


def assert_embeddings_received(
    publisher: EmbeddingPublisher | None, multimodal_context
) -> None:
    """EPD invariant: every handshaked item a forward scatters holds its
    embedding, published before the request was scheduled. One that does not
    leaked past the receive: fail loud instead of running the tower or
    publishing shard-only rows. An item whose tokens an earlier chunk
    prefilled no longer holds its encoding (the embedder drops it once
    nothing reads it) and is not scattered again, so only items overlapping
    this chunk count. A no-op off an EPD prefill node's first stage.
    """
    if (
        publisher is None
        or multimodal_context is None
        or not multimodal_context.has_extend_inputs()
    ):
        return
    for row, mm in enumerate(multimodal_context.mm_inputs):
        if mm is None or row >= len(multimodal_context.extend_seq_lens):
            continue
        chunk_start = multimodal_context.extend_prefix_lens[row]
        chunk_end = chunk_start + multimodal_context.extend_seq_lens[row]
        missing = [
            i
            for i, item in enumerate(mm.mm_items)
            if item.encode_handshake is not None
            and item.encoded is None
            and any(
                start < chunk_end and chunk_start <= end
                for start, end in item.offsets or ()
            )
        ]
        if missing:
            raise RuntimeError(
                f"EPD: handshaked items {missing} reached the prefill forward "
                "un-received; their embeddings must land before the request "
                "is scheduled"
            )
