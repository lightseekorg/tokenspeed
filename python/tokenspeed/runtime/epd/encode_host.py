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

"""An EPD encode rank: the Python half of the host's encode mode.

The encode node runs the vision tower only and ships each item's embedding
to the prefill ranks waiting for it. The host (``ts-host``) runs the rest:
the first rank takes the frontend's requests and relays them across the
attention-TP group, the loop (``ts-control``'s ``encode``) batches them and
leases each item a slot of the send ring, and the sender (``ts-pd``'s
``epd::sender``) writes the rows to the prefill ranks. Python owns the vision
tower, the embedding cache and the ring's memory. Its data plane, on the
forward thread:

* ``admit``: take requests, open their shared-memory pixels on every rank
  before any rank consumes them, and look their items up in the cache;
* ``encode``: run the tower on a batch's other items, cache the results, and
  copy each item's rows into its send slot;

and, as a completion query on the control thread, ``encoded``: the items
whose copies finished, which the host then hands to the sender.
"""

from __future__ import annotations

import faulthandler
import logging
import threading
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import torch
import torch.distributed as dist

from tokenspeed.runtime.cache.embedding_cache import (
    EmbeddingCache,
    TieredEmbeddingCache,
)
from tokenspeed.runtime.engine.io_struct import MsgpackDecoder
from tokenspeed.runtime.epd.encode_worker import EncodeRequest
from tokenspeed.runtime.multimodal.embedder import _item_token_count
from tokenspeed.runtime.multimodal.inputs import MultimodalDataItem
from tokenspeed.runtime.pd.host_peer import (
    enable_mooncake_ipv6_for_host,
    mooncake_library,
    mooncake_protocol,
)
from tokenspeed.runtime.utils import configure_logger
from tokenspeed.runtime.utils.env import envs, global_server_args_dict_update
from tokenspeed.runtime.utils.jit_compile_check import (
    install_jit_compile_check,
    mark_jit_compile_serving,
)
from tokenspeed.runtime.utils.network import (
    get_ip,
    get_local_ip_by_remote,
    join_host_port,
)
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

# A send slot's bytes without TOKENSPEED_EPD_ENCODE_RING_SLOT_MB.
_DEFAULT_RING_SLOT_BYTES = 256 << 20


def assign_encoded_embeddings(
    items: list[MultimodalDataItem], output: torch.Tensor, model
) -> None:
    """Scatter a packed vision-tower output onto each item, in place.

    ``output`` is the tower's ``[sum_tokens, width]`` result for ``items`` in
    order (``width = hidden``, or ``hidden * (1 + n_deepstack)`` for
    deepstack models like Qwen3.5). Each item's row span is its post-merge
    token count (``_item_token_count``); for deepstack models the rows are
    column-split by ``model.separate_deepstack_embeds`` into the main
    ``[N, hidden]`` and deepstack ``[N, hidden * n_deepstack]`` halves. The
    results are contiguous: a TP-gathered tower output may not be, and the
    transfer ships raw row-major bytes.

    Sets ``item.encoded`` and ``item.encoded_deepstack`` (None without
    deepstack), the form the prefill ``VisionEmbedder`` takes in place of
    running the tower.
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
    for item, emb in zip(items, torch.split(output, per_item_tokens, dim=0)):
        if has_deepstack:
            main, deep = model.separate_deepstack_embeds(emb)
            item.encoded = main.contiguous()
            item.encoded_deepstack = deep.contiguous()
        else:
            item.encoded = emb.contiguous()
            item.encoded_deepstack = None


def embedding_cache_bytes(env_field) -> int:
    """A whole-MiB cache capacity knob, in bytes; negative is refused."""
    mb = env_field.get()
    if mb < 0:
        raise ValueError(f"{env_field.name} must be >= 0 MiB, got {mb}")
    return mb * 1024 * 1024


def make_embedding_cache(l1_bytes: int, l2_bytes: int, device: str):
    """The embedding cache: VRAM only, or with a host-DRAM tier catching the
    VRAM tier's evictions when ``l2_bytes > 0``.

    Both capacities are per encode rank: at TP>1 every rank keeps its own.
    """
    if l2_bytes > 0:
        logger.info(
            f"EPD encode embedding cache: L1(VRAM)={l1_bytes >> 20:d} MiB, L2(host "
            f"DRAM)={l2_bytes >> 20:d} MiB",
        )
        return TieredEmbeddingCache(l1_bytes, l2_bytes, device=device)
    logger.info(
        f"EPD encode embedding cache: L1(VRAM)={l1_bytes >> 20:d} MiB (host-DRAM L2 "
        "disabled)",
    )
    return EmbeddingCache(l1_bytes)


class SendRing:
    """The send ring: equal slots the transfer engine reads each item's rows
    (and deepstack rows) from, registered by the host's sender once.

    Every item goes through a slot: the transfer engine needs its sources
    registered, and registering each item's own tensor would overlap the
    caching allocator's reused ranges.
    """

    def __init__(
        self, *, slots: int, slot_bytes: int, deepstack: bool, device: Any
    ) -> None:
        self.slots = slots
        self.slot_bytes = slot_bytes
        self.buffer = torch.empty(slots * slot_bytes, dtype=torch.uint8, device=device)
        self.deepstack_buffer = (
            torch.empty(slots * slot_bytes, dtype=torch.uint8, device=device)
            if deepstack
            else None
        )

    def regions(self) -> tuple[tuple[int, int], ...]:
        """The buffers as ``(address, bytes)``."""
        buffers = [self.buffer]
        if self.deepstack_buffer is not None:
            buffers.append(self.deepstack_buffer)
        return tuple((buffer.data_ptr(), buffer.numel()) for buffer in buffers)

    def _copy(self, buffer: torch.Tensor, slot: int, rows: torch.Tensor) -> int:
        nbytes = rows.numel() * rows.element_size()
        if nbytes > self.slot_bytes:
            raise ValueError(
                f"EPD encode embedding {nbytes} B exceeds the send slot's "
                f"{self.slot_bytes} B; raise TOKENSPEED_EPD_ENCODE_RING_SLOT_MB"
            )
        start = slot * self.slot_bytes
        buffer[start : start + nbytes].view(rows.dtype).copy_(rows.reshape(-1))
        return nbytes

    def copy_into(self, slot: int, item: MultimodalDataItem) -> tuple:
        """Copy ``item``'s rows (and deepstack rows) into ``slot``, on the
        current stream; ``(rows, hidden, dtype, bytes, deepstack width,
        deepstack bytes)``."""
        encoded = item.encoded
        nbytes = self._copy(self.buffer, slot, encoded)
        deep_width = deep_nbytes = 0
        deep = item.encoded_deepstack
        if deep is not None and deep.numel() > 0:
            if self.deepstack_buffer is None:
                raise ValueError("a deepstack embedding without a deepstack ring")
            deep_width = deep.shape[1]
            deep_nbytes = self._copy(self.deepstack_buffer, slot, deep)
        return (
            encoded.shape[0],
            encoded.shape[1],
            str(encoded.dtype),
            nbytes,
            deep_width,
            deep_nbytes,
        )


@dataclass(frozen=True)
class EncodeSettings:
    """What the host's encode mode starts from (``ts-host``'s ``encode``).

    Attributes:
        rank: This rank.
        tp_ranks: The attention-TP group, in rank order; its first rank takes
            the frontend's requests and hosts the bootstrap server.
        collective_bind_host: Where the rank listens for the other ranks'
            collective connections.
        intake: On the first rank, where the frontend pushes encode requests
            (``scheduler_input_ipc_name``); else None.
        bootstrap_port: On the first rank, the bootstrap server's port; else
            None.
        bootstrap_url: The bootstrap server's ``/route``, which every rank
            registers with.
        library: The Mooncake library with the TokenSpeed C entry points.
        local_ip: Where prefill ranks reach this rank.
        protocol: The Mooncake transport selected for this rank's device.
        ib_device: The RDMA devices to use, comma-separated; empty for all.
        bootstrap_timeout_s: How long a handed-over item waits for its
            prefill ranks to ask (``TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT``).
        transfer_workers: Rooms written concurrently
            (``TOKENSPEED_DISAGGREGATION_QUEUE_SIZE``).
        ring_slots: The send ring's slots.
        ring_slot_bytes: A slot's bytes, in either buffer.
        ring_base: The rows' buffer.
        ring_deepstack_base: The deepstack rows' buffer; 0 without
            deepstack.
        max_batch_tokens: The most vision tokens a forward embeds.
        max_batch_items: The most items a forward embeds.
    """

    rank: int
    tp_ranks: tuple[int, ...]
    collective_bind_host: str
    intake: str | None
    bootstrap_port: int | None
    bootstrap_url: str
    library: str
    local_ip: str
    protocol: str
    ib_device: str
    bootstrap_timeout_s: float
    transfer_workers: int
    ring_slots: int
    ring_slot_bytes: int
    ring_base: int
    ring_deepstack_base: int
    max_batch_tokens: int
    max_batch_items: int


class EncodeDataPlane:
    """An encode rank's device side, which the host's forward thread and
    control thread call.

    Every rank of the attention-TP group gets the same calls in the same
    order, so the ranks' caches answer alike and their batches match.
    """

    def __init__(
        self, *, model, cache, ring: SendRing, device: Any, tp_cpu_group
    ) -> None:
        self._model = model
        self._cache = cache
        self._ring = ring
        self._device = torch.device(device)
        # The attention-TP group the shared-memory pixels are opened over;
        # None with one rank.
        self._tp_cpu_group = tp_cpu_group
        self._decoder = MsgpackDecoder(EncodeRequest)
        # Admission -> the request's item, until it is encoded or released.
        self._items: dict[int, MultimodalDataItem] = {}
        # (the event after a batch's ring copies, its items' outcomes), in
        # batch order: appended by the forward thread, read by the control
        # thread.
        self._copies: list[tuple[torch.cuda.Event | None, list[tuple]]] = []
        self._lock = threading.Lock()

    def enter_forward_thread(self) -> None:
        """Called once by the host's forward thread, before its first
        operation."""
        threading.current_thread().name = "tokenspeed::forward"
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

    def admit(self, requests: list[tuple[int, list[bytes]]]) -> list[bool]:
        """Take encode requests ``(admission, wire frames)``; say for each
        whether the cache holds its item.

        A request carries one item; one that does not is still taken, so its
        shared-memory pixels are consumed, and the host releases it.
        """
        decoded = [
            (admission, self._decoder.decode(frames)) for admission, frames in requests
        ]
        handles = [
            item.feature_shm
            for _, request in decoded
            for item in request.items
            if item.feature_shm is not None
        ]
        for handle in handles:
            handle.attach()
        if handles and self._tp_cpu_group is not None:
            # Consuming a segment unlinks it, even on a cache hit: every rank
            # opens all of them before any rank consumes one.
            dist.barrier(group=self._tp_cpu_group)
        cached = []
        for admission, request in decoded:
            single = len(request.items) == 1
            hit = False
            for item in request.items:
                entry = (
                    self._cache.get(item.hash)
                    if single and item.hash is not None
                    else None
                )
                if item.feature_shm is not None:
                    handle, item.feature_shm = item.feature_shm, None
                    if entry is None:
                        item.feature = handle.consume()
                    else:
                        handle.consume()
                if entry is not None:
                    # The (rows, deepstack rows) pair: both halves go out.
                    item.encoded, item.encoded_deepstack = entry
                    hit = True
            if single:
                self._items[admission] = request.items[0]
            cached.append(hit)
        return cached

    def release(self, admissions: list[int]) -> None:
        """Forget admitted requests that will not be encoded."""
        for admission in admissions:
            self._items.pop(admission, None)

    def _feature_fn(self, modality):
        get_specs = getattr(self._model, "get_multimodal_encoder_specs", None)
        if not callable(get_specs):
            raise ValueError("model does not register multimodal encoders")
        spec = get_specs().get(modality)
        if spec is None:
            raise ValueError(f"unsupported modality for encode: {modality}")
        return spec.fn

    def encode(self, batch: list[tuple[int, int]]) -> None:
        """Embed a batch ``(admission, slot)``: run the tower on the items
        the cache did not hold, cache them, and copy every item's rows into
        its slot. An item that fails, or a tower failure, fails only its own
        rooms."""
        entries = [
            (admission, slot, self._items.pop(admission)) for admission, slot in batch
        ]
        misses = [item for _, _, item in entries if item.encoded is None]
        failure = None
        try:
            by_modality: dict = {}
            for item in misses:
                by_modality.setdefault(item.modality, []).append(item)
            with torch.inference_mode():
                for modality, items in by_modality.items():
                    output = self._feature_fn(modality)(items)
                    assign_encoded_embeddings(items, output, self._model)
        except Exception as error:  # noqa: BLE001 - fails this batch's rooms only
            logger.exception("an EPD encode batch failed")
            failure = f"the vision tower failed: {error}"
        outcomes = []
        for admission, slot, item in entries:
            if failure is not None:
                outcomes.append((admission, failure))
                continue
            try:
                outcomes.append((admission, self._ring.copy_into(slot, item)))
            except ValueError as error:
                outcomes.append((admission, str(error)))
        if failure is None:
            for item in misses:
                if item.hash is None:
                    continue
                # Cache the pair: a later hit ships the deepstack rows too.
                deep = item.encoded_deepstack
                nbytes = item.encoded.numel() * item.encoded.element_size()
                if deep is not None:
                    nbytes += deep.numel() * deep.element_size()
                self._cache.put(item.hash, (item.encoded, deep), nbytes)
        event = None
        if self._device.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        with self._lock:
            self._copies.append((event, outcomes))

    def encoded(self) -> list[tuple]:
        """The items whose rows reached their slots since the last call:
        ``(admission, failure or None, rows, hidden, dtype, bytes, deepstack
        width, deepstack bytes)`` each; a completion query, on the control
        thread."""
        done = []
        with self._lock:
            pending = []
            for event, outcomes in self._copies:
                if event is None or event.query():
                    done.extend(outcomes)
                else:
                    pending.append((event, outcomes))
            self._copies = pending
        return [
            (
                (admission, outcome, 0, 0, "", 0, 0, 0)
                if isinstance(outcome, str)
                else (admission, None, *outcome)
            )
            for admission, outcome in done
        ]


class EncodeBootstrap:
    """One encode rank, built: the settings the host starts from and the
    data plane it calls."""

    #: Which host mode this rank runs (``HostBootstrap`` serves a language
    #: model).
    mode = "encode"

    def __init__(
        self, server_args: ServerArgs, port_args: PortArgs, connection: Connection
    ) -> None:
        from tokenspeed.runtime.configs.model_config import ModelConfig
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )
        from tokenspeed.runtime.execution.distributed_initializer import (
            DistributedConfig,
            DistributedInitializer,
        )
        from tokenspeed.runtime.execution.factory import create_model_runner

        self._server_args = server_args
        self._connection = connection
        mapping = server_args.mapping
        if mapping.attn.dp_size != 1:
            raise ValueError(
                "disaggregation_mode=encode currently supports encode "
                f"data_parallel_size == 1, got {mapping.attn.dp_size}"
            )
        global_rank = mapping.rank
        attn_tp_rank = mapping.attn.tp_rank
        gpu_id = mapping.rank % mapping.nprocs_per_node + server_args.base_gpu_id
        device = f"cuda:{gpu_id}"
        faulthandler.enable()
        configure_logger(server_args, prefix=f" ATTN TP RANK {attn_tp_rank}")
        install_jit_compile_check()

        model_config = ModelConfig(
            server_args.model,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.max_model_len,
            model_override_args=server_args.hf_overrides,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
            server_args=server_args,
        )
        # The communication backend's probes read the resolved launch from
        # this dict; publish before the distributed init runs them.
        global_server_args_dict_update(server_args)
        DistributedInitializer.initialize(
            DistributedConfig.from_server_args(
                server_args=server_args,
                port_args=port_args,
                gpu_id=gpu_id,
                global_rank=global_rank,
                hidden_size=model_config.hidden_size,
                max_num_tokens=server_args.chunked_prefill_size or 8192,
            )
        )
        # Built vision-only (``disaggregation_mode == "encode"`` skips the
        # language model and its weights); no cache pool is allocated.
        model_runner = create_model_runner(
            server_args, model_config, None, gpu_id, global_rank
        )[0]
        model_runner.prepare_multimodal_runtime()
        model = model_runner.model
        self._model_config = model_config
        self._multimodal_encoder_dtype = model_runner.multimodal_encoder_dtype

        slot_mb = envs.TOKENSPEED_EPD_ENCODE_RING_SLOT_MB.get()
        ring = SendRing(
            slots=envs.TOKENSPEED_EPD_ENCODE_RING_SLOTS.get(),
            slot_bytes=slot_mb << 20 if slot_mb else _DEFAULT_RING_SLOT_BYTES,
            deepstack=getattr(model, "num_deepstack_embeddings", 0) > 0,
            device=device,
        )
        cache = make_embedding_cache(
            embedding_cache_bytes(envs.TOKENSPEED_EPD_ENCODE_EMBED_CACHE_MB),
            embedding_cache_bytes(envs.TOKENSPEED_EPD_ENCODE_EMBED_CACHE_DRAM_MB),
            device,
        )
        tp_ranks = tuple(mapping.attn.tp_group)
        tp_cpu_group = (
            pg_manager.get_process_group("gloo", mapping.attn.tp_group)
            if len(tp_ranks) > 1
            else None
        )
        self._world_cpu_group = pg_manager.get_process_group(
            "gloo", mapping.world_group
        )
        self._collective_host = "127.0.0.1" if mapping.nnodes == 1 else get_ip()

        local_ip = get_local_ip_by_remote()
        enable_mooncake_ipv6_for_host(local_ip)
        # Rank 0 hosts the node's bootstrap server; every rank registers with
        # it.
        bootstrap_host = (
            server_args.dist_init_addr.split(":", 1)[0]
            if server_args.dist_init_addr
            else local_ip
        )
        first = attn_tp_rank == 0
        regions = ring.regions()
        #: What the host starts from.
        self.config = EncodeSettings(
            rank=global_rank,
            tp_ranks=tp_ranks,
            collective_bind_host=self._collective_host,
            intake=port_args.scheduler_input_ipc_name if first else None,
            bootstrap_port=server_args.disaggregation_bootstrap_port if first else None,
            bootstrap_url=(
                f"http://{join_host_port(bootstrap_host, server_args.disaggregation_bootstrap_port)}"
                "/route"
            ),
            library=mooncake_library(),
            local_ip=local_ip,
            protocol=mooncake_protocol(),
            ib_device=server_args.disaggregation_ib_device or "",
            bootstrap_timeout_s=float(
                envs.TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
            ),
            transfer_workers=envs.TOKENSPEED_DISAGGREGATION_QUEUE_SIZE.get(),
            ring_slots=ring.slots,
            ring_slot_bytes=ring.slot_bytes,
            ring_base=regions[0][0],
            ring_deepstack_base=regions[1][0] if len(regions) > 1 else 0,
            max_batch_tokens=server_args.chunked_prefill_size or 8192,
            max_batch_items=server_args.max_num_seqs,
        )
        #: The data plane the host's threads call.
        self.data_plane = EncodeDataPlane(
            model=model,
            cache=cache,
            ring=ring,
            device=device,
            tp_cpu_group=tp_cpu_group,
        )

    def collective_addresses(self, port: int) -> list[str]:
        """Every rank's collective address, in rank order, given this rank's
        port."""
        gathered: list = [None] * self._world_cpu_group.size()
        dist.all_gather_object(
            gathered,
            join_host_port(self._collective_host, port),
            group=self._world_cpu_group,
        )
        return gathered

    def report_ready(self) -> None:
        """Tell the launcher the rank is serving. An encode rank has no
        cache, so the token fields only keep the ready message's shape."""
        server_args = self._server_args
        context_len = self._model_config.context_len
        self._connection.send(
            {
                "status": "ready",
                "max_total_num_tokens": 0,
                "max_req_input_len": context_len,
                "max_single_request_tokens": context_len,
                "max_num_seqs": server_args.max_num_seqs,
                "chunked_prefill_size": server_args.chunked_prefill_size,
                "max_model_len": context_len,
                "multimodal_encoder_dtype": self._multimodal_encoder_dtype,
                # No language model here: nothing scores prompt rows.
                "supports_prompt_logprobs": False,
            }
        )

    def enter_serving(self) -> None:
        """End startup: from here on a kernel compilation is a stall."""
        mark_jit_compile_serving()
